import threading
import time
from typing import List

import numpy as np
import torch
import hashlib
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.data.metadata import enumerate_and_hash_batches, verify_batch_metadata
from src.training.training_manager import TrainingManager
from src.model.llama_model import LlamaModel
from src.model.pythia_model import PythiaModel
from src.inference.inference_manager import InferenceManager
from src.data.loader import get_loader, RawDataset
from src.types.conversation import Chat, Role
from src.utils.logger import log
from config import Config
import grpc
from concurrent import futures
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedDataParallelKwargs
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardedStateDictConfig
from collections import namedtuple

from src.utils.profiler import Profiler

SampleMetadata = namedtuple('SampleMetadata', ['id', 'hash'])


class ModelNode:
    def __init__(self, config: Config, raw_dataset: RawDataset, is_reference: bool, system_prompt: str = "",
                 debug: bool = False):
        self.config = config
        self.is_reference = is_reference
        self.raw_dataset = raw_dataset
        self.system_prompt = system_prompt
        self.debug = debug

        if self.debug:
            log.info("Initializing ModelNode in debug mode")

        self.model = self.setup_model()
        self.tokenizer = self.model.get_tokenizer()
        self.dataloader = self.setup_dataloader()

        fsdp_plugin = None
        if self.config.fsdp.enabled:
            fsdp_plugin = FullyShardedDataParallelPlugin(
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                state_dict_config=ShardedStateDictConfig(),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                auto_wrap_policy=self.get_auto_wrap_policy(),
                forward_prefetch=True,
            )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            fsdp_plugin=fsdp_plugin
        )

        if not is_reference:
            self.optimizer = AdamW(self.model.parameters(), lr=config.training.learning_rate)
            self.scheduler = LambdaLR(self.optimizer,
                                      lr_lambda=lambda step: min(1.0, (step + 1) / config.training.warmup_steps))

            # Prepare everything
            self.model.prepare(self.accelerator)
            self.optimizer, self.scheduler, self.dataloader = self.accelerator.prepare(
                self.optimizer, self.scheduler, self.dataloader
            )

            self.training_manager = TrainingManager(config, self.model, self.optimizer, self.scheduler,
                                                    self.accelerator)
        else:
            self.model.prepare(self.accelerator)
            self.dataloader = self.accelerator.prepare(self.dataloader)
            self.inference_manager = InferenceManager(self.model, self.accelerator, self.debug)

    @Profiler.cuda()
    def setup_model(self):
        if self.debug:
            log.info(f"Setting up model: {self.config.model.name}")
            # Take snapshot before model loading
            Profiler.take_snapshot('before_model_load')
            Profiler.print_memory_stats()

        if self.config.model.name == "llama3":
            model = LlamaModel(self.config)
        elif self.config.model.name == "pythia-160m":
            model = PythiaModel(self.config)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model.name}")

        model.get_tokenizer().model_max_length = self.config.model.max_sequence_length

        log.info("Model setup completed.")
        if self.debug:
            Profiler.take_snapshot('after_model_load')
            Profiler.print_memory_stats()
        return model

    def setup_dataloader(self):
        loader = get_loader(self.config, self.tokenizer, self.raw_dataset)
        return loader.get_dataloader()

    def get_auto_wrap_policy(self):
        def auto_wrap_policy(module):
            return isinstance(module, (self.model.get_transformer_layer_class(),))
        return auto_wrap_policy

    @Profiler.cuda()
    def forward(self, batch):
        with self.accelerator.autocast():
            return self.inference_manager.forward(batch)

    @Profiler.cuda()
    def backward(self, reference_logits):
        if self.is_reference:
            raise ValueError("Backward not supported for reference model")

        if self.last_forward is None:
            raise ValueError("No matching forward pass found for this backward request")

        policy_logits = self.last_forward['output']
        reference_logits = torch.from_numpy(reference_logits).to(self.device).reshape_as(policy_logits)

        loss = self.training_manager.backward(policy_logits, reference_logits, self.last_forward['batch'])

        return loss

    @Profiler.cuda()
    def run_inference(self):
        if self.debug:
            log.info("Starting inference")
        self.model.eval()
        all_outputs = []

        with torch.no_grad():
            for item in self.dataloader:
                chats = [Chat(role=Role.user.value, message=item[0]), Chat(role=Role.assistant.value, message="")]
                tokenized = self.model.tokenize(chats, self.system_prompt)
                input_ids = tokenized['input_ids'].to(self.accelerator.device)
                attention_mask = tokenized['attention_mask'].to(self.accelerator.device)
                generated = self.inference_manager.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.model.max_sequence_length,
                    temperature=0.6,
                    top_p=0.9,
                )
                all_outputs.append(generated)

        if self.debug:
            log.info("Inference completed")
        return all_outputs

    def handle_inference_request(self, conversation: List[Chat], max_length: int, temperature: float, top_p: float) -> \
    List[Chat]:
        tokenized = self.model.tokenize(conversation)
        input_ids = tokenized['input_ids'].to(self.accelerator.device)
        attention_mask = tokenized['attention_mask'].to(self.accelerator.device)

        generated = self.inference_manager.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )

        decoded_text = self.model.decode(generated)

        # Find the last assistant message and replace its content
        for chat in reversed(conversation):
            if chat.role == Role.assistant.value:
                chat.message = decoded_text.strip()
                break

        return conversation

    def decode_outputs(self, outputs):
        if self.debug:
            log.info("Decoding outputs")
        decoded_outputs = []
        for output in outputs:
            decoded = self.model.decode(output)
            decoded_outputs.append(decoded)
        return decoded_outputs