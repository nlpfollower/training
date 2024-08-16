import threading
import time

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
from src.data.loader import get_loader
from src.utils.logger import log
from config import Config
import grpc
from concurrent import futures
import model_node_pb2
import model_node_pb2_grpc
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import DistributedDataParallelKwargs
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardedStateDictConfig
from collections import namedtuple

SampleMetadata = namedtuple('SampleMetadata', ['id', 'hash'])


class ModelNode(model_node_pb2_grpc.ModelNodeServicer):
    def __init__(self, config: Config, is_reference: bool):
        self.config = config
        self.is_reference = is_reference
        self.model = self.setup_model()
        self.server = None
        self.exit_event = threading.Event()

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
        self.device = self.accelerator.device

        self.data_loader = get_loader(self.config, self.model.get_tokenizer())
        self.train_dataloader, self.eval_dataloader = self.data_loader.split_data()
        self.train_dataloader_iter, self.eval_dataloader_iter = iter(self.train_dataloader), iter(self.eval_dataloader)
        self.train_metadata = enumerate_and_hash_batches(self.train_dataloader)
        self.eval_metadata = enumerate_and_hash_batches(self.eval_dataloader)

        if not is_reference:
            self.optimizer = AdamW(self.model.parameters(), lr=config.training.learning_rate)
            self.scheduler = LambdaLR(self.optimizer,
                                      lr_lambda=lambda step: min(1.0, (step + 1) / config.training.warmup_steps))

            # Prepare model, optimizer, scheduler, and dataloaders with Accelerator
            self.model, self.optimizer, self.scheduler, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler, self.train_dataloader, self.eval_dataloader
            )

            self.training_manager = TrainingManager(config, self.model, self.optimizer, self.scheduler,
                                                    self.accelerator)
        else:
            self.model = self.accelerator.prepare(self.model)
            self.inference_manager = InferenceManager(self.model, self.accelerator)

        self.last_forward = None
        self.action_queue = []
        self.action_lock = threading.Lock()

    def setup_model(self):
        if self.config.model.name == "llama3":
            self.model = LlamaModel(self.config, device=self.device)
        elif self.config.model.name == "pythia-160m":
            self.model = PythiaModel(self.config, device=self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model.name}")
        self.model.get_tokenizer().model_max_length = self.config.model.max_sequence_length
        return self.model

    def get_auto_wrap_policy(self):
        def auto_wrap_policy(module):
            return isinstance(module, (self.model.get_transformer_layer_class(),))

        return auto_wrap_policy


    def Forward(self, request, context):
        batch_id = request.batch_id
        batch_hash = request.batch_hash
        is_train = request.is_train

        metadata = self.train_metadata if is_train else self.eval_metadata
        if not verify_batch_metadata(batch_id, batch_hash, metadata):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid batch ID or hash")

        dataloader_iter = self.train_dataloader_iter if is_train else self.eval_dataloader_iter
        batch = next(dataloader_iter)
        batch = self.trainer.move_batch_to_device(batch, self.device)

        with self.accelerator.autocast():
            if self.is_reference:
                output = self.inference_manager.forward(batch)
            else:
                output = self.training_manager.forward(batch)
                self.last_forward = {
                    'batch_id': batch_id,
                    'batch_hash': batch_hash,
                    'output': output
                }

        response = model_node_pb2.ForwardResponse(batch_id=batch_id, batch_hash=batch_hash)
        if self.is_reference:
            response.output = output.cpu().numpy().astype(np.float32).tobytes()

        return response

    def Backward(self, request, context):
        if self.is_reference:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Backward not supported for reference model")

        if (self.last_forward is None or
            self.last_forward['batch_id'] != request.batch_id or
            self.last_forward['batch_hash'] != request.batch_hash):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "No matching forward pass found for this backward request")

        policy_logits = self.last_forward['output']
        reference_logits = torch.from_numpy(
            np.frombuffer(request.reference_output, dtype=np.float32)
        ).to(self.device).reshape_as(policy_logits)

        loss = self.training_manager.backward(policy_logits, reference_logits, self.last_forward['batch'])

        return model_node_pb2.BackwardResponse(
            success=True,
            batch_id=request.batch_id,
            batch_hash=request.batch_hash,
            loss=loss
        )

    def Exit(self, request, context):
        log.info("Received exit request. Initiating shutdown.")
        self.is_shutting_down = True
        if self.server:
            self.server.stop(0)
        return model_node_pb2.ExitResponse(success=True)

    def serve(self):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        model_node_pb2_grpc.add_ModelNodeServicer_to_server(self, self.server)
        self.server.add_insecure_port(f'[::]:{self.config.training.rpc_server_port}')
        self.server.start()
        log.info(f"RPC Server started, listening on port {self.config.training.rpc_server_port}")

        try:
            while not self.is_shutting_down:
                self.server.wait_for_termination(timeout=1)
        except KeyboardInterrupt:
            log.info("Received keyboard interrupt. Initiating shutdown.")
        finally:
            if self.server:
                self.server.stop(0)
            log.info("Server shut down successfully.")