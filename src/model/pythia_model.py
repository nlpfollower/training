import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer
from typing import Dict, Optional, List, Union
from src.types.conversation import Chat, Thread
from config import Config
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

class PythiaModel:
    def __init__(self, config: Config, device: torch.device):
        self.device = device
        self.config = config
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(config.model.model_path, quantization_config=quantization_config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)

        # Ensure the tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if config.fsdp.enabled:
            self.model = self._wrap_model_with_fsdp(self.model)

    def _wrap_model_with_fsdp(self, model):
        from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock  # Adjust this import based on the actual Pythia model architecture

        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={GPTNeoBlock},
        )

        fsdp_kwargs = {
            "auto_wrap_policy": wrap_policy,
            "sharding_strategy": getattr(FSDP.ShardingStrategy, self.config.fsdp.sharding_strategy),
            "device_id": torch.cuda.current_device(),
        }

        if self.config.fsdp.mixed_precision:
            from torch.distributed.fsdp import MixedPrecision
            fsdp_kwargs["mixed_precision"] = MixedPrecision(
                param_dtype=getattr(torch, self.config.fsdp.mixed_precision)
            )

        if self.config.fsdp.cpu_offload:
            from torch.distributed.fsdp import CPUOffload
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

        return FSDP(model, **fsdp_kwargs)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[
        str, torch.Tensor]:
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @staticmethod
    def format_conversation(chats: List[Chat]) -> str:
        formatted = ""
        for chat in chats:
            formatted += f"{chat.role}: {chat.message}\n"
        return formatted.strip()

    @staticmethod
    def tokenize(tokenizer: PreTrainedTokenizer, chats: List[Chat]) -> Dict[str, torch.Tensor]:
        formatted_text = PythiaModel.format_conversation(chats)
        encoded = tokenizer(formatted_text,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=tokenizer.model_max_length)
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

    def generate(self, input_ids: torch.Tensor, max_length: int, **kwargs) -> torch.Tensor:
        if self.config.fsdp.enabled:
            with FSDP.summon_full_params(self.model):
                return self.model.generate(input_ids, max_length=max_length, **kwargs)
        else:
            return self.model.generate(input_ids, max_length=max_length, **kwargs)

    def get_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.forward(input_ids, attention_mask)
        return outputs['logits']

    def get_log_probs(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.get_logits(input_ids, attention_mask)
        return torch.log_softmax(logits, dim=-1)

    def to(self, device: str):
        self.device = device
        if not self.config.fsdp.enabled:
            self.model.to(device)

    def get_tokenizer(self):
        return self.tokenizer

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)

    def save(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        if self.config.fsdp.enabled:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = self.state_dict()
            if torch.distributed.get_rank() == 0:
                torch.save(state_dict, os.path.join(save_directory, 'model.pt'))
        else:
            torch.save(self.state_dict(), os.path.join(save_directory, 'model.pt'))
