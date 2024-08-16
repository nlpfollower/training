import os
import torch
from typing import Dict, Optional, List, Union

from accelerate import Accelerator

from config import Config
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

from src.types.conversation import Chat


class Model:
    def __init__(self, config: Config, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.model = None
        self.tokenizer = None

    def get_auto_wrap_policy(self):
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={self.get_transformer_layer_class()},
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def generate(self, input_ids: torch.Tensor, max_length: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def get_tokenizer(self):
        return self.tokenizer

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def save(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, 'model.pt')
        torch.save(self.model.state_dict(), model_path)

    def get_transformer_layer_class(self):
        raise NotImplementedError

    @staticmethod
    def format_conversation(chats: List[Chat]) -> str:
        raise NotImplementedError

    @staticmethod
    def tokenize(tokenizer, text, bos: bool = False, eos: bool = True) -> Dict[str, torch.Tensor]:
        raise NotImplementedError