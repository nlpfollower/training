import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from typing import Dict, Optional, List
from src.model.model import Model
from src.types.conversation import Chat
from config import Config
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP
)

class PythiaModel(Model):
    def __init__(self, config: Config, device: torch.device):
        super().__init__(config, device)
        self.model = AutoModelForCausalLM.from_pretrained(config.model.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }

    def generate(self, input_ids: torch.Tensor, max_length: int, **kwargs) -> torch.Tensor:
        return self.model.generate(input_ids, max_length=max_length, **kwargs)

    def get_transformer_layer_class(self):
        from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
        return GPTNeoBlock

    @staticmethod
    def format_conversation(chats: List[Chat]) -> str:
        formatted = ""
        for chat in chats:
            formatted += f"{chat.role}: {chat.message}\n"
        return formatted.strip()

    @staticmethod
    def tokenize(tokenizer: PreTrainedTokenizer, text: List[Chat], bos: bool = False, eos: bool = True) -> Dict[str, torch.Tensor]:
        formatted_text = PythiaModel.format_conversation(text)
        encoded = tokenizer(formatted_text,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=tokenizer.model_max_length)
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
