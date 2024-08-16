import os
import torch
from transformers import LlamaForCausalLM
from llama_models.llama3_1.api.tokenizer import Tokenizer as LlamaTokenizer
from typing import Dict, Optional, List, Union

from src.model.model import Model
from src.types.conversation import Chat, Thread
from config import Config

class LlamaModel(Model):
    def __init__(self, config: Config, device: torch.device):
        super().__init__(config, device)
        self.model = LlamaForCausalLM.from_pretrained(config.model.model_path)
        self.tokenizer = LlamaTokenizer(config.model.tokenizer_path)

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
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        return LlamaDecoderLayer

    @staticmethod
    def format_conversation(chats: List[Chat]) -> str:
        formatted = "<|begin_of_text|>"
        for chat in chats:
            formatted += f"<|start_header_id|>{chat.role}<|end_header_id|>{chat.message}<|eot_id|>"
        return formatted

    @staticmethod
    def tokenize(tokenizer: LlamaTokenizer, text: Union[str, List[Chat]], bos: bool = False, eos: bool = True) -> Dict[str, torch.Tensor]:
        if isinstance(text, list):
            text = LlamaModel.format_conversation(text)
        encoded = tokenizer.encode(text, bos=bos, eos=eos)
        input_ids = torch.tensor([encoded], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }