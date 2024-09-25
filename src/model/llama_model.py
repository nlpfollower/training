import os
import torch
from transformers import LlamaForCausalLM, BitsAndBytesConfig, LlamaConfig
from llama_models.llama3_1.api.tokenizer import Tokenizer as LlamaTokenizer
from typing import Dict, Optional, List, Union, Tuple

from src.model.model import Model
from src.types.conversation import Chat, Thread, Role
from config import Config
from src.utils.logger import log


class LlamaModel(Model):
    def __init__(self, config: Config):
        super().__init__(config)
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        llama_config = LlamaConfig.from_pretrained(config.model.model_path)
        llama_config._attn_implementation = "flash_attention_2"
        self.model = LlamaForCausalLM.from_pretrained(
            config.model.model_path,
            config=llama_config,
            device_map='auto',
            torch_dtype=torch.float16,
        )
        self.tokenizer = LlamaTokenizer(config.model.tokenizer_path)

        # Store token IDs for special tokens
        self.begin_of_text_id = self.tokenizer.bos_id
        self.end_of_text_id = self.tokenizer.eos_id
        self.eot_id = self.tokenizer.eot_id
        self.start_header_id = self.tokenizer.special_tokens["<|start_header_id|>"]
        self.end_header_id = self.tokenizer.special_tokens["<|end_header_id|>"]

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple] = None) -> Dict[str, torch.Tensor]:
        outputs = self.model(input_ids, attention_mask=attention_mask, past_key_values=past_key_values)
        return {
            'logits': outputs.logits,
            'past_key_values': outputs.past_key_values
        }

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.generate(input_ids, **kwargs)

    def format_conversation(self, chats: List[Chat], system_prompt: str = "", include_eot: bool = False) -> List[int]:
        formatted = [self.begin_of_text_id]

        if system_prompt:
            formatted.extend([
                self.start_header_id,
                *self.tokenizer.encode(Role.system.value, bos=False, eos=False),
                self.end_header_id,
                *self.tokenizer.encode(system_prompt, bos=False, eos=False),
                self.eot_id
            ])

        for i, chat in enumerate(chats):
            formatted.extend([
                self.start_header_id,
                *self.tokenizer.encode(chat.role, bos=False, eos=False),
                self.end_header_id,
                *self.tokenizer.encode(chat.message, bos=False, eos=False)
            ])
            if i < len(chats) - 1 or include_eot:
                formatted.append(self.eot_id)

        return formatted

    def tokenize(self, chats: List[Chat], system_prompt: str = "") -> Dict[str, torch.Tensor]:
        formatted_ids = self.format_conversation(chats, system_prompt)
        input_ids = torch.tensor([formatted_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def decode(self, token_ids: List[int]) -> str:
        decoded = self.tokenizer.decode(token_ids)
        return decoded

    def get_transformer_layer_class(self):
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        return LlamaDecoderLayer

    def stop_tokens(self) -> List[int]:
        return self.tokenizer.stop_tokens