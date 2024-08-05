import os
import torch
from transformers import LlamaForCausalLM, BitsAndBytesConfig
from llama_models.llama3_1.api.tokenizer import Tokenizer as LlamaTokenizer
from typing import Dict, Optional, List, Union
from src.types.conversation import Chat, Thread
from config import Config
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

class LlamaModel:
    def __init__(self, config: Config, device: torch.device):
        self.device = device
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        self.model = LlamaForCausalLM.from_pretrained(config.model.model_path, quantization_config=quantization_config)
        self.tokenizer = LlamaTokenizer(config.model.tokenizer_path)
        if config.fsdp.enabled:
            self.model = self._wrap_model_with_fsdp(self.model, config)

    def _wrap_model_with_fsdp(self, model, config):
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )

        fsdp_kwargs = {
            "auto_wrap_policy": wrap_policy,
            "sharding_strategy": getattr(ShardingStrategy, config.fsdp.sharding_strategy),
            "device_id": torch.cuda.current_device(),
        }

        if config.fsdp.mixed_precision:
            fsdp_kwargs["mixed_precision"] = MixedPrecision(
                param_dtype=getattr(torch, config.fsdp.mixed_precision)
            )

        if config.fsdp.cpu_offload:
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

        return FSDP(model, **fsdp_kwargs)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[
        str, torch.Tensor]:
        """
        Perform a forward pass through the model.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor, optional): The attention mask.

        Returns:
            Dict[str, torch.Tensor]: A d
            ictionary containing the model outputs.
        """
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
        """
        Format a conversation for Llama 3.1 model input.

        Args:
            chats (List[Chat]): A list of Chat objects representing the conversation.

        Returns:
            str: Formatted conversation string.
        """
        formatted = "<|begin_of_text|>"
        for chat in chats:
            formatted += f"<|start_header_id|>{chat.role}<|end_header_id|>{chat.message}<|eot_id|>"
        return formatted

    @staticmethod
    def tokenize(tokenizer: LlamaTokenizer, text: Union[str, List[Chat]], bos: bool = False, eos: bool = True) -> Dict[
        str, torch.Tensor]:
        """
        Tokenize text or a conversation using the Llama tokenizer.

        Args:
            tokenizer (LlamaTokenizer): The Llama tokenizer.
            text (Union[str, List[Chat]]): Either a string or a list of Chat objects representing the conversation.
            bos (bool): Whether to add the beginning-of-sequence token.
            eos (bool): Whether to add the end-of-sequence token.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the tokenized inputs.
        """
        if isinstance(text, list):
            text = LlamaModel.format_conversation(text)

        encoded = tokenizer.encode(text, bos=bos, eos=eos)
        input_ids = torch.tensor([encoded], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


    def generate(self, input_ids: torch.Tensor, max_length: int, **kwargs) -> torch.Tensor:
        """
        Generate text based on input.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            max_length (int): The maximum length of the generated sequence.
            **kwargs: Additional arguments for generation.

        Returns:
            torch.Tensor: The generated token IDs.
        """
        return self.model.generate(input_ids, max_length=max_length, **kwargs)

    def get_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the logits for the input sequence.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor, optional): The attention mask.

        Returns:
            torch.Tensor: The logits for each token in the sequence.
        """
        outputs = self.forward(input_ids, attention_mask)
        return outputs['logits']

    def get_log_probs(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the log probabilities for the input sequence.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor, optional): The attention mask.

        Returns:
            torch.Tensor: The log probabilities for each token in the sequence.
        """
        logits = self.get_logits(input_ids, attention_mask)
        return torch.log_softmax(logits, dim=-1)

    def to(self, device: str):
        """
        Move the model to the specified device.

        Args:
            device (str): The device to move the model to (e.g., 'cuda', 'cpu').
        """
        self.device = device
        self.model.to(device)

    def get_tokenizer(self):
        """ Get the tokenizer """
        return self.tokenizer

    def train(self):
        """Set the model to training mode."""
        self.model.train()

    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()

    def parameters(self):
        """Get the model parameters."""
        return self.model.parameters()

    def save(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, 'model.pt')
        torch.save(self.model.state_dict(), model_path)