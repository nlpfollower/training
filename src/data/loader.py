import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Iterator, Union
from config.config import Config
import os
from llama_models.llama3_1.api.tokenizer import Tokenizer as LlamaTokenizer
from src.model.llama_model import LlamaModel
from src.types.conversation import Chat
from src.types.datasets import DPOSample, DPOBatch
from src.types.conversation import Role

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)

class BaseLoader:
    def __init__(self, config: Config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def get_pad_token_id(self):
        if hasattr(self.tokenizer, 'pad_id'):
            return self.tokenizer.pad_id
        else:
            return self.tokenizer.pad_token_id

    def tokenize_conversation(self, chats: List[Chat]) -> Dict[str, torch.Tensor]:
        if isinstance(self.tokenizer, LlamaTokenizer):
            return LlamaModel.tokenize(self.tokenizer, chats)
        else:
            # Assume it's a standard Hugging Face tokenizer
            text = " ".join(chat.message for chat in chats)
            encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask']
            }

    def get_train_iterator(self) -> Iterator[Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def get_eval_iterator(self) -> Iterator[Dict[str, torch.Tensor]]:
        raise NotImplementedError

class DPODataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return DPOSample(
            generated_prompt=item['generated_prompt'],
            chat1=item['chat1'],
            chat2=item['chat2']
        )

class DPOLoader(BaseLoader):
    def __init__(self, config: Config, tokenizer: LlamaTokenizer):
        super().__init__(config, tokenizer)
        self.file_path = os.path.join(config.paths.dpo_output_dir, 'dpo_samples.json')
        self.data = self._load_data()
        self.train_data, self.eval_data = self._split_data()

    def _load_data(self):
        with open(self.file_path, 'r') as f:
            return json.load(f)

    def _split_data(self):
        eval_size = int(len(self.data) * self.config.training.eval_split)
        return self.data[:-eval_size], self.data[-eval_size:]

    def _collate_fn(self, batch: List[DPOSample]) -> DPOBatch:
        prompt_chats = [
            [Chat(role=Role.user, message=item.generated_prompt)]
            for item in batch
        ]
        chosen_chats = [
            [
                Chat(role=Role.user, message=item.generated_prompt),
                Chat(role=Role.assistant, message=item.chat2)
            ]
            for item in batch
        ]
        rejected_chats = [
            [
                Chat(role=Role.user, message=item.generated_prompt),
                Chat(role=Role.assistant, message=item.chat1)
            ]
            for item in batch
        ]

        tokenized_prompts = [self.tokenize_conversation(chats) for chats in prompt_chats]
        tokenized_chosen = [self.tokenize_conversation(chats) for chats in chosen_chats]
        tokenized_rejected = [self.tokenize_conversation(chats) for chats in rejected_chats]

        max_length = max(max(len(x['input_ids'][0]) for x in tokenized_chosen),
                         max(len(x['input_ids'][0]) for x in tokenized_rejected))

        def pad_and_mask(tokenized_list):
            padded_ids = torch.nn.utils.rnn.pad_sequence(
                [x['input_ids'][0] for x in tokenized_list],
                batch_first=True,
                padding_value=self.get_pad_token_id()
            )
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [x['attention_mask'][0] for x in tokenized_list],
                batch_first=True,
                padding_value=0
            )
            return padded_ids, attention_mask

        chosen_input_ids, chosen_attention_mask = pad_and_mask(tokenized_chosen)
        rejected_input_ids, rejected_attention_mask = pad_and_mask(tokenized_rejected)

        # Create labels (set to -100 for prompt tokens and padding)
        chosen_labels = chosen_input_ids.clone()
        rejected_labels = rejected_input_ids.clone()
        for i, prompt_len in enumerate(len(x['input_ids'][0]) for x in tokenized_prompts):
            chosen_labels[i, :prompt_len] = -100
            rejected_labels[i, :prompt_len] = -100
        chosen_labels[chosen_attention_mask == 0] = -100
        rejected_labels[rejected_attention_mask == 0] = -100

        # Concatenate chosen and rejected
        concatenated_input_ids = torch.cat([
            pad_to_length(chosen_input_ids, max_length, self.get_pad_token_id()),
            pad_to_length(rejected_input_ids, max_length, self.get_pad_token_id())
        ], dim=0)

        concatenated_attention_mask = torch.cat([
            pad_to_length(chosen_attention_mask, max_length, 0),
            pad_to_length(rejected_attention_mask, max_length, 0)
        ], dim=0)

        concatenated_labels = torch.cat([
            pad_to_length(chosen_labels, max_length, -100),
            pad_to_length(rejected_labels, max_length, -100)
        ], dim=0)

        return DPOBatch(
            input_ids=concatenated_input_ids,
            attention_mask=concatenated_attention_mask,
            labels=concatenated_labels,
            chosen_length=len(chosen_input_ids)
        )

    def get_train_iterator(self) -> Iterator[DPOBatch]:
        dataset = DPODataset(self.train_data)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=self.config.training.num_workers,
            pin_memory=True
        )
        return iter(dataloader)

    def get_eval_iterator(self) -> Iterator[DPOBatch]:
        dataset = DPODataset(self.eval_data)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.eval_batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=self.config.training.num_workers,
            pin_memory=True
        )
        return iter(dataloader)

class KTOLoader(BaseLoader):
    # Implement KTO-specific loading logic here
    pass

class SPFTLoader(BaseLoader):
    # Implement SPFT-specific loading logic here
    pass

def get_loader(config: Config, tokenizer) -> BaseLoader:
    method = config.training.method
    if method == "dpo":
        return DPOLoader(config, tokenizer)
    elif method == "kto":
        return KTOLoader(config, tokenizer)
    elif method == "spft":
        return SPFTLoader(config, tokenizer)
    else:
        raise ValueError(f"Unsupported training method: {method}")