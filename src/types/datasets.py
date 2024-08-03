from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
import torch

@dataclass
class DPOBatch:
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor
    chosen_length: int

@dataclass
class DPOSample:
    generated_prompt: str
    chat1: str
    chat2: str

@dataclass
class DPOSampleStats:
    partial_context: List[str]
    extra_context: List[str]
    final_context: Optional[List[str]]
    chat1: str
    chat2: str
    similarities: Dict[str, float]
    sentiment: Dict[str, Union[float, str]]
    llm_judgment: Dict[str, Optional[str]]

@dataclass
class KTOSample:
    prompt: str
    chat: str
    sentiment: Dict[str, Union[float, str]]

@dataclass
class KTOSampleStats:
    partial_context: List[str]
    chat: str
    final_context: str
    sentiment: Dict[str, Union[float, str]]

@dataclass
class SPFTSample:
    # Define SPFT sample structure here
    pass