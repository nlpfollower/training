from datetime import datetime
from typing import Dict, Optional, List

import torch
from transformers import GenerationConfig, StoppingCriteriaList, StoppingCriteria

class InferenceManager:
    def __init__(self, model, accelerator, debug=False):
        self.model = model
        self.accelerator = accelerator
        self.debug = debug

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model.forward(**batch)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_length: int, temperature: float = 0.6,
                 top_p: float = 0.9) -> List[int]:
        generation_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            use_cache=True,
            num_return_sequences=1,
        )

        stopping_criteria = StoppingCriteriaList([
            EOTEOMStoppingCriteria(self.model.stop_tokens())
        ])

        with torch.inference_mode(), self.accelerator.autocast():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
            )

        # Trim the input_ids length from the generated output
        generated_tokens = outputs[0][input_ids.shape[1]:].tolist()

        # Remove the EOT token if it's present at the end
        if generated_tokens and generated_tokens[-1] in self.model.stop_tokens():
            generated_tokens = generated_tokens[:-1]

        return generated_tokens

class EOTEOMStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens: List[int]):
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0, -1:] in self.stop_tokens:
            return True
        return False