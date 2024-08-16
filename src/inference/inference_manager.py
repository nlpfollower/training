import torch
from accelerate import Accelerator
from src.utils.logger import log


class InferenceManager:
    def __init__(self, model, accelerator: Accelerator):
        self.model = model
        self.accelerator = accelerator

    def forward(self, batch):
        self.model.eval()
        with torch.no_grad():
            return self.model(**batch)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        log.info(f"Evaluation completed. Average Loss: {avg_loss:.4f}")
        return avg_loss