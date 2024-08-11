import re
from tqdm import tqdm
from src.utils.logger import logger


class TrainingProgress:
    def __init__(self):
        self.epoch = 0
        self.batch = 0
        self.total_batches = 0
        self.loss = 0.0
        self.lr = 0.0
        self.progress_bar = None

    def update(self, output):
        match = re.search(
            r'Epoch:\s+(\d+)\s+\|\s+Batch:\s+(\d+)/\s*(\d+)\s+\|\s+Loss:\s+([\d.]+|nan)\s+\|\s+LR:\s+([\d.]+)', output)
        if match:
            self.epoch = int(match.group(1))
            self.batch = int(match.group(2))
            self.total_batches = int(match.group(3))
            self.loss = float(match.group(4)) if match.group(4) != 'nan' else float('nan')
            self.lr = float(match.group(5))

            if not self.progress_bar:
                self.progress_bar = tqdm(total=self.total_batches, desc=f"Epoch {self.epoch}", ncols=100)

            self.progress_bar.n = self.batch
            self.progress_bar.set_postfix({"Loss": f"{self.loss:.4f}", "LR": f"{self.lr:.6f}"})
            self.progress_bar.update(0)  # Force refresh
        else:
            logger.info(output.strip())

    def close(self):
        if self.progress_bar:
            self.progress_bar.close()