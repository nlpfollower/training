from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from src.model.llama_model import LlamaModel
from src.model.pythia_model import PythiaModel
from src.data.loader import get_loader
from src.utils.logger import log, log_training_progress
from src.types.datasets import DPOBatch
from src.training.mock import Trainer
from config import Config
from typing import Dict, Any
import os
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardedStateDictConfig,
    FullOptimStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools
from accelerate import Accelerator, FullyShardedDataParallelPlugin

class TrainingManager:
    def __init__(self, config: Config, trainer: Trainer, model, optimizer, scheduler, accelerator: Accelerator):
        self.config = config
        self.trainer = trainer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.current_epoch = 0
        self.global_step = 0

    def forward(self, batch):
        self.model.train()
        with self.accelerator.autocast():
            outputs = self.trainer.forward(self.model, batch)
        # TODO: Do we need to gather?
        return outputs.logits

    def backward(self, policy_logits, reference_logits, batch):
        loss = self.trainer.compute_loss(policy_logits, reference_logits, batch)

        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        self.global_step += 1
        if self.global_step % self.config.training.log_interval == 0:
            log_training_progress(self.current_epoch, self.global_step, loss.item(), self.scheduler.get_last_lr()[0])

        return loss.item()

    def save_checkpoint(self, output_dir, epoch):
        self.accelerator.save_state(output_dir=output_dir)
        log.info(f"Saved checkpoint for epoch {epoch} to {output_dir}")