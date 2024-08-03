from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from src.model.llama_model import LlamaModel
from src.model.pythia_model import PythiaModel
from src.data.loader import get_loader
from src.utils.logger import log
from src.types.datasets import DPOBatch
from src.training.mock import Trainer
from config.config import Config
from typing import Dict, Any
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    FullOptimStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

class TrainingManager:
    def __init__(self, config: Config, trainer, rank: int, world_size: int):
        self.config = config
        self.trainer = trainer
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')

        # Initialize models
        self.model = self._init_model(config)
        self.reference_model = self._init_model(config)
        self.reference_model.eval()

        # Wrap model with DDP if using multiple GPUs without FSDP
        if self.world_size > 1 and not config.fsdp.enabled:
            self.model = DDP(self.model, device_ids=[rank])
            self.reference_model = DDP(self.reference_model, device_ids=[rank])

        # Initialize optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.training.learning_rate)
        self.scheduler = LambdaLR(self.optimizer,
                                  lr_lambda=lambda step: min(1.0, (step + 1) / self.config.training.warmup_steps))

        # Initialize data loader
        self.data_loader = get_loader(config, self.model.get_tokenizer())

    def _init_model(self, config: Config):
        if config.model.name == "llama3":
            model = LlamaModel(config, device=self.device)
        elif config.model.name == "pythia-160m":
            model = PythiaModel(config, device=self.device)
        else:
            raise ValueError(f"Unsupported model type: {config.model.name}")

        if config.fsdp.enabled:
            model = self._wrap_model_with_fsdp(model)

        return model

    def _wrap_model_with_fsdp(self, model):
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        wrap_policy = transformer_auto_wrap_policy(transformer_layer_cls={LlamaDecoderLayer})

        fsdp_kwargs = {
            "auto_wrap_policy": wrap_policy,
            "sharding_strategy": getattr(FSDP.ShardingStrategy, self.config.fsdp.sharding_strategy),
            "device_id": self.rank,
        }

        if self.config.fsdp.mixed_precision:
            from torch.distributed.fsdp import MixedPrecision
            fsdp_kwargs["mixed_precision"] = MixedPrecision(
                param_dtype=getattr(torch, self.config.fsdp.mixed_precision)
            )

        if self.config.fsdp.cpu_offload:
            from torch.distributed.fsdp import CPUOffload
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

        wrapped_model = FSDP(model, **fsdp_kwargs)

        if self.config.fsdp.activation_checkpointing:
            self._enable_activation_checkpointing(wrapped_model)

        return wrapped_model

    def _enable_activation_checkpointing(self, model):
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            apply_activation_checkpointing,
        )

        check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn
        )

    def train(self):
        for epoch in range(self.config.training.num_epochs):
            self._log_rank0(f"Starting epoch {epoch + 1}/{self.config.training.num_epochs}")
            epoch_loss = self._train_epoch()
            self._log_rank0(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss:.4f}")

            if (epoch + 1) % self.config.training.save_interval == 0:
                self._save_checkpoint(epoch + 1)

            if (epoch + 1) % self.config.training.eval_interval == 0:
                eval_loss = self.evaluate()
                self._log_rank0(f"Evaluation after epoch {epoch + 1}. Eval Loss: {eval_loss:.4f}")

        self._save_checkpoint(self.config.training.num_epochs, is_final=True)

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        train_iter = self.data_loader.get_train_iterator()

        if self.rank == 0:
            progress_bar = tqdm(total=len(train_iter), desc="Training", ncols=100)

        for batch in train_iter:
            batch = self.trainer.move_batch_to_device(batch, self.device)
            loss = self._process_batch(batch)

            total_loss += loss.item()
            num_batches += 1

            if self.rank == 0:
                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        if self.rank == 0:
            progress_bar.close()

        # Aggregate loss across all processes
        if self.world_size > 1:
            dist.all_reduce(torch.tensor(total_loss).to(self.device))
            total_loss /= self.world_size

        return total_loss / num_batches

    def _process_batch(self, batch) -> torch.Tensor:
        self.optimizer.zero_grad()
        loss = self.trainer.compute_loss(self.model, self.reference_model, batch)
        loss.backward()

        if self.config.training.max_grad_norm > 0:
            self._clip_gradients()

        self.optimizer.step()
        self.scheduler.step()

        return loss

    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.data_loader.get_eval_iterator():
                batch = self.trainer.move_batch_to_device(batch, self.device)
                loss = self.trainer.compute_loss(self.model, self.reference_model, batch)
                total_loss += loss.item()
                num_batches += 1

        # Aggregate loss across all processes
        if self.world_size > 1:
            dist.all_reduce(torch.tensor(total_loss).to(self.device))
            total_loss /= self.world_size

        self.model.train()
        return total_loss / num_batches

    def _clip_gradients(self):
        if self.config.fsdp.enabled:
            self.model.clip_grad_norm_(self.config.training.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)

    def _save_checkpoint(self, epoch: int, is_final: bool = False):
        if self.rank != 0:
            return

        model_name = self.config.model.name
        output_dir = os.path.join(self.config.paths.local_weights_dir, model_name,
                                  f'final_model' if is_final else f'checkpoint-{epoch}')
        os.makedirs(output_dir, exist_ok=True)

        if self.config.fsdp.enabled:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, 'model.pt'))

            optim_state = FSDP.optim_state_dict(self.model, self.optimizer)
            torch.save(optim_state, os.path.join(output_dir, 'optimizer.pt'))
        else:
            torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.pt'))
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))

        torch.save(self.scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
        self._log_rank0(f"Model checkpoint saved to {output_dir}")

    def _log_rank0(self, message: str):
        if self.rank == 0:
            log.info(message)