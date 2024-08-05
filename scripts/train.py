import fire
import os
import sys
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from config import get_config, update_config, set_model_preset
from src.training.training_manager import TrainingManager
from src.utils.logger import log
import pydevd_pycharm

def setup(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    torch.distributed.destroy_process_group()

def get_trainer(config):
    method = config.training.method
    if method == "dpo":
        from src.training.dpo import DPOTrainer
        return DPOTrainer(config)
    elif method == "kto":
        from src.training.kto import KTOTrainer
        return KTOTrainer(config)
    elif method == "spft":
        from src.training.spft import SPFTTrainer
        return SPFTTrainer(config)
    elif method == "multi":
        from src.training.multi_method import MultiMethodTrainer
        return MultiMethodTrainer(config)
    else:
        raise ValueError(f"Unsupported training method: {method}")

def train(rank, world_size, config, master_addr, master_port):
    setup(rank, world_size, master_addr, master_port)
    trainer = get_trainer(config)
    manager = TrainingManager(config, trainer, rank, world_size)
    manager.train()

def main(method='dpo', model='llama3', config_updates=None,
         nnodes=1, node_rank=0, nproc_per_node=None,
         master_addr='localhost', master_port='12355', **kwargs):
    config = get_config()
    set_model_preset(config, model)
    if config_updates:
        update_config(config, **config_updates)
    config.training.method = method

    if nproc_per_node is None:
        nproc_per_node = torch.cuda.device_count()

    world_size = nnodes * nproc_per_node
    if world_size > 1:
        mp.spawn(train, args=(world_size, config, master_addr, master_port),
                 nprocs=nproc_per_node, join=True)
    else:
        train(0, 1, config, master_addr, master_port)

if __name__ == "__main__":
    #pydevd_pycharm.settrace('localhost', port=6789, stdoutToServer=True, stderrToServer=True)
    fire.Fire(main)