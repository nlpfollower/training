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
import json
import subprocess

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

def train(rank, config):
    setup(rank, config.training.world_size, config.training.master_addr, config.training.master_port)
    trainer = get_trainer(config)
    manager = TrainingManager(config, trainer, rank)
    manager.train()
    cleanup()

def main(method='dpo', model='pythia-160m', config_updates=None,
         nnodes=1, node_rank=0, nproc_per_node=None,
         master_addr='localhost', master_port='12355',
         use_fsdp=False, **kwargs):
    config = get_config()
    set_model_preset(config, model)
    if config_updates:
        update_config(config, **config_updates)
    config.training.method = method
    config.fsdp.enabled = use_fsdp
    config.training.nodes = nnodes
    config.training.node_rank = node_rank
    config.training.master_addr = master_addr
    config.training.master_port = master_port

    if nproc_per_node is None:
        nproc_per_node = torch.cuda.device_count()

    config.training.world_size = nnodes * nproc_per_node
    config.training.distributed = (config.training.world_size > 1)

    if use_fsdp and nnodes > 1:
        # Use the FSDP multi-node launch script
        script_path = os.path.join(os.path.dirname(__file__), 'fsdp', 'launch_multi_node.py')
        cmd = [
            sys.executable, script_path,
            '--config', json.dumps(config.__dict__)
        ]
        subprocess.run(cmd, check=True)
    elif config.training.world_size > 1:
        mp.spawn(train, args=(config,), nprocs=nproc_per_node, join=True)
    else:
        train(0, config)

    # Run merge_shards if this is the master node in FSDP mode
    if use_fsdp and node_rank == 0:
        merge_script_path = os.path.join(os.path.dirname(__file__), 'fsdp', 'merge_shards.py')
        subprocess.run([sys.executable, merge_script_path, '--config', json.dumps(config.__dict__)], check=True)

if __name__ == "__main__":
    fire.Fire(main)