# launch_multi_node.py
import os
import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.multiprocessing as mp
from config import get_config
from training_manager import TrainingManager


def setup(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run(rank, world_size, config, node_rank, nodes):
    setup(rank, world_size, config.master_addr, config.master_port)

    trainer = get_trainer(config)  # Implement this function to get the appropriate trainer
    manager = TrainingManager(config, trainer, rank, world_size, node_rank, nodes)
    manager.train()

    cleanup()


def main():
    config = get_config()
    world_size = torch.cuda.device_count()
    node_rank = int(os.environ.get("NODE_RANK", 0))
    nodes = int(os.environ.get("WORLD_SIZE", 1))

    mp.spawn(run, args=(world_size, config, node_rank, nodes), nprocs=world_size, join=True)

    # If this is the master node, run the shard merging script
    if node_rank == 0:
        os.system(f"{sys.executable} merge_shards.py")


if __name__ == "__main__":
    main()