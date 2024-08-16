import fire
import os
import sys
import json
import torch
from config import get_config, update_config, set_model_preset
from src.coordination.model_node import ModelNode
from src.utils.logger import log

def start_server(config, is_reference):
    node = ModelNode(config, is_reference=is_reference)
    node.serve()

def main(method='dpo', model='pythia-160m', config_updates=None,
         nnodes=1, node_rank=0, nproc_per_node=None,
         master_addr='localhost', master_port='12355',
         use_fsdp=False, is_reference=False, **kwargs):
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

    start_server(config, is_reference)

    log.info(f"RPC Server started for {'reference' if is_reference else 'training'} model")

if __name__ == "__main__":
    fire.Fire(main)