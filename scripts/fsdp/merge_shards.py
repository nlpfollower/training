# merge_shards.py
import os
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from config import get_config
from src.model.llama_model import LlamaModel  # or whichever model you're using

def merge_shards():
    config = get_config()
    model = LlamaModel(config)  # Initialize your model
    model = FSDP(model)  # Wrap with FSDP

    # Load all shards
    shards = []
    for filename in os.listdir(config.paths.node_specific_dir):
        if filename.startswith("model_shard_"):
            shard_path = os.path.join(config.paths.node_specific_dir, filename)
            shard = torch.load(shard_path)
            shards.append(shard)

    # Merge shards
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model.load_state_dict(shards)

    # Save full model
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()
        torch.save(state_dict, f"{config.paths.shared_model_dir}/merged_model.pt")

if __name__ == "__main__":
    merge_shards()