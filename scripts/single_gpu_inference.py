# scripts/single_gpu_inference.py

import fire
import pydevd_pycharm
import torch
from config import get_config, update_config, set_model_preset
from src.coordination.model_node import ModelNode
from src.types.conversation import Chat, Role
from src.utils.logger import log
from src.data.loader import RawDataset
from src.utils.profiler import Profiler, add_profiler_args
import argparse


def main(model='llama3',
         input_data="Are you working?",
         method="inference",
         model_path=None,
         config_updates=None,
         max_sequence_length=2048,
         system_prompt="You are a helpful AI assistant",
         debug=False,
         **kwargs):
    parser = argparse.ArgumentParser()
    add_profiler_args(parser)
    args, unknown = parser.parse_known_args()

    if args.profile:
        Profiler.initialize(snapshot_dir=args.profile_dir)

    # Setup configuration
    config = get_config()
    set_model_preset(config, model)
    if config_updates is None:
        config_updates = {}
    config_updates['model.max_sequence_length'] = max_sequence_length
    update_config(config, **config_updates)

    # Update model path if provided
    if model_path:
        config.model.model_path = model_path

    # Create RawDataset
    raw_dataset = RawDataset([input_data], method)

    # Initialize ModelNode
    model_node = ModelNode(config, raw_dataset, is_reference=True, system_prompt=system_prompt, debug=debug)

    log.info(f"Input: {input_data}")

    # Run inference
    outputs = model_node.run_inference()

    # Decode outputs
    decoded_outputs = model_node.decode_outputs(outputs)

    # Print results
    for i, decoded_output in enumerate(decoded_outputs):
        log.info(f"Output: {decoded_output}")

    log.info("Inference completed.")

    if args.profile:
        Profiler.take_snapshot('end_of_inference')


if __name__ == "__main__":
    # pydevd_pycharm.settrace('localhost', port=6789, stdoutToServer=True, stderrToServer=True)
    fire.Fire(main)