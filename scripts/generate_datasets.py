import fire
import json
from config import (get_config, update_config)
from src.user.user_manager import UserManager
from src.training.dpo import DPOGenerator
from src.training.kto import KTOGenerator
from src.utils.logger import log
from src.utils.json import dump_json
import pydevd_pycharm

def save_dpo_dataset(dpo_samples, dpo_sample_stats, output_dir):
    # Save DPOSamples
    dpo_samples_path = f"{output_dir}/dpo_samples.json"
    with open(dpo_samples_path, 'w', encoding='utf-8') as f:
        dump_json([sample.__dict__ for sample in dpo_samples], f, indent=2, ensure_ascii=False)
    log.info(f"DPO samples saved to {dpo_samples_path}")

    # Save DPOSampleStats
    dpo_sample_stats_path = f"{output_dir}/dpo_sample_stats.json"
    with open(dpo_sample_stats_path, 'w', encoding='utf-8') as f:
        dump_json([stats.__dict__ for stats in dpo_sample_stats], f, indent=2, ensure_ascii=False)
    log.info(f"DPO sample stats saved to {dpo_sample_stats_path}")

def save_kto_dataset(kto_samples, kto_sample_stats, output_dir):
    # Save KTOSamples
    kto_samples_path = f"{output_dir}/kto_samples.json"
    with open(kto_samples_path, 'w', encoding='utf-8') as f:
        dump_json([sample.__dict__ for sample in kto_samples], f, indent=2, ensure_ascii=False)
    log.info(f"KTO samples saved to {kto_samples_path}")

    # Save KTOSampleStats
    kto_sample_stats_path = f"{output_dir}/kto_sample_stats.json"
    with open(kto_sample_stats_path, 'w', encoding='utf-8') as f:
        dump_json([stats.__dict__ for stats in kto_sample_stats], f, indent=2, ensure_ascii=False)
    log.info(f"KTO sample stats saved to {kto_sample_stats_path}")

def generate_dpo_dataset(config, threads):
    log.info("Generating DPO dataset...")
    dpo_generator = DPOGenerator(threads)
    dpo_samples, dpo_sample_stats = dpo_generator.generate_dataset()

    log.info("Saving DPO dataset and stats...")
    save_dpo_dataset(dpo_samples, dpo_sample_stats, config.paths.dpo_output_dir)

    log.info("DPO dataset generation completed successfully!")

def generate_kto_dataset(config, threads):
    log.info("Generating KTO dataset...")
    kto_generator = KTOGenerator(threads)
    kto_samples, kto_sample_stats = kto_generator.generate_dataset()

    log.info("Saving KTO dataset and stats...")
    save_kto_dataset(kto_samples, kto_sample_stats, config.paths.kto_output_dir)

    log.info("KTO dataset generation completed successfully!")

def main(type='dpo', config_updates=None, **kwargs):
    # Load and update configuration
    config = get_config()
    if config_updates:
        update_config(config, **config_updates)

    # Step 1: Create UserManager instance and parse input data
    log.info("Initializing UserManager and parsing input data...")
    user_manager = UserManager(gptConversationsInputFile=config.paths.input_file)
    user_manager.load_chat_history()

    # Step 2: Get threads from UserManager
    threads = list(user_manager.get_all_threads()["threads"].values())

    # Step 3: Generate the specified dataset
    if type.lower() == 'dpo':
        generate_dpo_dataset(config, threads)
    elif type.lower() == 'kto':
        generate_kto_dataset(config, threads)
    else:
        log.error(f"Invalid dataset type: {type}. Please use 'dpo' or 'kto'.")

if __name__ == "__main__":
    pydevd_pycharm.settrace('localhost', port=6789, stdoutToServer=True, stderrToServer=True)
    fire.Fire(main)