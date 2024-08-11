#!/usr/bin/env python3
import argparse
import os
from pod_manager import PodManager
from src.utils.logger import log
from src.utils.training_progress import TrainingProgress


def get_args():
    parser = argparse.ArgumentParser(description='Run Pythia training on RunPod')
    parser.add_argument('--model', type=str, default='pythia-160m', help='Model type to train')
    parser.add_argument('--network-volume-id', type=str, required=True, help='Network volume ID')
    return parser.parse_args()


def main():
    args = get_args()

    pod_manager = PodManager()
    internal_pod_id = "pythia-single-node"

    # Transfer input file from local to remote
    local_path = os.path.join(os.getcwd(), 'data', 'output', 'dpo', 'dpo_samples.json')
    remote_path = '/workspace/training/data/output/dpo/dpo_samples.json'

    try:
        log.info("Starting Pythia training on RunPod")
        log.info(f"Model: {args.model}")

        if not pod_manager.create_pod(internal_pod_id, args.network_volume_id, args.model, preset='pythia'):
            log.error("Failed to create pod")
            return

        pod = pod_manager.get_pod(internal_pod_id)

        # Ensure the local file exists
        if not os.path.exists(local_path):
            log.error(f"Local file {local_path} does not exist")
            return

        if pod_manager.transfer_file_to_pod(internal_pod_id, local_path, remote_path):
            log.info(f"Successfully transferred {local_path} to {remote_path}")
        else:
            log.error(f"Failed to transfer {local_path} to {remote_path}")
            return

        # Run the Pythia training command using the pre-built environment
        train_command = (
            "cd /workspace/training && bash -c '"
            "/workspace/llama-venv/bin/python -m scripts.train "
            f"--model {args.model} --method dpo --nnodes 1 --nproc_per_node 1"
            "'"
        )
        log.info("Starting training command...")

        training_progress = TrainingProgress()
        exit_status = pod_manager.run_command_with_stream_on_pod(internal_pod_id, train_command, training_progress.update)
        training_progress.close()

        if exit_status is not None:
            log.info(f"Training command completed with exit status: {exit_status}")
        else:
            log.error("Failed to execute training command")

        log.info("Training completed.")

    except KeyboardInterrupt:
        log.warning("Keyboard interrupt received. Terminating...")
    except Exception as e:
        log.exception(f"An error occurred: {str(e)}")
    finally:
        # Only try to delete the file if the pod still exists
        if pod_manager.get_pod(internal_pod_id):
            if pod_manager.delete_file_on_pod(internal_pod_id, remote_path):
                log.info(f"Successfully deleted {remote_path} from the pod")
            else:
                log.warning(f"Failed to delete {remote_path} from the pod")

        # Cleanup the pod
        pod_manager.cleanup_pod(internal_pod_id)
        log.info("Script execution completed")


if __name__ == '__main__':
    main()