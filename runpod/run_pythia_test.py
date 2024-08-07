#!/usr/bin/env python3
import argparse
import os
from pod_manager import PodManager
from pod_config import create_pod_config
from src.utils.logger import log as logger


def get_args():
    parser = argparse.ArgumentParser(description='Run Pythia training on RunPod')
    parser.add_argument('--model', type=str, default='pythia-160m', help='Model type to train')
    parser.add_argument('--network-volume-id', type=str, required=True, help='Network volume ID')
    return parser.parse_args()


def log_output(output):
    logger.info(output.strip())


def main():
    args = get_args()

    # Create pod configuration using the factory function
    pod_config = create_pod_config(
        network_volume_id=args.network_volume_id,
        model_name=args.model,
        preset='pythia'
    )

    pod_manager = PodManager(pod_config.to_dict())

    try:
        logger.info("Starting Pythia training on RunPod")
        logger.info("Model: {}", args.model)

        if not pod_manager.create_pod() or not pod_manager.wait_for_pod_ready() or not pod_manager.establish_ssh_connection():
            logger.error("Failed to set up the pod")
            return

        # List contents of /workspace
        ls_result = pod_manager.run_ssh_command_with_stream("ls -la /workspace", callback=log_output)
        if ls_result:
            if ls_result['error']:
                logger.error("Error listing /workspace: {}", ls_result['error'])
        else:
            logger.error("Failed to list /workspace contents")

        # Run the Pythia training command using the pre-built environment
        train_command = (
            "cd /workspace/training && bash -c '"
            "/workspace/llama-venv/bin/python -m scripts.train "
            f"--model {args.model} --method dpo --nnodes 1 --nproc_per_node 1"
            "'"
        )
        logger.info("Starting training command...")
        train_result = pod_manager.run_ssh_command_with_stream(train_command, callback=log_output)

        if train_result:
            logger.info("Training command exit status: {}", train_result['exit_status'])
            if train_result['error']:
                logger.error("Training command error: {}", train_result['error'])
        else:
            logger.error("Failed to execute training command")

        logger.info("Training completed.")

        # Transfer the output file
        remote_path = '/workspace/training/data/output/dpo/dpo_samples.json'
        local_path = os.path.join(os.getcwd(), 'data', 'output', 'dpo', 'dpo_samples.json')

        # Ensure the local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if pod_manager.transfer_file_to_local(remote_path, local_path):
            logger.info(f"Successfully transferred {remote_path} to {local_path}")

            # Delete the file from the pod
            if pod_manager.delete_file_on_pod(remote_path):
                logger.info(f"Successfully deleted {remote_path} from the pod")
            else:
                logger.error(f"Failed to delete {remote_path} from the pod")
        else:
            logger.error(f"Failed to transfer {remote_path} to {local_path}")

    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received. Terminating...")
    except Exception as e:
        logger.exception("An error occurred: {}", str(e))
    finally:
        pod_manager.cleanup()
        logger.info("Script execution completed")


if __name__ == '__main__':
    main()