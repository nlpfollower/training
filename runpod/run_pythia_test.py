#!/usr/bin/env python3
import argparse
from pod_manager import PodManager
from src.utils.logger import log as logger

def get_args():
    parser = argparse.ArgumentParser(description='Run Pythia training on RunPod')
    parser.add_argument('--model', type=str, default='pythia-160m', help='Model type to train')
    return parser.parse_args()


def main():
    args = get_args()
    pod_manager = PodManager()

    try:
        logger.info("Starting Pythia training on RunPod")
        logger.info("Model: {}", args.model)

        if not pod_manager.create_pod() or not pod_manager.wait_for_pod_ready() or not pod_manager.establish_ssh_connection():
            logger.error("Failed to set up the pod")
            return

        # List contents of /workspace
        ls_result = pod_manager.run_ssh_command("ls -la /workspace")
        if ls_result:
            logger.info("Contents of /workspace:\n{}", ls_result['output'])
            if ls_result['error']:
                logger.error("Error listing /workspace: {}", ls_result['error'])
        else:
            logger.error("Failed to list /workspace contents")

        # Run the Pythia training command using the pre-built environment
        train_command = (
            "bash -c '"
            "/workspace/llama-venv/bin/python "
            "/workspace/training/scripts/train.py "
            f"--model {args.model} --method dpo --nnodes 1 --nproc_per_node 1"
            "'"
        )
        train_result = pod_manager.run_ssh_command(train_command)

        if train_result:
            logger.info("Training command exit status: {}", train_result['exit_status'])
            logger.info("Training command output: {}", train_result['output'])
            if train_result['error']:
                logger.error("Training command error: {}", train_result['error'])
        else:
            logger.error("Failed to execute training command")

        logger.info("Training completed.")
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received. Terminating...")
    except Exception as e:
        logger.exception("An error occurred: {}", str(e))
    finally:
        pod_manager.cleanup()
        logger.info("Script execution completed")


if __name__ == '__main__':
    main()