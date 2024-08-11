import time
import os
import argparse
from pod_manager import PodManager
from runpod_launcher.pod_config import create_pod_config
from src.utils.logger import log

def get_args():
    parser = argparse.ArgumentParser(description='Setup distributed training nodes')
    parser.add_argument('--model', type=str, default='pythia-160m', help='Model type to train')
    parser.add_argument('--network-volume-id', type=str, required=True, help='Network volume ID')
    return parser.parse_args()

def main():
    args = get_args()
    manager = PodManager()

    # Add cleanup function to delete dpo_samples.json
    manager.add_cleanup_function(
        lambda: manager.delete_file_on_pod("node2-a40", "/root/training/data/output/dpo/dpo_samples.json"))

    try:
        # Create node 1 (A40) with existing network volume
        log.info(f"Creating node 1 (A40) with network volume {args.network_volume_id}")
        node1_config = create_pod_config(
            network_volume_id=args.network_volume_id,
            name="node1-a40",
            gpuTypeId="NVIDIA A40",
            ports="22/tcp,8080/http"
        )
        if not manager.create_pod("node1-a40", **node1_config.to_dict()):
            log.error("Failed to create node 1 (A40)")
            return

        # Move boot.tar on node1
        log.info("Copying boot.tar to /root on node1")
        move_command = "cp /workspace/boot.tar /root && cd /root"
        result = manager.run_command_on_pod("node1-a40", move_command)
        if not result or result['exit_status'] != 0:
            log.error(f"Failed to move boot.tar on node1")
            return

        # Start HTTP server on node1
        log.info("Starting HTTP server on node1")
        serve_command = "nohup python -m http.server 8080 > /dev/null 2>&1 &"
        result = manager.run_command_on_pod("node1-a40", serve_command)
        if not result or result['exit_status'] != 0:
            log.error(f"Failed to start HTTP server on node1")
            return

        # Create node 2 (A40)
        log.info("Creating node 2 (A40)")
        node2_config = create_pod_config(
            network_volume_id="",  # No network volume
            name="node2-a40",
            gpuTypeId="NVIDIA A40"
        )
        if not manager.create_pod("node2-a40", **node2_config.to_dict()):
            log.error("Failed to create node 2 (A40)")
            return

        # Download and extract boot.tar on node2
        node1_id = manager.get_pod("node1-a40").runpod_id
        # Download boot.tar on node2 with progress information
        log.info("Downloading boot.tar on node2")
        download_command = f"wget --progress=bar:force:noscroll https://{node1_id}-8080.proxy.runpod.net/boot.tar -O /root/boot.tar"
        exit_status = manager.run_command_with_stream_on_pod("node2-a40", download_command, log.info)
        if exit_status is None or exit_status != 0:
            log.error("Failed to download boot.tar on node2")
            return

        # Extract boot.tar on node2
        log.info("Extracting boot.tar on node2")
        extract_command = (
            "cd /root && "
            "tar -xvf boot.tar && "
            "tar -xvf boot-env.tar.zst && "
            "tar -xzvf zstd-portable.tar.gz && "
            "rm boot.tar boot-env.tar.zst zstd-portable.tar.gz"
        )
        result = manager.run_command_on_pod("node2-a40", extract_command)
        if not result or result['exit_status'] != 0:
            log.error("Failed to extract boot.tar on node2")
            return

        # Transfer dpo_samples.json to node2
        log.info("Transferring dpo_samples.json to node2")
        source_path = os.path.join(os.getcwd(), 'data', 'output', 'dpo', 'dpo_samples.json')
        dest_path = '/root/training/data/output/dpo/dpo_samples.json'
        if not manager.transfer_file_to_pod("node2-a40", source_path, dest_path):
            log.error(f"Failed to transfer dpo_samples.json to node2")
            return

        # Start training on node2
        log.info("Starting training on node2...")
        train_command = (
            "cd /root/training && "
            "/root/boot-env/bin/python -m scripts.train "
            f"--model {args.model} --method dpo --nnodes 1 --nproc_per_node 1"
        )
        exit_status = manager.run_command_with_stream_on_pod("node2-a40", train_command, log.info)

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
        manager.cleanup_all_pods()

if __name__ == "__main__":
    main()