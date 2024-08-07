from pod_manager import PodManager
from src.utils.logger import log

def main():
    with PodManager() as manager:
        # Create pod with new volume
        new_volume_config = {
            "name": "new-volume-pod",
            "imageName": "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04", # Use an actual image name instead of templateId
            "gpuCount": 1,
            "volumeInGb": 50,
            "containerDiskInGb": 10,  # Add this if you want to specify the container disk size
            "env": [{"key": "EXAMPLE_ENV", "value": "example_value"}],  # Add any environment variables if needed
            "ports": "8888/http,22/tcp",  # Specify the ports you want to expose
            "volumeMountPath": "/workspace"  # Specify where you want to mount the volume
        }
        log.info(f"Attempting to create new volume pod with config: {new_volume_config}")
        new_volume_pod = manager.create_pod(new_volume_config)
        if not new_volume_pod:
            log.error("Failed to create new volume pod")
            return

        # Create pod with existing volume
        existing_volume_config = {
            "name": "existing-volume-pod",
            "networkVolumeId": "71thl7fnmp",
            "gpuCount": 1,
        }
        log.info(f"Attempting to create existing volume pod with config: {existing_volume_config}")
        existing_volume_pod = manager.create_pod(existing_volume_config)
        if not existing_volume_pod:
            log.error("Failed to create existing volume pod")
            return

        # Create test file
        result = new_volume_pod.run_ssh_command("dd if=/dev/urandom of=/tmp/testfile bs=1M count=1024")
        if not result or result['exit_status'] != 0:
            log.error("Failed to create test file")
            return

        # Transfer file
        if not manager.transfer_file_between_pods(new_volume_pod, existing_volume_pod, "/tmp/testfile", "/tmp/"):
            log.error("Failed to transfer file between pods")
            return

        # Verify transfer
        result = existing_volume_pod.run_ssh_command("ls -la /tmp/testfile")
        if result and result['exit_status'] == 0:
            log.info(f"File transfer verified:\n{result['output']}")
        else:
            log.error("Failed to verify file transfer")
            return

        log.info("File transfer test completed successfully")

if __name__ == "__main__":
    main()