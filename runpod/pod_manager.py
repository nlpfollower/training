import signal
import sys
import atexit
from typing import Dict, Callable, Any, List
from pod import Pod
from pod_config import create_pod_config
from src.utils.logger import log as logger
import concurrent.futures
from queue import Queue

class PodManager:
    def __init__(self):
        self.pods: Dict[str, Pod] = {}
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup_all_pods)

    def __del__(self):
        self.cleanup_all_pods()

    def _signal_handler(self, signum, frame):
        logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
        self.cleanup_all_pods()
        sys.exit(0)

    def create_pod(self, internal_id: str, network_volume_id: str, model_name: str = None, preset: str = 'default',
                   **kwargs) -> bool:
        if internal_id in self.pods:
            logger.error(f"Pod with internal id {internal_id} already exists")
            return False

        pod_config = create_pod_config(network_volume_id, model_name, preset, **kwargs)
        pod = Pod(internal_id, pod_config)
        if pod.create():
            self.pods[internal_id] = pod
        else :
            logger.error(f"Failed to create pod with internal id {internal_id}")
            return False
        if pod.wait_for_ready() and pod.establish_ssh_connection():
            return True
        return False

    def get_pod(self, internal_id: str) -> Pod:
        return self.pods.get(internal_id)

    def cleanup_pod(self, internal_id: str) -> None:
        pod = self.get_pod(internal_id)
        if pod:
            logger.info(f"Cleaning up pod {internal_id} (RunPod ID: {pod.runpod_id})")
            pod.cleanup()
            del self.pods[internal_id]
        else:
            logger.warning(f"Pod with internal id {internal_id} not found or already cleaned up")

    def cleanup_all_pods(self) -> None:
        logger.info("Cleaning up all pods")
        for internal_id in list(self.pods.keys()):
            self.cleanup_pod(internal_id)
        logger.info("All pods have been terminated")

    def run_command_on_pod(self, pod_id: str, command: str) -> Dict[str, Any]:
        pod = self.get_pod(pod_id)
        if pod:
            return pod.run_ssh_command(command)
        else:
            logger.error(f"Pod with id {pod_id} not found")
            return None

    def run_command_with_stream_on_pod(self, pod_id: str, command: str, callback: Callable[[str], None]):
        pod = self.get_pod(pod_id)
        if pod:
            return pod.run_ssh_command_with_stream(command, callback)
        else:
            logger.error(f"Pod with id {pod_id} not found")
            return None

    def transfer_file_to_pod(self, pod_id: str, local_path: str, remote_path: str) -> bool:
        pod = self.get_pod(pod_id)
        if pod:
            return pod.transfer_file_to_remote(local_path, remote_path)
        else:
            logger.error(f"Pod with id {pod_id} not found")
            return False

    def transfer_file_from_pod(self, pod_id: str, remote_path: str, local_path: str) -> bool:
        pod = self.get_pod(pod_id)
        if pod:
            return pod.transfer_file_to_local(remote_path, local_path)
        else:
            logger.error(f"Pod with id {pod_id} not found")
            return False

    def delete_file_on_pod(self, pod_id: str, remote_path: str) -> bool:
        pod = self.get_pod(pod_id)
        if pod:
            return pod.delete_file_on_pod(remote_path)
        else:
            logger.error(f"Pod with id {pod_id} not found")
            return False

    def bulk_execute(self, operation: Callable[[str], Any], pod_ids: List[str] = None) -> Dict[str, Any]:
        if pod_ids is None:
            pod_ids = list(self.pods.keys())

        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pod_ids)) as executor:
            future_to_pod = {executor.submit(operation, pod_id): pod_id for pod_id in pod_ids}

            # Wait for all futures to complete
            concurrent.futures.wait(future_to_pod.keys())

            for future in future_to_pod:
                pod_id = future_to_pod[future]
                try:
                    results[pod_id] = future.result()
                except Exception as exc:
                    logger.error(f"Operation failed for pod {pod_id}: {exc}")
                    results[pod_id] = None

        return results