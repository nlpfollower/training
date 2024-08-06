import time
import signal
import paramiko
from runpod_api import PodAPI
from src.utils.logger import log as logger

class PodManager:
    NAME = 'pythia-training'
    IMAGE_NAME = 'runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04'
    OS_DISK_SIZE_GB = 10
    PERSISTENT_DISK_SIZE_GB = 100
    CLOUD_TYPE = 'SECURE'
    COUNTRY_CODE = 'SK,SE,BE,BG,CA,CZ,NL'
    MIN_DOWNLOAD = 700
    ALLOWED_CUDA_VERSIONS = ['11.8', '12.0', '12.1', '12.2', '12.3']
    PORTS = '22/tcp,3000/http,6006/http,8888/http'
    PREFERRED_GPUS = ['NVIDIA A40', 'NVIDIA A100 80GB PCIe']

    def __init__(self):
        self.runpod_api = PodAPI()
        self.pod = None
        self.ssh_client = None
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        raise KeyboardInterrupt

    def create_pod(self):
        for gpu in self.PREFERRED_GPUS:
            pod_config = self._get_pod_config(gpu)
            response = self.runpod_api.create_on_demand_pod(pod_config)

            if response.status_code == 200:
                response_json = response.json()
                if 'data' in response_json and 'podFindAndDeployOnDemand' in response_json['data']:
                    self.pod = response_json['data']['podFindAndDeployOnDemand']
                    self.api_key = self.pod.get('apiKey')  # Store the API key
                    logger.info("Created pod: ID = {}, GPU = {}", self.pod['id'],
                                self.pod['machine']['podHostId'].split('-')[1])
                    return True

            logger.error("Failed to create pod with GPU: {}", gpu)

        return False

    def _get_pod_config(self, gpu_type_id):
        return {
            "countryCode": self.COUNTRY_CODE,
            "minDownload": self.MIN_DOWNLOAD,
            "allowedCudaVersions": self.ALLOWED_CUDA_VERSIONS,
            "gpuCount": 1,
            "volumeInGb": self.PERSISTENT_DISK_SIZE_GB,
            "containerDiskInGb": self.OS_DISK_SIZE_GB,
            "gpuTypeId": gpu_type_id,
            "cloudType": self.CLOUD_TYPE,
            "supportPublicIp": True,
            "name": self.NAME,
            "dockerArgs": "",
            "ports": self.PORTS,
            "volumeMountPath": "/workspace",
            "networkVolumeId": "71thl7fnmp",
            "imageName": self.IMAGE_NAME,
            "startJupyter": True,
            "startSsh": True,
            "env": [
                {"key": "PYTHONUNBUFFERED", "value": "1"},
                {"key": "PYTHONPATH", "value": "/workspace/training"}
            ]
        }

    def wait_for_pod_ready(self):
        max_retries = 30
        for _ in range(max_retries):
            status_response = self.runpod_api.get_pod(self.pod['id'])
            pod_data = status_response.json().get('data', {}).get('pod', {})
            pod_status = pod_data.get('desiredStatus')
            runtime = pod_data.get('runtime')

            if pod_status == 'RUNNING' and runtime is not None:
                logger.info("Pod is running with runtime information.")
                self.pod = pod_data  # Update the pod information
                return True
            elif pod_status in ['STOPPED', 'FAILED']:
                logger.error("Pod failed to start. Status: {}", pod_status)
                return False

            time.sleep(10)

        logger.error("Max retries reached. Pod did not enter RUNNING state.")
        return False

    def establish_ssh_connection(self):
        if not self.pod or 'runtime' not in self.pod or not self.api_key:
            logger.error("Pod information or API key is not available")
            return False

        ssh_details = next((port for port in self.pod['runtime']['ports'] if port['type'] == 'tcp'), None)
        if not ssh_details:
            logger.error("SSH details not found in pod runtime information")
            return False

        ssh_ip = ssh_details['ip']
        ssh_port = ssh_details['publicPort']

        logger.info(f"Attempting to connect via SSH to {ssh_ip}:{ssh_port}")

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            self.ssh_client.connect(ssh_ip, port=ssh_port, username='root', password=self.api_key, timeout=30)
            logger.info("SSH connection established successfully")
            return True
        except Exception as e:
            logger.error("Failed to establish SSH connection: {}", str(e))
            return False

    def run_ssh_command(self, command):
        if not self.ssh_client:
            logger.error("SSH connection not established")
            return None

        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')

            return {
                'exit_status': exit_status,
                'output': output,
                'error': error
            }
        except Exception as e:
            logger.error("Failed to execute command on pod: {}", str(e))
            return None

    def cleanup(self):
        if self.ssh_client:
            self.ssh_client.close()

        if self.pod:
            logger.info("Terminating pod {}...", self.pod['id'])
            response = self.runpod_api.terminate_pod(self.pod['id'])
            if response.status_code == 200:
                logger.info("Pod {} terminated successfully", self.pod['id'])
            else:
                logger.error("Failed to terminate pod {}: {}", self.pod['id'], response.status_code)

            # Verify pod termination
            max_retries = 10
            for _ in range(max_retries):
                status_response = self.runpod_api.get_pod(self.pod['id'])
                pod_data = status_response.json().get('data', {}).get('pod')
                if pod_data is None:
                    logger.info("Pod {} no longer exists, termination confirmed", self.pod['id'])
                    break
                pod_status = pod_data.get('desiredStatus')
                if pod_status == 'TERMINATED':
                    logger.info("Pod {} termination confirmed", self.pod['id'])
                    break
                time.sleep(5)
            else:
                logger.error("Failed to confirm pod termination. Please check manually.")