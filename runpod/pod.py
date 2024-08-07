import time
import paramiko
from runpod_api import PodAPI
from src.utils.logger import log as logger

class Pod:
    NAME = 'pythia-training'
    IMAGE_NAME = 'runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04'
    OS_DISK_SIZE_GB = 10
    PERSISTENT_DISK_SIZE_GB = 100
    CLOUD_TYPE = 'SECURE'
    COUNTRY_CODE = 'SK,SE,BE,BG,CA,CZ,NL'
    MIN_DOWNLOAD = 700
    ALLOWED_CUDA_VERSIONS = ['11.8', '12.0', '12.1', '12.2', '12.3']
    PORTS = '2222/tcp,3000/http,6006/http,8888/http'
    PREFERRED_GPUS = ['NVIDIA A40', 'NVIDIA A100 80GB PCIe']

    def __init__(self, pod_config=None):
        self.runpod_api = PodAPI()
        self.pod_data = None
        self.ssh_client = None
        self.custom_ssh_port = 2222
        self.pod_config = pod_config or self._get_default_config()

    def _get_default_config(self):
        return {
            "name": self.NAME,
            "imageName": self.IMAGE_NAME,
            "gpuCount": 1,
            "cloudType": self.CLOUD_TYPE,
            "containerDiskInGb": self.OS_DISK_SIZE_GB,
            "volumeInGb": self.PERSISTENT_DISK_SIZE_GB,
            "volumeMountPath": "/workspace",
            "startJupyter": True,
            "startSsh": True,
            "dockerArgs": "sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config && service ssh restart",
            "ports": self.PORTS,
            "env": [
                {"key": "PYTHONUNBUFFERED", "value": "1"},
                {"key": "PYTHONPATH", "value": "/workspace/training"}
            ]
        }

    def create(self, max_retries=3, retry_delay=5):
        for attempt in range(max_retries):
            response = self.runpod_api.create_on_demand_pod(self.pod_config)
            logger.info(f"Pod creation attempt {attempt + 1}: Status code {response.status_code}")

            if response.status_code != 200:
                logger.error(f"Failed to create pod. Status code: {response.status_code}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                continue

            response_data = response.json()
            logger.debug(f"Full API response: {response_data}")

            if 'errors' in response_data:
                logger.error(f"Error creating pod: {response_data['errors']}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                continue

            pod_data = response_data.get('data', {}).get('podFindAndDeployOnDemand')
            if not pod_data:
                logger.error("Pod data is missing from the response")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                continue

            self.pod_data = pod_data
            logger.info(f"Created pod: ID = {self.pod_data['id']}")
            return True

        logger.error(f"Failed to create pod after {max_retries} attempts")
        return False

    def wait_for_ready(self, max_retries=30, delay=10):
        for _ in range(max_retries):
            status_response = self.runpod_api.get_pod(self.pod_data['id'])
            pod_data = status_response.json().get('data', {}).get('pod', {})
            if pod_data.get('desiredStatus') == 'RUNNING' and pod_data.get('runtime') is not None:
                self.pod_data = pod_data
                logger.info("Pod is running with runtime information.")
                return True
            time.sleep(delay)
        logger.error("Pod did not enter RUNNING state.")
        return False

    def establish_ssh_connection(self):
        if not self.pod_data or 'runtime' not in self.pod_data:
            logger.error("Pod information is not available")
            return False

        ssh_details = next((port for port in self.pod_data['runtime']['ports'] if port['privatePort'] == self.custom_ssh_port), None)
        if not ssh_details:
            logger.error("SSH details not found in pod runtime information")
            return False

        ssh_ip, ssh_port = ssh_details['ip'], ssh_details['publicPort']
        logger.info(f"Attempting to connect via SSH to {ssh_ip}:{ssh_port}")

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            self.ssh_client.connect(ssh_ip, port=ssh_port, username='root', password=self.pod_data['apiKey'], timeout=30)
            logger.info("SSH connection established successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to establish SSH connection: {str(e)}")
            return False

    def run_ssh_command(self, command):
        if not self.ssh_client:
            logger.error("SSH connection not established")
            return None

        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()
            return {
                'exit_status': exit_status,
                'output': stdout.read().decode('utf-8'),
                'error': stderr.read().decode('utf-8')
            }
        except Exception as e:
            logger.error(f"Failed to execute command on pod: {str(e)}")
            return None

    def cleanup(self):
        if self.ssh_client:
            self.ssh_client.close()

        if self.pod_data:
            logger.info(f"Terminating pod {self.pod_data['id']}...")
            response = self.runpod_api.terminate_pod(self.pod_data['id'])
            if response.status_code != 200:
                logger.error(f"Failed to terminate pod {self.pod_data['id']}: {response.status_code}")
                return

            self._wait_for_termination()

    def _wait_for_termination(self, max_retries=10, delay=5):
        for _ in range(max_retries):
            status_response = self.runpod_api.get_pod(self.pod_data['id'])
            pod_data = status_response.json().get('data', {}).get('pod')
            if pod_data is None or pod_data.get('desiredStatus') == 'TERMINATED':
                logger.info(f"Pod {self.pod_data['id']} termination confirmed")
                return
            time.sleep(delay)
        logger.error("Failed to confirm pod termination. Please check manually.")

    def get_ip(self):
        if not self.pod_data or 'runtime' not in self.pod_data:
            logger.error("Pod information is not available")
            return None

        ssh_details = next((port for port in self.pod_data['runtime']['ports'] if port['privatePort'] == self.custom_ssh_port), None)
        return ssh_details['ip'] if ssh_details else None