import time
import signal
import paramiko
from scp import SCPClient
from runpod_api import PodAPI
from src.utils.logger import log as logger

class PodManager:
    def __init__(self, config):
        self.config = config
        self.runpod_api = PodAPI()
        self.pod = None
        self.ssh_client = None
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        raise KeyboardInterrupt

    def create_pod(self):
        for gpu in self.config['PREFERRED_GPUS']:
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
            "countryCode": self.config['COUNTRY_CODE'],
            "minDownload": self.config['MIN_DOWNLOAD'],
            "allowedCudaVersions": self.config['ALLOWED_CUDA_VERSIONS'],
            "gpuCount": 1,
            "volumeInGb": self.config['PERSISTENT_DISK_SIZE_GB'],
            "containerDiskInGb": self.config['OS_DISK_SIZE_GB'],
            "gpuTypeId": gpu_type_id,
            "cloudType": self.config['CLOUD_TYPE'],
            "supportPublicIp": True,
            "name": self.config['NAME'],
            "dockerArgs": "",
            "ports": self.config['PORTS'],
            "volumeMountPath": self.config['VOLUME_MOUNT_PATH'],
            "networkVolumeId": self.config['NETWORK_VOLUME_ID'],
            "imageName": self.config['IMAGE_NAME'],
            "startJupyter": True,
            "startSsh": True,
            "env": self.config['ENV']
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

    def run_ssh_command_with_stream(self, command, callback=None):
        if not self.ssh_client:
            logger.error("SSH connection not established")
            return None

        try:
            # Open a new channel
            channel = self.ssh_client.get_transport().open_session()
            # Execute the command
            channel.exec_command(command)

            # Read the output stream in a loop
            while True:
                if channel.exit_status_ready():
                    break
                r, w, e = channel.recv_ready(), channel.recv_stderr_ready(), channel.exit_status_ready()
                if r:
                    output = channel.recv(1024).decode('utf-8')
                    if callback:
                        callback(output)
                    else:
                        print(output, end='')
                if w:
                    error = channel.recv_stderr(1024).decode('utf-8')
                    if callback:
                        callback(error)
                    else:
                        print(error, end='')
                if e:
                    break
                time.sleep(0.1)

            exit_status = channel.recv_exit_status()
            return {
                'exit_status': exit_status,
                'output': channel.recv(1024).decode('utf-8'),
                'error': channel.recv_stderr(1024).decode('utf-8')
            }
        except Exception as e:
            logger.error("Failed to execute command on pod: {}", str(e))
            return None

    def transfer_file_to_local(self, remote_path, local_path):
        if not self.ssh_client:
            logger.error("SSH connection not established")
            return False

        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.get(remote_path, local_path)
            logger.info(f"File transferred successfully from pod:{remote_path} to local:{local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to transfer file: {str(e)}")
            return False

    def delete_file_on_pod(self, remote_path):
        if not self.ssh_client:
            logger.error("SSH connection not established")
            return False

        try:
            _, stdout, stderr = self.ssh_client.exec_command(f"rm {remote_path}")
            exit_status = stdout.channel.recv_exit_status()
            if exit_status == 0:
                logger.info(f"File {remote_path} deleted successfully from the pod")
                return True
            else:
                logger.error(f"Failed to delete file {remote_path}: {stderr.read().decode('utf-8')}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete file: {str(e)}")
            return False

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