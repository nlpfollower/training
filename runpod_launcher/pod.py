import time
import os
import paramiko
from scp import SCPClient
from runpod_api import PodAPI
from src.utils.logger import log
from pod_config import PodConfig

class Pod:
    def __init__(self, pod_id: str, pod_config: PodConfig):
        self.pod_id = pod_id  # This is our internal ID
        self.runpod_id = None  # This will store the RunPod's assigned ID
        self.pod_config = pod_config
        self.runpod_api = PodAPI()
        self.pod_data = None
        self.ssh_client = None
        self.api_key = None  # Store the API key separately

    def create(self, max_retries=3, retry_delay=5):
        for attempt in range(max_retries):
            log.info(f"Pod creation attempt {attempt + 1}/{max_retries}")

            response = self.runpod_api.create_on_demand_pod(self.pod_config.to_dict())

            if response.status_code != 200:
                log.error(f"Failed to create pod. Status code: {response.status_code}")
                if attempt < max_retries - 1:
                    log.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                continue

            response_data = response.json()

            if 'errors' in response_data:
                log.error(f"Error creating pod: {response_data['errors']}")
                if attempt < max_retries - 1:
                    log.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                continue

            pod_data = response_data.get('data', {}).get('podFindAndDeployOnDemand')
            if not pod_data:
                log.error("Pod data is missing from the response")
                if attempt < max_retries - 1:
                    log.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                continue

            self.pod_data = pod_data
            self.runpod_id = self.pod_data['id']
            self.api_key = self.pod_data.get('apiKey')  # Store the API key
            log.info(f"Created pod: Internal ID = {self.pod_id}, RunPod ID = {self.runpod_id}")
            return True

        log.error(f"Failed to create pod after {max_retries} attempts")
        return False

    def wait_for_ready(self, max_retries=30, delay=10):
        for _ in range(max_retries):
            status_response = self.runpod_api.get_pod(self.pod_data['id'])
            pod_data = status_response.json().get('data', {}).get('pod', {})
            if pod_data.get('desiredStatus') == 'RUNNING' and pod_data.get('runtime') is not None:
                self.pod_data = pod_data
                log.info("Pod is running with runtime information.")
                return True
            time.sleep(delay)
        log.error("Pod did not enter RUNNING state.")
        return False

    def establish_ssh_connection(self):
        if not self.pod_data or 'runtime' not in self.pod_data:
            log.error("Pod information is not available")
            return False

        if not self.api_key:
            log.error("API key is not available")
            return False

        ssh_details = next(
            (port for port in self.pod_data['runtime']['ports'] if port['privatePort'] == 22), None)
        if not ssh_details:
            log.error("SSH details not found in pod runtime information")
            return False

        ssh_ip, ssh_port = ssh_details['ip'], ssh_details['publicPort']
        log.info(f"Attempting to connect via SSH to {ssh_ip}:{ssh_port}")

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            self.ssh_client.connect(ssh_ip, port=ssh_port, username='root', password=self.api_key, timeout=30)
            log.info("SSH connection established successfully")
            return True
        except Exception as e:
            log.error(f"Failed to establish SSH connection: {str(e)}")
            return False

    def run_ssh_command(self, command):
        if not self.ssh_client:
            log.error("SSH connection not established")
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
            log.error(f"Failed to execute command on pod: {str(e)}")
            return None

    def run_ssh_command_with_stream(self, command, callback):
        if not self.ssh_client:
            log.error("SSH connection not established")
            return None

        try:
            channel = self.ssh_client.get_transport().open_session()
            channel.exec_command(command)

            while True:
                if channel.exit_status_ready():
                    break
                r, w, e = channel.recv_ready(), channel.recv_stderr_ready(), channel.exit_status_ready()
                if r:
                    output = channel.recv(1024).decode('utf-8')
                    callback(output)
                if w:
                    error = channel.recv_stderr(1024).decode('utf-8')
                    callback(error)
                if e:
                    break
                time.sleep(0.1)

            exit_status = channel.recv_exit_status()
            return exit_status
        except Exception as e:
            log.error(f"Failed to execute command on pod: {str(e)}")
            return None
    def transfer_file_to_remote(self, local_path, remote_path):
        if not self.ssh_client:
            log.error("SSH connection not established")
            return False

        try:
            # Create the remote directory
            remote_dir = os.path.dirname(remote_path)
            self.run_ssh_command(f"mkdir -p {remote_dir}")

            # Transfer the file
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.put(local_path, remote_path)
            log.info(f"File transferred successfully from local:{local_path} to pod:{remote_path}")
            return True
        except Exception as e:
            log.error(f"Failed to transfer file: {str(e)}")
            return False

    def transfer_file_to_local(self, remote_path, local_path):
        if not self.ssh_client:
            log.error("SSH connection not established")
            return False

        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.get(remote_path, local_path)
            log.info(f"File transferred successfully from pod:{remote_path} to local:{local_path}")
            return True
        except Exception as e:
            log.error(f"Failed to transfer file: {str(e)}")
            return False

    def delete_file_on_pod(self, remote_path):
        if not self.ssh_client:
            log.error("SSH connection not established")
            return False

        try:
            _, stdout, stderr = self.ssh_client.exec_command(f"rm {remote_path}")
            exit_status = stdout.channel.recv_exit_status()
            if exit_status == 0:
                log.info(f"File {remote_path} deleted successfully from the pod")
                return True
            else:
                log.error(f"Failed to delete file {remote_path}: {stderr.read().decode('utf-8')}")
                return False
        except Exception as e:
            log.error(f"Failed to delete file: {str(e)}")
            return False

    def cleanup(self):
        if self.ssh_client:
            self.ssh_client.close()

        if self.runpod_id:
            log.info(f"Terminating pod {self.runpod_id}...")
            response = self.runpod_api.terminate_pod(self.runpod_id)
            if response.status_code != 200:
                log.error(f"Failed to terminate pod {self.runpod_id}: {response.status_code}")
                return

            self._wait_for_termination()

    def _wait_for_termination(self, max_retries=10, delay=5):
        for _ in range(max_retries):
            status_response = self.runpod_api.get_pod(self.runpod_id)
            pod_data = status_response.json().get('data', {}).get('pod')
            if pod_data is None or pod_data.get('desiredStatus') == 'TERMINATED':
                log.info(f"Pod {self.runpod_id} termination confirmed")
                return
            time.sleep(delay)
        log.error(f"Failed to confirm pod {self.runpod_id} termination. Please check manually.")