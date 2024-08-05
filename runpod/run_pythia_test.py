#!/usr/bin/env python3
import sys
import json
import time
import argparse
import logging
from runpod import API
from config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def get_args():
    parser = argparse.ArgumentParser(description='Run Pythia training on RunPod')
    parser.add_argument('--model', type=str, default='pythia-160m', help='Model type to train')
    return parser.parse_args()

def create_on_demand_pod(runpod_api, gpu_type_id):
    allowed_cuda_versions_string = ", ".join([f'"{version}"' for version in ALLOWED_CUDA_VERSIONS])

    pod_config = f"""
        countryCode: "{COUNTRY_CODE}",
        minDownload: {MIN_DOWNLOAD},
        allowedCudaVersions: [{allowed_cuda_versions_string}],
        gpuCount: 1,
        volumeInGb: {PERSISTENT_DISK_SIZE_GB},
        containerDiskInGb: {OS_DISK_SIZE_GB},
        gpuTypeId: "{gpu_type_id}",
        cloudType: {CLOUD_TYPE},
        supportPublicIp: true,
        name: "{NAME}",
        dockerArgs: "",
        ports: "{PORTS}",
        volumeMountPath: "/workspace",
        imageName: "{IMAGE_NAME}",
        startJupyter: true,
        startSsh: true,
        env: [
            {{
                key: "PYTHONUNBUFFERED",
                value: "1"
            }}
        ]
    """

    logger.info(f"Attempting to create pod with GPU {gpu_type_id}")
    logger.info(f"Pod config: {pod_config}")
    response = runpod_api.create_on_demand_pod(pod_config)
    logger.info(f"Create pod response: {response.text}")
    resp_json = response.json()

    if response.status_code == 200:
        if 'errors' in resp_json:
            logger.error(f'Error creating pod with GPU {gpu_type_id}: {resp_json["errors"]}')
            return None
        else:
            logger.info(f"Successfully created pod with {gpu_type_id}")
            return resp_json['data']['podFindAndDeployOnDemand']
    else:
        logger.error(f"Failed to create pod with {gpu_type_id}: {response.status_code}")
        logger.error(f"Response: {resp_json}")
        return None


def run_job(runpod_api, pod, job_script):
    logger.info(f"Running job on pod {pod['id']}")
    logger.info(f"Job script: {job_script}")

    # Wait for the pod to be in the RUNNING state and have runtime information
    max_retries = 30  # Maximum number of retries
    retry_count = 0
    while retry_count < max_retries:
        status_response = runpod_api._run_query({
            "query": """
            query ($podId: String!) {
                pod(input: { podId: $podId }) {
                    id
                    desiredStatus
                    runtime {
                        uptimeInSeconds
                    }
                }
            }
            """,
            "variables": {"podId": pod['id']}
        }, auth_required=True)

        status_json = status_response.json()
        logger.info(f"Pod status response: {status_json}")

        if 'data' in status_json and 'pod' in status_json['data']:
            pod_data = status_json['data']['pod']
            pod_status = pod_data['desiredStatus']
            logger.info(f"Pod status: {pod_status}")

            if pod_status == 'RUNNING' and pod_data['runtime'] is not None:
                uptime = pod_data['runtime']['uptimeInSeconds']
                logger.info(f"Pod is running. Uptime: {uptime} seconds")
                break
            elif pod_status in ['STOPPED', 'FAILED']:
                logger.error(f"Pod failed to start. Status: {pod_status}")
                return

        retry_count += 1
        time.sleep(10)  # Poll every 10 seconds

    if retry_count >= max_retries:
        logger.error("Max retries reached. Pod did not enter RUNNING state or runtime information not available.")
        return

    # If the pod is running, we can now execute our script
    exec_response = runpod_api._run_query({
        "query": """
        mutation ($input: PodRunCommandInput!) {
            podRunCommand(input: $input) {
                output
                errorMessage
            }
        }
        """,
        "variables": {
            "input": {
                "podId": pod['id'],
                "command": job_script
            }
        }
    }, auth_required=True)

    exec_json = exec_response.json()
    logger.info(f"Script execution response: {exec_json}")

    if 'data' in exec_json and 'podRunCommand' in exec_json['data']:
        exec_output = exec_json['data']['podRunCommand']
        logger.info(f"Script output: {exec_output['output']}")
        if exec_output['errorMessage']:
            logger.error(f"Script error: {exec_output['errorMessage']}")
    else:
        logger.error(f"Failed to execute script. Response: {exec_json}")

def terminate_pod(runpod_api, pod_id):
    logger.info(f"Terminating pod {pod_id}")
    response = runpod_api.terminate_pod(pod_id)
    if response.status_code == 200:
        logger.info(f"Pod {pod_id} terminated successfully")
    else:
        logger.error(f"Failed to terminate pod {pod_id}: {response.status_code}")
        logger.error(f"Response: {response.json()}")

def main():
    args = get_args()
    config = get_config()
    runpod_api = API()
    pod = None

    try:
        logger.info("Starting Pythia training on RunPod")
        logger.info(f"Model: {args.model}")

        for gpu in PREFERRED_GPUS:
            pod = create_on_demand_pod(runpod_api, gpu)
            if pod:
                logger.info(f"Created pod: ID = {pod['id']}, GPU = {pod['machine']['podHostId'].split('-')[1]}")
                break
        else:
            logger.error("Failed to create pod with any preferred GPU")
            return

        job_script = f"source /root/.bashrc && conda activate base && python -m scripts.train --model {args.model} --method dpo --nnodes 1 --nproc_per_node 1"

        run_job(runpod_api, pod, job_script)

        logger.info("Training completed.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        if pod:
            logger.info(f"Terminating pod {pod['id']}...")
            terminate_pod(runpod_api, pod['id'])
        logger.info("Script execution completed")

if __name__ == '__main__':
    main()