from typing import List, Dict, Optional

class PodConfig:
    def __init__(
        self,
        name: str,
        image_name: str,
        os_disk_size_gb: int,
        persistent_disk_size_gb: int,
        cloud_type: str,
        country_code: str,
        min_download: int,
        allowed_cuda_versions: List[str],
        ports: str,
        preferred_gpus: List[str],
        volume_mount_path: str,
        network_volume_id: str,
        env: List[Dict[str, str]]
    ):
        self.name = name
        self.image_name = image_name
        self.os_disk_size_gb = os_disk_size_gb
        self.persistent_disk_size_gb = persistent_disk_size_gb
        self.cloud_type = cloud_type
        self.country_code = country_code
        self.min_download = min_download
        self.allowed_cuda_versions = allowed_cuda_versions
        self.ports = ports
        self.preferred_gpus = preferred_gpus
        self.volume_mount_path = volume_mount_path
        self.network_volume_id = network_volume_id
        self.env = env

    def to_dict(self) -> Dict[str, any]:
        return {
            'NAME': self.name,
            'IMAGE_NAME': self.image_name,
            'OS_DISK_SIZE_GB': self.os_disk_size_gb,
            'PERSISTENT_DISK_SIZE_GB': self.persistent_disk_size_gb,
            'CLOUD_TYPE': self.cloud_type,
            'COUNTRY_CODE': self.country_code,
            'MIN_DOWNLOAD': self.min_download,
            'ALLOWED_CUDA_VERSIONS': self.allowed_cuda_versions,
            'PORTS': self.ports,
            'PREFERRED_GPUS': self.preferred_gpus,
            'VOLUME_MOUNT_PATH': self.volume_mount_path,
            'NETWORK_VOLUME_ID': self.network_volume_id,
            'ENV': self.env
        }

# Preset configurations
DEFAULT_CONFIG = {
    'name': 'default-training',
    'image_name': 'runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04',
    'os_disk_size_gb': 10,
    'persistent_disk_size_gb': 100,
    'cloud_type': 'SECURE',
    'country_code': 'SK,SE,BE,BG,CA,CZ,NL',
    'min_download': 700,
    'allowed_cuda_versions': ['11.8', '12.0', '12.1', '12.2', '12.3'],
    'ports': '22/tcp,3000/http,6006/http,8888/http',
    'preferred_gpus': ['NVIDIA A40', 'NVIDIA A100 80GB PCIe'],
    'volume_mount_path': '/workspace',
    'env': [
        {"key": "PYTHONUNBUFFERED", "value": "1"},
        {"key": "PYTHONPATH", "value": "/workspace/training"}
    ]
}

PYTHIA_CONFIG = {
    **DEFAULT_CONFIG,
    'name': 'pythia-training',
}

def create_pod_config(
    network_volume_id: str,
    model_name: Optional[str] = None,
    preset: str = 'default',
    **kwargs
) -> PodConfig:
    if preset == 'pythia':
        config = PYTHIA_CONFIG.copy()
    else:
        config = DEFAULT_CONFIG.copy()

    config['network_volume_id'] = network_volume_id

    if model_name:
        config['name'] = f"{config['name']}-{model_name}"
        config['env'].append({"key": "MODEL_NAME", "value": model_name})

    # Override any config values with provided kwargs
    config.update(kwargs)

    return PodConfig(**config)