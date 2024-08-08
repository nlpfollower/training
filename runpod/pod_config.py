from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class PodConfig:
    name: str
    imageName: str
    containerDiskInGb: int
    volumeInGb: int
    cloudType: str
    countryCode: str
    minDownload: int
    allowedCudaVersions: List[str]
    ports: str
    gpuTypeId: str
    volumeMountPath: str
    networkVolumeId: str
    env: List[Dict[str, str]] = field(default_factory=list)
    gpuCount: int = 1
    startJupyter: bool = False
    startSsh: bool = True
    dockerArgs: str = ""

    def to_dict(self) -> Dict[str, any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# Preset configurations
DEFAULT_CONFIG = {
    'name': 'default-training',
    'imageName': 'runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04',
    'containerDiskInGb': 10,
    'volumeInGb': 100,
    'cloudType': 'SECURE',
    'countryCode': 'SK,SE,BE,BG,CA,CZ,NL',
    'minDownload': 700,
    'allowedCudaVersions': ['11.8', '12.0', '12.1', '12.2', '12.3'],
    'ports': '22/tcp,3000/http,6006/http,8888/http',
    'gpuTypeId': 'NVIDIA A40',
    'volumeMountPath': '/workspace',
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

    config['networkVolumeId'] = network_volume_id

    if model_name:
        config['name'] = f"{config['name']}-{model_name}"
        config['env'].append({"key": "MODEL_NAME", "value": model_name})

    # Override any config values with provided kwargs
    config.update(kwargs)

    return PodConfig(**config)
