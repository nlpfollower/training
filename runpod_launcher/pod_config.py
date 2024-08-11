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
    templateId: str
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
DEFAULT_CONFIG = PodConfig(
    name='default-training',
    imageName='runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04',
    containerDiskInGb=10,
    volumeInGb=100,
    cloudType='SECURE',
    countryCode='SK,SE,BE,BG,CA,CZ,NL',
    minDownload=700,
    allowedCudaVersions=['11.8', '12.0', '12.1', '12.2', '12.3'],
    ports='22/tcp,3000/http,6006/http,8888/http',
    templateId='',
    gpuTypeId='NVIDIA A40',
    volumeMountPath='/workspace',
    networkVolumeId='',
    env=[
        {"key": "PYTHONUNBUFFERED", "value": "1"},
        {"key": "PYTHONPATH", "value": "/workspace/training"}
    ]
)


def create_pod_config(
    network_volume_id: str,
    model_name: Optional[str] = None,
    preset: str = 'default',
    **kwargs
) -> PodConfig:
    config = DEFAULT_CONFIG

    # Create a new PodConfig instance with updated values
    updated_config = PodConfig(
        **{**config.to_dict(), 'networkVolumeId': network_volume_id, **kwargs}
    )

    if model_name:
        updated_config.name = f"{updated_config.name}-{model_name}"
        updated_config.env.append({"key": "MODEL_NAME", "value": model_name})

    return updated_config
