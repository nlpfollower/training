import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class APIConfig:
    gpt_api_key: str = os.getenv('GPT_API_KEY')
    gpt_org_id: str = os.getenv('GPT_ORG_ID')
    gpt_project_id: str = os.getenv('GPT_PROJECT_ID')
    gpt_model: str = "gpt-4o-mini"
    runpod_api_key: str = os.getenv('RUNPOD_API_KEY')

@dataclass
class PathConfig:
    base_dir: str = BASE_DIR
    input_file: str = os.path.join(BASE_DIR, 'data', 'input', 'conversations.json')
    output_dir: str = os.path.join(BASE_DIR, 'data', 'output')
    dpo_output_dir: str = os.path.join(BASE_DIR, 'data', 'output', 'dpo')
    kto_output_dir: str = os.path.join(BASE_DIR, 'data', 'output', 'kto')
    spft_output_dir: str = os.path.join(BASE_DIR, 'data', 'output', 'spft')
    stats_dir: str = os.path.join(BASE_DIR, 'data', 'stats')
    local_weights_dir: str = os.path.join(BASE_DIR, 'models', 'weights')
    checkpoint_dir: str = os.path.join(BASE_DIR, 'models', 'checkpoints')
    shared_model_dir: str = os.path.join(BASE_DIR, 'models', 'shared_models')
    node_specific_dir: str = os.path.join(BASE_DIR, 'models', 'node_specific_models')

@dataclass
class SimilarityConfig:
    window_size: int = 5
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "Jaccard": 0.65,
        "Sequence": 0.65,
        "BERT": 0.925,
        "USE": 0.9,
        "TF-IDF": 0.9
    })

@dataclass
class SentimentConfig:
    model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    max_length: int = 512
    nltk_threshold: float = 0.5
    roberta_threshold: float = 0.5

@dataclass
class ModelConfig:
    name: str
    model_path: str
    tokenizer_path: str
    max_sequence_length: int = 2048

@dataclass
class TrainingConfig:
    method: str = "dpo"  # Can be 'dpo', 'kto', or 'sft'
    batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-7
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    beta: float = 0.2
    warmup_steps: int = 100
    eval_split: float = 0.1
    save_interval: int = 1  # Save checkpoint every n epochs
    eval_interval: int = 1  # Run evaluation every n epochs
    num_workers: int = 4
    distributed: bool = False
    world_size: int = 1
    nodes: int = 1
    master_addr: str = 'localhost'
    master_port: str = '12355'
    node_rank: int = 0

@dataclass
class FSDPConfig:
    enabled: bool = False
    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: Optional[str] = None
    activation_checkpointing: bool = False
    cpu_offload: bool = False
    flatten_parameters: bool = True
    move_params_to_cpu: bool = False
    compute_dtype: str = "float32"

@dataclass
class Config:
    api: APIConfig = field(default_factory=APIConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="llama3",
        model_path="models/Meta-HF-Llama-3.1-8B-Instruct",
        tokenizer_path="src/model/llama_tokenizer.model"
    ))
    training: TrainingConfig = field(default_factory=TrainingConfig)

MODEL_PRESETS = {
    "llama3": ModelConfig(
        name="llama3",
        model_path="models/Meta-HF-Llama-3.1-8B-Instruct",
        tokenizer_path="src/model/llama_tokenizer.model",
        max_sequence_length=2048
    ),
    "pythia-160m": ModelConfig(
        name="pythia-160m",
        model_path="EleutherAI/pythia-160m",
        tokenizer_path="EleutherAI/pythia-160m",
        max_sequence_length=2048
    )
}

def get_config() -> Config:
    config = Config()
    # Ensure directories exist
    for directory in [config.paths.dpo_output_dir, config.paths.kto_output_dir,
                      config.paths.spft_output_dir, config.paths.stats_dir,
                      config.paths.checkpoint_dir, config.paths.shared_model_dir,
                      config.paths.node_specific_dir]:
        os.makedirs(directory, exist_ok=True)
    return config

def update_config(config: Config, **kwargs):
    for key, value in kwargs.items():
        if "." in key:
            section, param = key.split(".", 1)
            if hasattr(config, section):
                section_config = getattr(config, section)
                if hasattr(section_config, param):
                    setattr(section_config, param, value)
                else:
                    print(f"Warning: {section} does not have parameter: {param}")
        elif hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: unknown parameter {key}")

def set_model_preset(config: Config, model_name: str):
    if model_name not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset: {model_name}")
    config.model = MODEL_PRESETS[model_name]

# Usage example:
# config = get_config()
# update_config(config, paths.input_file="/new/path/to/input.json", similarity.window_size=10)