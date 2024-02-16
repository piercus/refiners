from pathlib import Path
from pydantic import BaseModel
from refiners.training_utils.config import BaseConfig, ModelConfig
from refiners.training_utils.wandb import WandbConfig
from ip_adapter_lora.config import LatentDiffusionConfig, SDModelConfig

class IPAdapterConfig(ModelConfig):
    weights: Path

class Config(BaseConfig):
    wandb: WandbConfig
    latent_diffusion: LatentDiffusionConfig
    data: Path
    offload_to_cpu: bool = False
    sd: SDModelConfig
    ip_adapter: IPAdapterConfig
