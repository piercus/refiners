from pathlib import Path

from pydantic import BaseModel

from refiners.training_utils.config import BaseConfig, ModelConfig
from refiners.training_utils.wandb import WandbConfig


class LatentDiffusionConfig(BaseModel):
    unconditional_sampling_probability: float = 0.2
    offset_noise: float = 0.1

class SDModelConfig(ModelConfig):
    unet: Path
    text_encoder: Path
    lda: Path

class IPAdapterConfig(ModelConfig):
    weights: Path
    image_encoder_weights: Path
    fine_grained: bool = True


class Config(BaseConfig):
    wandb: WandbConfig
    latent_diffusion: LatentDiffusionConfig
    data: Path
    offload_to_cpu: bool = False
    sd: SDModelConfig
    ip_adapter: IPAdapterConfig
