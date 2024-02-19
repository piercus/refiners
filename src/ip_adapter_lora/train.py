import random
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from torch.nn import functional as F

from ip_adapter_lora.callback import (
    OffloadToCPU,
    OffloadToCPUConfig,
    SaveBestModel,
    SaveBestModelConfig,
)
from ip_adapter_lora.config import Config, IPAdapterConfig, LatentDiffusionConfig, SDModelConfig
from ip_adapter_lora.latent_diffusion import SD1TrainerMixin
from refiners.fluxion import load_from_safetensors
from refiners.fluxion.utils import no_grad
from refiners.foundationals.latent_diffusion import SD1IPAdapter
from refiners.training_utils import (
    register_callback,
    register_model,
)
from refiners.training_utils.config import (
    OptimizerConfig,
    Optimizers,
    SchedulerConfig,
    SchedulerType,
    TrainingConfig,
)
from refiners.training_utils.trainer import Trainer
from refiners.training_utils.wandb import WandbConfig, WandbLoggable, WandbMixin


@dataclass
class Batch:
    latent: torch.Tensor
    image_embedding: torch.Tensor
    text_embedding: torch.Tensor

    def to(self, device: torch.device, dtype: torch.dtype) -> "Batch":
        return Batch(
            latent=self.latent.to(device=device, dtype=dtype),
            image_embedding=self.image_embedding.to(device=device, dtype=dtype),
            text_embedding=self.text_embedding.to(device=device, dtype=dtype),
        )


class SD1IPLora(Trainer[Config, Batch], WandbMixin, SD1TrainerMixin):
    @register_model()
    def ip_adapter(self, config: IPAdapterConfig) -> SD1IPAdapter:
        logger.info("Loading IP Adapter.")
        weights = load_from_safetensors(config.weights)
        ip_adapter = SD1IPAdapter(
            self.unet,
            weights=weights,
            fine_grained=True,
        ).inject()
        ip_adapter.image_proj.requires_grad_(True)
        for adapter in ip_adapter.sub_adapters:
            adapter.image_key_projection.requires_grad_(True)
            adapter.image_value_projection.requires_grad_(True)

        ip_adapter.clip_image_encoder.load_from_safetensors(
            config.image_encoder_weights
        )
        ip_adapter.clip_image_encoder.requires_grad_(False)
        logger.info("IP Adapter loaded.")

        return ip_adapter

    @register_callback()
    def save_best_model(self, config: SaveBestModelConfig) -> SaveBestModel:
        return SaveBestModel(config)

    @register_callback()
    def offload_to_cpu(self, config: OffloadToCPUConfig) -> OffloadToCPU:
        return OffloadToCPU()

    @cached_property
    def data(self) -> list[Batch]:
        return [
            torch.load(batch).to(device=self.device, dtype=self.dtype)  # type: ignore
            for batch in self.config.data.rglob("*.pt")
        ]

    @cached_property
    @no_grad()
    def unconditional_text_embedding(self) -> torch.Tensor:
        self.text_encoder.to(device=self.device)
        embedding = self.text_encoder("")
        self.text_encoder.to(device="cpu")
        return embedding

    @cached_property
    @no_grad()
    def unconditional_image_embedding(self) -> torch.Tensor:
        zero_embedding = self.ip_adapter.grid_image_encoder(
            torch.zeros(1, 3, 224, 224, device=self.device, dtype=self.dtype)
        )
        return self.ip_adapter.image_proj(zero_embedding)


    def get_item(self, index: int) -> Batch:
        item = self.data[index]
        if (
            random.random()
            < self.config.latent_diffusion.unconditional_sampling_probability
        ):
            item = Batch(
                latent=item.latent,
                image_embedding=self.unconditional_image_embedding,
                text_embedding=self.unconditional_text_embedding,
            )
        return item

    def collate_fn(self, batch: list[Batch]) -> Batch:
        return Batch(
            latent=torch.cat([b.latent for b in batch]),
            image_embedding=torch.cat([b.image_embedding for b in batch]),
            text_embedding=torch.cat([b.text_embedding for b in batch]),
        )

    @property
    def dataset_length(self) -> int:
        return len(self.data)

    def compute_loss(self, batch: Batch) -> torch.Tensor:
        latent, image_embedding, text_embedding = (
            batch.latent,
            batch.image_embedding,
            batch.text_embedding,
        )

        timestep = self.sample_timestep(latent.shape[0])
        noise = self.sample_noise(latent.shape)
        noisy_latents = self.add_noise_to_latents(latent, noise)
        self.unet.set_timestep(timestep)
        self.unet.set_clip_text_embedding(text_embedding)
        self.ip_adapter.set_clip_image_embedding(image_embedding)
        prediction = self.unet(noisy_latents)
        loss = F.mse_loss(input=prediction, target=noise)
        return loss

    def compute_evaluation(self) -> None:
        samples = [batch for batch in self.data[:4]]
        num_images_per_prompt = 4
        images: dict[str, WandbLoggable] = {}
        for j, batch in enumerate(samples):
            canvas_image: Image.Image = Image.new(
                mode="RGB", size=(512, 512 * num_images_per_prompt)
            )
            for i in range(num_images_per_prompt):
                x = torch.randn(1, 4, 64, 64, device=self.device, dtype=self.dtype)

                clip_text_embedding = torch.cat(
                    [self.unconditional_text_embedding, batch.text_embedding]
                )
                image_embedding = torch.cat(
                    [self.unconditional_image_embedding, batch.image_embedding]
                )
                self.ip_adapter.set_clip_image_embedding(image_embedding)

                for step in self.sd.steps:
                    x = self.sd(
                        x,
                        step=step,
                        clip_text_embedding=clip_text_embedding,
                    )

                self.lda.to(device=self.device)
                canvas_image.paste(self.lda.decode_latents(x), (0, i * 512))
                self.lda.to(device="cpu")

            images[str(j)] = canvas_image
        self.wandb_log(data=images)


if __name__ == "__main__":
    hub = Path("/home/trom/weights/")
    sd_path = hub / "stable-diffusion-1-5/"
    sd_config = SDModelConfig(
        unet=sd_path / "unet.safetensors",
        text_encoder=sd_path / "CLIPTextEncoderL.safetensors",
        lda=sd_path / "lda.safetensors",
    )
    ip_adapter_config = IPAdapterConfig(
        weights=hub / "IP-Adapter/ip-adapter-plus_sd15.safetensors",
        image_encoder_weights=hub / "stable-diffusion-2-1-unclip/CLIPImageEncoderH.safetensors",
        fine_grained=True,
    )
    training = TrainingConfig(
        duration="10_000:epoch",  # type: ignore
        batch_size=4,
        device="cuda:0",
        dtype="bfloat16",
        gradient_accumulation="16:step", # type: ignore
        evaluation_interval="2000:step",  # type: ignore
    )
    optimizer = OptimizerConfig(optimizer=Optimizers.AdamW8bit, learning_rate=2e-4)
    scheduler = SchedulerConfig(
        scheduler_type=SchedulerType.CONSTANT_LR,
        warmup="200:step",  # type: ignore
    )

    wandb = WandbConfig(
        notes="Testing latest changes.",
        mode="online",
        project="FG-1438-ip-adapter-lora",
        entity="finegrain",
        group="debug-runs",
        tags=["test"],
        #id="test-sd1-ip-adapter-no-lora-finegrained",
    )

    ld = LatentDiffusionConfig(
        unconditional_sampling_probability=0.2,
        offset_noise=0.1,
    )

    config = Config(
        latent_diffusion=ld,
        offload_to_cpu=True,
        data=Path("/home/trom/trainings/ip-adapter-lora/data/processed_embeddings/"),
        training=training,
        optimizer=optimizer,
        scheduler=scheduler,
        wandb=wandb,
        sd=sd_config,
        ip_adapter=ip_adapter_config,
    )

    trainer = SD1IPLora(config)  # , callbacks=[OffloadToCPU(), SaveBestModel()])
    trainer.train()
