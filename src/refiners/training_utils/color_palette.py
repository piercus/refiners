from dataclasses import dataclass
from functools import cached_property
from random import randint
from typing import Any

from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch import Tensor, cat, randn, tensor
from torch.utils.data import Dataset

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.color_palette import ColorPaletteEncoder, SD1ColorPaletteAdapter
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.latent_diffusion import (
    DPMSolver,
    StableDiffusion_1,
)
from refiners.training_utils.callback import Callback
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig
from refiners.training_utils.latent_diffusion import (
    CaptionImage,
    FinetuneLatentDiffusionBaseConfig,
    LatentDiffusionBaseTrainer,
    TestDiffusionBaseConfig,
    TextEmbeddingLatentsBaseDataset,
    TextEmbeddingLatentsBatch,
)
from refiners.training_utils.wandb import WandbLoggable


class ColorPaletteConfig(BaseModel):
    model_dim: int
    trigger_phrase: str = ""
    use_only_trigger_probability: float = 0.0
    max_colors: int


class ColorPalettePromptConfig(BaseModel):
    text: str
    color_palette: list[list[float]]


class ColorPaletteDatasetConfig(HuggingfaceDatasetConfig):
    local_folder: str = "data/color-palette"


class TestColorPaletteConfig(TestDiffusionBaseConfig):
    prompts: list[ColorPalettePromptConfig]


@dataclass
class TextEmbeddingColorPaletteLatentsBatch(TextEmbeddingLatentsBatch):
    text_embeddings: Tensor
    latents: Tensor
    color_palette_embeddings: Tensor


class CaptionPaletteImage(CaptionImage):
    palette_1: list[list[float]]
    palette_2: list[list[float]]
    palette_3: list[list[float]]
    palette_4: list[list[float]]
    palette_5: list[list[float]]
    palette_6: list[list[float]]
    palette_7: list[list[float]]
    palette_8: list[list[float]]


class ColorPaletteDataset(TextEmbeddingLatentsBaseDataset[TextEmbeddingColorPaletteLatentsBatch]):
    def __init__(
        self,
        trainer: "ColorPaletteLatentDiffusionTrainer",
    ) -> None:
        super().__init__(trainer=trainer)
        self.trigger_phrase = trainer.config.color_palette.trigger_phrase
        self.use_only_trigger_probability = trainer.config.color_palette.use_only_trigger_probability
        logger.info(f"Trigger phrase: {self.trigger_phrase}")
        self.color_palette_encoder = trainer.color_palette_encoder

    def get_color_palette(self, index: int) -> Tensor:
        # Randomly pick a palette between 1 and 8
        palette_index = randint(1, 8)
        return tensor([self.dataset[index][f"palette_{palette_index}"]])

    def __getitem__(self, index: int) -> TextEmbeddingColorPaletteLatentsBatch:
        caption = self.get_caption(index=index, caption_key=self.config.dataset.caption_key)
        color_palette = self.get_color_palette(index=index)
        image = self.get_image(index=index)
        resized_image = self.resize_image(
            image=image,
            min_size=self.config.dataset.resize_image_min_size,
            max_size=self.config.dataset.resize_image_max_size,
        )
        processed_image = self.process_image(resized_image)
        latents = self.lda.encode_image(image=processed_image)
        processed_caption = self.process_caption(caption=caption)

        clip_text_embedding = self.text_encoder(processed_caption)
        color_palette_embedding = self.color_palette_encoder(color_palette)
        return TextEmbeddingColorPaletteLatentsBatch(
            text_embeddings=clip_text_embedding, latents=latents, color_palette_embeddings=color_palette_embedding
        )

    def collate_fn(self, batch: list[TextEmbeddingColorPaletteLatentsBatch]) -> TextEmbeddingColorPaletteLatentsBatch:
        text_embeddings = cat(tensors=[item.text_embeddings for item in batch])
        latents = cat(tensors=[item.latents for item in batch])
        color_palette_embeddings = cat(tensors=[item.color_palette_embeddings for item in batch])
        return TextEmbeddingColorPaletteLatentsBatch(
            text_embeddings=text_embeddings, latents=latents, color_palette_embeddings=color_palette_embeddings
        )


class ColorPaletteLatentDiffusionConfig(FinetuneLatentDiffusionBaseConfig):
    color_palette: ColorPaletteConfig
    test_color_palette: TestColorPaletteConfig


class ColorPaletteLatentDiffusionTrainer(
    LatentDiffusionBaseTrainer[ColorPaletteLatentDiffusionConfig, TextEmbeddingColorPaletteLatentsBatch]
):
    @cached_property
    def color_palette_encoder(self) -> ColorPaletteEncoder:
        assert (
            self.config.models["color_palette_encoder"] is not None
        ), "The config must contain a color_palette_encoder entry."

        # TO FIX : connect this to unet cross attention embedding dim
        EMBEDDING_DIM = 768

        return ColorPaletteEncoder(
            max_colors=self.config.color_palette.max_colors,
            embedding_dim=EMBEDDING_DIM,
            model_dim=self.config.color_palette.model_dim,
            device=self.device,
        )

    @cached_property
    def color_palette_adapter(self) -> ColorPaletteEncoder:
        adapter = SD1ColorPaletteAdapter(target=self.unet, color_palette_encoder=self.color_palette_encoder)

        return adapter

    def __init__(
        self,
        config: ColorPaletteLatentDiffusionConfig,
        callbacks: "list[Callback[Any]] | None" = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks)
        self.callbacks.extend((LoadColorPalette(), SaveColorPalette()))

    def load_dataset(self) -> Dataset[TextEmbeddingColorPaletteLatentsBatch]:
        return ColorPaletteDataset(trainer=self)

    def load_models(self) -> dict[str, fl.Module]:
        return {
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "lda": self.lda,
            "color_palette_encoder": self.color_palette_encoder
        }

    def compute_loss(self, batch: TextEmbeddingColorPaletteLatentsBatch) -> Tensor:
        text_embeddings, latents, color_palette_embeddings = (
            batch.text_embeddings,
            batch.latents,
            batch.color_palette_embeddings,
        )
        timestep = self.sample_timestep()
        noise = self.sample_noise(size=latents.shape, dtype=latents.dtype)
        noisy_latents = self.ddpm_scheduler.add_noise(x=latents, noise=noise, step=self.current_step)
        self.unet.set_timestep(timestep=timestep)

        self.unet.set_clip_text_embedding(clip_text_embedding=text_embeddings)
        self.color_palette_adapter.set_color_palette_embedding(color_palette_embeddings)

        prediction = self.unet(noisy_latents)
        loss = self.mse_loss(prediction, noise)
        return loss

    @cached_property
    def sd(self) -> StableDiffusion_1:
        scheduler = DPMSolver(
            device=self.device,
            num_inference_steps=self.config.test_color_palette.num_inference_steps,
        )

        self.sharding_manager.add_device_hooks(scheduler, scheduler.device)

        return StableDiffusion_1(unet=self.unet, lda=self.lda, clip_text_encoder=self.text_encoder, scheduler=scheduler)

    def compute_evaluation(self) -> None:
        sd = self.sd
        prompts = self.config.test_color_palette.prompts
        num_images_per_prompt = self.config.test_color_palette.num_images_per_prompt
        images: dict[str, WandbLoggable] = {}
        for prompt in prompts:
            canvas_image: Image.Image = Image.new(mode="RGB", size=(512, 512 * num_images_per_prompt))
            image_name = prompt.text + str(prompt.color_palette)
            for i in range(num_images_per_prompt):
                logger.info(
                    f"Generating image {i+1}/{num_images_per_prompt} for prompt: {prompt.text} and palette {prompt.color_palette}"
                )
                x = randn(1, 4, 64, 64)

                cfg_clip_text_embedding = sd.compute_clip_text_embedding(text=prompt.text).to(device=self.device)
                cfg_color_palette_embedding = self.color_palette_encoder.compute_color_palette_embedding(
                    [prompt.color_palette]
                )

                self.color_palette_adapter.set_color_palette_embedding(cfg_color_palette_embedding)

                for step in sd.steps:
                    x = sd(
                        x,
                        step=step,
                        clip_text_embedding=cfg_clip_text_embedding,
                    )
                canvas_image.paste(sd.lda.decode_latents(x=x), box=(0, 512 * i))

            images[image_name] = canvas_image
        self.log(data=images)


class LoadColorPalette(Callback[ColorPaletteLatentDiffusionTrainer]):
    def on_train_begin(self, trainer: ColorPaletteLatentDiffusionTrainer) -> None:
        trainer.color_palette_adapter.inject()


class SaveColorPalette(Callback[ColorPaletteLatentDiffusionTrainer]):
    def on_checkpoint_save(self, trainer: ColorPaletteLatentDiffusionTrainer) -> None:
        tensors: dict[str, Tensor] = {}
        metadata: dict[str, str] = {}

        model = trainer.unet
        if model.parent is None:
            raise ValueError("The model must have a parent.")
        adapter = model.parent

        tensors = {f"unet.{i:03d}": w for i, w in enumerate(adapter.weights)}
        metadata = {f"unet_targets": ",".join(adapter.sub_targets)}

        save_to_safetensors(
            path=trainer.ensure_checkpoints_save_folder / f"step{trainer.clock.step}.safetensors",
            tensors=tensors,
            metadata=metadata,
        )
