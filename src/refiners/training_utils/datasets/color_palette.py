import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from pydantic import BaseModel
from torch import Tensor, cat, tensor, empty

from refiners.fluxion.adapters.color_palette import ColorPaletteEncoder
from refiners.foundationals.clip.text_encoder import CLIPTextEncoder
from refiners.foundationals.latent_diffusion.stable_diffusion_1.model import SD1Autoencoder
from refiners.training_utils.datasets.latent_diffusion import TextEmbeddingLatentsBaseDataset, TextEmbeddingLatentsBatch
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig
from PIL import Image
Color = Tuple[int, int, int]

ColorPalette = List[Color]


@dataclass
class TextEmbeddingColorPaletteLatentsBatch(TextEmbeddingLatentsBatch):
    color_palette_embeddings: Tensor
    images: List[Image.Image]
    texts: List[str]
    color_palettes: List[ColorPalette]


class SamplingByPalette(BaseModel):
    palette_1: float = 1.0
    palette_2: float = 2.0
    palette_3: float = 3.0
    palette_4: float = 4.0
    palette_5: float = 5.0
    palette_6: float = 6.0
    palette_7: float = 7.0
    palette_8: float = 8.0


class ColorPaletteDataset(TextEmbeddingLatentsBaseDataset[TextEmbeddingColorPaletteLatentsBatch]):
    def __init__(
        self,
        config: HuggingfaceDatasetConfig,
        lda: SD1Autoencoder,
        text_encoder: CLIPTextEncoder,
        color_palette_encoder: ColorPaletteEncoder,
        sampling_by_palette: SamplingByPalette = SamplingByPalette(),
        unconditional_sampling_probability: float = 0.2,
    ) -> None:
        self.sampling_by_palette = sampling_by_palette
        self.color_palette_encoder = color_palette_encoder
        super().__init__(
            config=config,
            lda=lda,
            text_encoder=text_encoder,
            unconditional_sampling_probability=unconditional_sampling_probability,
        )

    def __getitem__(self, index: int) -> TextEmbeddingColorPaletteLatentsBatch:
        (latents, image) = self.get_processed_latents(index)
        
        caption = self.get_caption(index=index)

        (processed_caption, conditionnal_flag) = self.process_caption(caption=caption)

        if not conditionnal_flag:
            clip_text_embedding = self.text_encoder(caption)
            color_palette = []
        else:
            clip_text_embedding = self.text_encoder(processed_caption)
            color_palette = self.get_color_palette(index)
        
        return TextEmbeddingColorPaletteLatentsBatch(
            text_embeddings=clip_text_embedding, 
            latents=latents, 
            color_palette_embeddings=self.color_palette_encoder([color_palette]),
            images=[image],
            texts=[caption],
            color_palettes=[color_palette]
        )

    def process_text_embedding_and_palette(self, index: int) -> tuple[Tensor, Tensor]:
        caption = self.get_caption(index=index)

        (processed_caption, conditionnal_flag) = self.process_caption(caption=caption)

        if not conditionnal_flag:
            return (self.text_encoder(caption), self.color_palette_encoder([[]]))

        clip_text_embedding = self.text_encoder(processed_caption)
        color_palette_embedding = self.get_processed_palette(index)
        return (clip_text_embedding, color_palette_embedding)

    def get_processed_palette(self, index: int) -> Tensor:
 
        return self.color_palette_encoder([self.get_color_palette(index)])
    
    def get_color_palette(self, index: int) -> ColorPalette:
        choices = range(1, 9)
        weights = np.array([getattr(self.sampling_by_palette, f"palette_{i}") for i in choices])
        sum = weights.sum()
        probabilities = weights / sum
        palette_index = int(random.choices(choices, probabilities, k=1)[0])
        item = self.hf_dataset[index]
        palette: ColorPalette = item[f"palettes"][str(palette_index)]
        return palette

    def collate_fn(self, batch: list[TextEmbeddingColorPaletteLatentsBatch]) -> TextEmbeddingColorPaletteLatentsBatch:
        text_embeddings = cat(tensors=[item.text_embeddings for item in batch])
        latents = cat(tensors=[item.latents for item in batch])
        color_palette_embeddings = cat(tensors=[item.color_palette_embeddings for item in batch])
        return TextEmbeddingColorPaletteLatentsBatch(
            text_embeddings=text_embeddings, 
            latents=latents, 
            color_palette_embeddings=color_palette_embeddings,
            images=[image for item in batch for image in item.images],
            texts=[text for item in batch for text in item.texts],
            color_palettes=[color_palette for item in batch for color_palette in item.color_palettes],
        )
