import random
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from PIL import Image
from pydantic import BaseModel

from refiners.fluxion.adapters.palette import Palette
from refiners.training_utils.datasets.latent_diffusion import TextEmbeddingLatentsBaseDataset
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig


class ColorJitterConfig(BaseModel):
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0

class ColorDatasetConfig(HuggingfaceDatasetConfig):
    color_jitter: ColorJitterConfig | None = None
    grayscale: float = 0.0

@dataclass
class PaletteDatasetItem:
    palette: Palette
    text: str
    image: Image.Image
    conditional_flag: bool
    
@dataclass
class DatasetItem:
    palettes: dict[str, Palette]
    image: Image.Image

from torch.nn import Module as TorchModule
from torchvision.transforms import (  # type: ignore
    ColorJitter,
    Compose,
    RandomCrop,
    RandomGrayscale,
    RandomHorizontalFlip,
)

TextEmbeddingPaletteLatentsBatch = List[PaletteDatasetItem]

DEFAULT_SAMPLING= {
    "palette_1": 1.0,
    "palette_2": 2.0,
    "palette_3": 3.0,
    "palette_4": 4.0,
    "palette_5": 5.0,
    "palette_6": 6.0,
    "palette_7": 7.0,
    "palette_8": 8.0
}

class SamplingByPalette:
    palette_1: float= 0.0
    palette_2: float= 0.0
    palette_3: float= 0.0
    palette_4: float= 0.0
    palette_5: float= 0.0
    palette_6: float= 0.0
    palette_7: float= 0.0
    palette_8: float= 0.0
    
    def __init__(self, sampling: dict[str, float] = DEFAULT_SAMPLING) -> None:
        for key in sampling:
            self.__setattr__(key, sampling[key])


class PaletteDataset(TextEmbeddingLatentsBaseDataset[TextEmbeddingPaletteLatentsBatch]):
    def __init__(
        self,
        config: HuggingfaceDatasetConfig,
        sampling_by_palette: SamplingByPalette = SamplingByPalette(),
        unconditional_sampling_probability: float = 0.2,
    ) -> None:
        self.sampling_by_palette = sampling_by_palette
        super().__init__(
            config=config,
            unconditional_sampling_probability=unconditional_sampling_probability,
        )

    def __getitem__(self, index: int) -> TextEmbeddingPaletteLatentsBatch:
        
        item = self.hf_dataset[index]
        
        resized_image = self.resize_image(
            image=item['image'],
            min_size=self.config.resize_image_min_size,
            max_size=self.config.resize_image_max_size,
        )

        image = self.process_image(resized_image)
        
        caption_key = self.config.caption_key
        caption = item[caption_key]
        (caption_processed, conditional_flag) = self.process_caption(caption)   
        
        return [
            PaletteDatasetItem(
                palette=self.process_palette(item),
                text=caption_processed,
                image=image,
                conditional_flag=conditional_flag
            )
        ]
    
    def random_palette_size(self) -> int:
        choices = range(1, 9)        
        weights_list : List[float] = []
        for i in choices:
            if hasattr(self.sampling_by_palette, f"palette_{i}"):
                weight = getattr(self.sampling_by_palette, f"palette_{i}")
                weights_list.append(weight)
        
        weights = np.array(weights_list)
        sum = weights.sum()
        probabilities = weights / sum
        palette_index = int(random.choices(choices, probabilities, k=1)[0])
        return palette_index
    
    def process_palette(self, item: DatasetItem) -> Palette:
        palette_color: list[Color] = item['palettes'][str(self.random_palette_size())]
        palette: Palette = [(color, 1.0/len(palette_color)) for color in palette_color]
        return palette

    def build_image_processor(self) -> Callable[[Image.Image], Image.Image]:
        # TODO: make this configurable and add other transforms
        transforms: list[TorchModule] = []
        if self.config.random_crop:
            transforms.append(RandomCrop(size=512))
        if self.config.horizontal_flip:
            transforms.append(RandomHorizontalFlip(p=0.5))
        if self.config.color_jitter is not None:
            transforms.append(ColorJitter(brightness=self.config.color_jitter.brightness, contrast=self.config.color_jitter.contrast, saturation=self.config.color_jitter.saturation, hue=self.config.color_jitter.hue))
        if self.config.grayscale > 0:
            transforms.append(RandomGrayscale(p=self.config.grayscale))            
        if not transforms:
            return lambda image: image
        return Compose(transforms)

    def extract_palette(self, item: DatasetItem) -> Palette:
        
        palette: Palette = item.palettes[str(self.random_palette_size())]
        return palette 
       
    def get_palette(self, index: int) -> Palette:
        item = self.hf_dataset[index]
        return self.process_palette(item)

    def collate_fn(self, batch: list[TextEmbeddingPaletteLatentsBatch]) -> TextEmbeddingPaletteLatentsBatch:
        return [item for sublist in batch for item in sublist]
