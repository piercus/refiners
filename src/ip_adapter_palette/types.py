from ip_adapter_palette.utils import AbstractBatchInput, AbstractBatchOutput
from PIL import Image
from typing import TypedDict

Color = tuple[int, int, int]
ColorWeight = float
PaletteCluster = tuple[Color, ColorWeight]
Palette = list[PaletteCluster]

class ImageAndPalette(TypedDict):
    image: Image.Image
    palette: list[Color]

class BatchInput(AbstractBatchInput):
    _list_keys: list[str] = ["source_palettes", "source_prompts", "source_images", "db_indexes"]
    _tensor_keys: dict[str, tuple[int, ...]] = {
        "text_embeddings": (77, 768)
    }

class BatchOutput(AbstractBatchOutput[BatchInput]):    
    _list_keys: list[str] = ["source_palettes", "source_prompts", "source_images", "db_indexes", "result_palettes"]
    _tensor_keys: dict[str, tuple[int, ...]] = {
        "text_embeddings": (77, 768),
        "result_images": (3, 512, 512)
    }