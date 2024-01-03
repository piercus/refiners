from typing import Any, Generic, Iterable, TypeVar
from jaxtyping import Float, Int
from torch import Tensor, tensor
import torch
import numpy as np 
from torch.nn.functional import pad
from refiners.foundationals.latent_diffusion.image_prompt import CrossAttentionAdapter
from torch import Tensor, arange, cat, cos, sin, device as Device, dtype as DType,float32, exp
from torch.nn import Parameter as TorchParameter
from torch.nn.init import normal_, zeros_
import math
import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.adapter import Adapter
from refiners.foundationals.latent_diffusion.range_adapter import compute_sinusoidal_embedding


# T = TypeVar("T", bound=fl.Chain)
# TLoraAdapter = TypeVar("TLoraAdapter", bound="LoraAdapter[Any]")  # Self (see PEP 673)

from torch import Tensor

from refiners.foundationals.latent_diffusion.cross_attention import CrossAttentionBlock2d
from refiners.foundationals.latent_diffusion.stable_diffusion_1.unet import SD1UNet
from refiners.foundationals.latent_diffusion.range_adapter import RangeEncoder
T = TypeVar("T", bound="SD1UNet | SDXLUNet")

class ColorPaletteEncoder(fl.Chain):
    def __init__(
        self,
        embedding_dim: int,
        max_colors: int,
        model_dim: int = 256,        
        sinuosidal_embedding_dim: int = 32,
        device: Device | str | None = None,
        dtype: DType | None = None,
        context_key: str = "color_palette_embedding",
    ) -> None:
        
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.max_colors = max_colors
        
        
        super().__init__(
            fl.Linear(
                in_features=3, 
                out_features=model_dim, 
                device=device,
                dtype=dtype
            ),
            fl.Residual(fl.Lambda(self.compute_sinuosoidal_embedding)),
            fl.Linear(in_features=model_dim, out_features=model_dim, device=device, dtype=dtype),
            fl.GeLU(),
            fl.Linear(in_features=model_dim, out_features=embedding_dim, device=device, dtype=dtype),
            fl.Lambda(self.end_of_sequence_token),
            fl.Lambda(self.zero_right_padding),
        )
    
    def compute_sinuosoidal_embedding(self, x: Int[Tensor, "*batch n_colors 3"]) -> Float[Tensor, "*batch n_colors 3 model_dim"]:
        range = arange(start=0, end=x.shape[1], dtype=float32, device=x.device).unsqueeze(1)
        embedding = compute_sinusoidal_embedding(range, embedding_dim=self.model_dim)
        return embedding.squeeze(1).unsqueeze(0).repeat(x.shape[0], 1, 1)

    def end_of_sequence_token(self, x: Float[Tensor, "*batch colors embedding_dim"]) -> Float[Tensor, "*batch colors_with_end embedding_dim"]:
        # Build a tensor of size (batch_size, 1, embedding_dim) with the end of string token
        # end _of string token is a dim_model vector with 1 in the last position
        numpy_end_of_sequence_token = np.zeros((1, self.embedding_dim))
        numpy_end_of_sequence_token[-1] = 1
        
        end_of_sequence_tensor : Float[Tensor, "*batch 1 embedding_dim"] = tensor(
            numpy_end_of_sequence_token, 
            device=x.device, 
            dtype=x.dtype
        ).reshape(1, 1, -1).repeat(x.shape[0], 1, 1)
                
        return torch.cat((x, end_of_sequence_tensor), dim=1)

    def zero_right_padding(self, x: Float[Tensor, "*batch colors_with_end embedding_dim"]) -> Float[Tensor, "*batch max_colors model_dim"]:
        # Zero padding for the right side
        padding_width = (self.max_colors - x.shape[1] % self.max_colors) % self.max_colors
        return pad(x, (0, 0, 0, padding_width))
        
class SD1ColorPaletteAdapter(Generic[T], fl.Chain, Adapter[T]):
    # Prevent PyTorch module registration
    _color_palette_encoder: list[ColorPaletteEncoder]    
    def __init__(
        self,
        target: T,
        model_dim: int,
        scale: float = 1.0,
        max_colors: int = 8,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        with self.setup_adapter(target):
            super().__init__(target)
        
        cross_attn_2d = target.ensure_find(CrossAttentionBlock2d)
        
        self._color_palette_encoder = [ColorPaletteEncoder(
            model_dim=model_dim,
            max_colors=max_colors,
            embedding_dim=cross_attn_2d.context_embedding_dim,
            device=device,
            dtype=dtype,
        )]

        self.sub_adapters = [
            CrossAttentionAdapter(target=cross_attn, scale=scale, image_sequence_length=max_colors)
            for cross_attn in filter(lambda attn: type(attn) != fl.SelfAttention, target.layers(fl.Attention))
        ]

    def inject(self, parent: fl.Chain | None = None) -> "SD1ColorPaletteAdapter":
        for adapter in self.sub_adapters:
            adapter.inject()
        return super().inject(parent)

    def eject(self) -> None:
        for adapter in self.sub_adapters:
            adapter.eject()
        super().eject()

    def set_scale(self, scale: float) -> None:
        for cross_attn in self.sub_adapters:
            cross_attn.scale = scale
    
    @property
    def color_palette_encoder(self) -> ColorPaletteEncoder:
        return self._color_palette_encoder[0]
    
    def encode_colors(self, x: Int[Tensor, "*batch n 3"]) -> Float[Tensor, "*batch max_colors model_dim"]:
        return self.color_palette_encoder[0](x)
