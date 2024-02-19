from functools import cached_property
from typing import Any, TypedDict

import numpy as np
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torch import Tensor

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.palette import Color, PaletteEncoder, PaletteExtractor, SD1PaletteAdapter
from refiners.fluxion.utils import load_from_safetensors, save_to_safetensors, tensor_to_images
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.foundationals.latent_diffusion import SD1UNet
from refiners.training_utils.callback import Callback, GradientNormLayerLogging
from refiners.training_utils.datasets.palette import (
    ColorDatasetConfig,
    Palette,
    PaletteDataset,
    TextEmbeddingPaletteLatentsBatch,
)
from refiners.training_utils.metrics.palette import (
    BatchHistogramPrompt,
    BatchHistogramResults,
    ImageAndPalette,
    batch_image_palette_metrics,
)
from refiners.training_utils.trainers.abstract_color_trainer import (
    AbstractColorTrainer,
    ColorTrainerEvaluationConfig,
    GridEvalDataset,
)
from refiners.training_utils.trainers.histogram import GridEvalHistogramDataset
from refiners.training_utils.trainers.latent_diffusion import (
    FinetuneLatentDiffusionBaseConfig,
)
from refiners.training_utils.wandb import WandbLoggable


class PaletteConfig(BaseModel):
    feedforward_dim: int = 3072
    num_attention_heads: int = 12
    num_layers: int = 12
    embedding_dim: int = 768
    trigger_phrase: str = ""
    use_only_trigger_probability: float = 0.0
    max_colors: int
    mode : str = "transformer"
    weighted_palette: bool = False
    without_caption_probability: float = 0.17

class PalettePromptConfig(BaseModel):
    text: str
    palette: Palette

class LatentPrompt(TypedDict):
    text: str
    palette_embedding: Tensor

class PaletteLatentDiffusionConfig(FinetuneLatentDiffusionBaseConfig):
    palette: PaletteConfig
    evaluation: ColorTrainerEvaluationConfig
    dataset: ColorDatasetConfig
    eval_dataset: ColorDatasetConfig

class GridEvalPaletteDataset(GridEvalDataset[BatchHistogramPrompt]):
    __prompt_type__ = BatchHistogramPrompt
    def __init__(self, db_indexes: list[int], hf_dataset: PaletteDataset, source_prompts: list[str], text_encoder: CLIPTextEncoderL, palette_extractor: PaletteExtractor):
        super().__init__(db_indexes, hf_dataset, source_prompts, text_encoder)
        self.palette_extractor = palette_extractor
    def process_item(self, items: TextEmbeddingPaletteLatentsBatch) -> dict[str, Any]:
        if len(items) != 1:
            raise ValueError("The items must have length 1.")

        source_palettes = [self.palette_extractor(item.image, size=len(item.palette)) for item in items]
        return {
            "source_palettes": source_palettes
        }

class PaletteLatentDiffusionTrainer(AbstractColorTrainer[BatchHistogramPrompt, BatchHistogramResults, PaletteLatentDiffusionConfig]):
    @cached_property
    def palette_encoder(self) -> PaletteEncoder:
        assert (
            self.config.models["palette_encoder"] is not None
        ), "The config must contain a palette_encoder entry."

        encoder = PaletteEncoder(
            max_colors=self.config.palette.max_colors,
            embedding_dim=self.config.palette.embedding_dim,
            num_layers=self.config.palette.num_layers,
            mode=self.config.palette.mode,
            weighted_palette=self.config.palette.weighted_palette,
            num_attention_heads=self.config.palette.num_attention_heads,
            feedforward_dim=self.config.palette.feedforward_dim,
            device=self.device,
        )
        return encoder

    @cached_property
    def palette_adapter(self) -> SD1PaletteAdapter[Any]:
        
        weights : dict[str, Tensor] | None = None
        scale = 1.0
        
        if "palette_adapter" in self.config.adapters:
            if checkpoint := self.config.adapters["palette_adapter"].checkpoint:
                weights = load_from_safetensors(checkpoint)
            scale = self.config.adapters["palette_adapter"].scale
        
        adapter : SD1PaletteAdapter[SD1UNet] = SD1PaletteAdapter(
            target=self.unet,
            weights=weights,
            scale=scale,
            palette_encoder=self.palette_encoder
        )
        
        if weights is None:
            adapter.zero_init()
        
        return adapter

    def __init__(
        self,
        config: PaletteLatentDiffusionConfig,
        callbacks: "list[Callback[Any]] | None" = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks)
        self.callbacks.extend((LoadPalette(), SavePalette(), GradientNormLayerLogging()))
    
    def load_models(self) -> dict[str, fl.Module]:
        return {
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "lda": self.lda,
            "palette_encoder": self.palette_encoder,
        }
    
    @cached_property
    def grid_eval_dataset(self) -> GridEvalDataset[BatchHistogramPrompt]:
        return GridEvalHistogramDataset(
            db_indexes=self.config.evaluation.db_indexes,
            hf_dataset=self.eval_dataset,
            source_prompts=self.config.evaluation.prompts,
            text_encoder=self.text_encoder,
            histogram_extractor=self.histogram_extractor,
            palette_extractor=self.palette_extractor
        )
        
    # def eval_dataset(self) -> list[BatchHistogramPrompt]:
    #     dataset = self.dataset
    #     indices = self.config.evaluation.db_indexes
    #     items = [dataset[i][0] for i in indices]
    #     print(f"color palette lenght : {[len(item.palette) for item in items]}")
    #     palette = [self.palette_extractor(item.image, size=len(item.palette)) for item in items]
    #     images = [item.image for item in items]
    #     eval_indices = list(zip(indices, palette, images))
        
    #     evaluations : list[BatchHistogramPrompt] = []
    #     prompts_list = [(prompt, self.text_encoder(prompt)) for prompt in self.config.evaluation.prompts]

    #     for (prompt, prompt_embedding) in prompts_list:
    #         for db_index, palette, image in eval_indices:
    #             batch_prompt = BatchHistogramPrompt(
    #                 source_prompts= [prompt],
    #                 db_indexes= [db_index],
    #                 source_palettes= [palette],
    #                 text_embeddings= prompt_embedding,
    #                 source_images= [image]
    #             )
    #             evaluations.append(batch_prompt)
        
    #     print(f"Eval dataset size: {len(evaluations)}")
    #     return evaluations
    
    def build_results(self, batch: BatchHistogramPrompt, result_images: Tensor) -> BatchHistogramResults:
        
        return BatchHistogramResults(
            source_prompts=batch.source_prompts,
            db_indexes=batch.db_indexes,
            source_histograms=batch.source_histograms,
            source_palettes=batch.source_palettes,
            result_histograms = self.histogram_extractor(result_images),
            result_images=result_images,
            source_images=batch.source_images,
            result_palettes=[self.palette_extractor(image, size=len(batch.source_palettes[i])) for i, image in enumerate(tensor_to_images(result_images))],
            text_embeddings=batch.text_embeddings
        )
    
    def collate_results(self, batch: list[BatchHistogramResults]) -> BatchHistogramResults:
        return BatchHistogramResults.collate_fn(batch)
    
    def empty(self) -> BatchHistogramResults:
        return BatchHistogramResults.empty()
    
    def collate_prompts(self, batch: list[BatchHistogramPrompt]) -> BatchHistogramPrompt:
        return BatchHistogramPrompt.collate_fn(batch)
    
    def compute_loss(self, batch: TextEmbeddingPaletteLatentsBatch) -> Tensor:
        
        texts = [item.text for item in batch]
        text_embeddings = self.text_encoder(texts)
        
        latents = self.lda.images_to_latents([item.image for item in batch])
        palettes = [self.palette_extractor(item.image, size=len(item.palette)) for item in batch]
        
        palette_embeddings = self.palette_encoder(
            palettes
        )

        timestep = self.sample_timestep()
        noise = self.sample_noise(size=latents.shape, dtype=latents.dtype)
        noisy_latents = self.ddpm_solver.add_noise(x=latents, noise=noise, step=self.current_step)
        self.unet.set_timestep(timestep=timestep)

        self.unet.set_clip_text_embedding(clip_text_embedding=text_embeddings)
        self.palette_adapter.set_palette_embedding(palette_embeddings)

        prediction = self.unet(noisy_latents)
        loss = self.mse_loss(prediction, noise)
        
        return loss
    
    def eval_set_adapter_values(self, batch: BatchHistogramPrompt) -> None:
        self.palette_adapter.set_palette_embedding(
            self.palette_encoder.compute_palette_embedding(
                batch.source_palettes
            )
        )
    
    def draw_cover_image(self, batch: BatchHistogramResults) -> Image.Image:
        (batch_size, _, height, width) = batch.result_images.shape
        
        palette_img_size = width // self.config.palette.max_colors
        source_images = batch.source_images

        join_canvas_image: Image.Image = Image.new(
            mode="RGB", size=(2*width, (height+palette_img_size) * batch_size)
        )
        images = tensor_to_images(batch.result_images)
        for i, image in enumerate(images):
            join_canvas_image.paste(source_images[i], box=(0, i*(height+palette_img_size)))
            join_canvas_image.paste(image, box=(width, i*(height+palette_img_size)))
            palette_out = batch.result_palettes[i]
            palette_out_img = self.draw_palette(palette_out, width, palette_img_size)
            palette_in_img = self.draw_palette(batch.source_palettes[i], width, palette_img_size)
            
            join_canvas_image.paste(palette_in_img, box=(0, i*(height+palette_img_size) + height))
            join_canvas_image.paste(palette_out_img, box=(width, i*(height+palette_img_size) + height))
        return join_canvas_image
    
    # def palette_distance(self, source: list[Palette], result: list[Palette]) -> float:
    #     if len(source) != len(result):
    #         raise ValueError("The source and result palettes must have the same length.")
        
    #     distance = 0.0
    #     for i in range(len(source)):
    #         distance += self.palette_extractor.distance(source[i], result[i])
            
    #     return distance
    
    def batch_metrics(self, results: BatchHistogramResults, prefix: str = "palette-img") -> None:
        palettes : list[list[Color]] = []
        for p in results.source_palettes:
            palettes.append([cluster[0] for cluster in p])
        
        images = tensor_to_images(results.result_images)
        
        batch_image_palette_metrics(
            self.log, 
            [
                ImageAndPalette({"image": image, "palette": palette})
                for image, palette in zip(images, palettes)
            ], 
            prefix
        )
        
        histo_metrics = self.histogram_distance.metrics(results.result_histograms, results.source_histograms.to(results.result_histograms.device))
        
        log_dict : dict[str, WandbLoggable] = {}
        for (key, value) in histo_metrics.items():
            log_dict[f"eval_histo/{key}"] = value
        
        self.log(log_dict)   

class LoadPalette(Callback[PaletteLatentDiffusionTrainer]):
    def on_train_begin(self, trainer: PaletteLatentDiffusionTrainer) -> None:
        adapter = trainer.palette_adapter
        adapter.inject()

class SavePalette(Callback[PaletteLatentDiffusionTrainer]):
    def on_checkpoint_save(self, trainer: PaletteLatentDiffusionTrainer) -> None:
        tensors: dict[str, Tensor] = {}

        model = trainer.unet
        if model.parent is None:
            raise ValueError("The model must have a parent.")
        adapter = model.parent

        tensors = {f"palette_adapter.{i:03d}.{j:03d}": w for i, ws in enumerate(adapter.weights) for j, w in enumerate(ws)}
        encoder = trainer.palette_encoder

        state_dict = encoder.state_dict()
        for i in state_dict:
            tensors.update({f"palette_encoder.{i}": state_dict[i]})
        
        path = f"{trainer.ensure_checkpoints_save_folder}/step{trainer.clock.step}.safetensors"
        logger.info(
            f"Saving {len(tensors)} tensors to {path}"
        )
        save_to_safetensors(
            path=path, tensors=tensors
        )