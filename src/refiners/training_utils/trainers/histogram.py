from functools import cached_property
from typing import Any, List, Tuple, TypedDict

from loguru import logger
from PIL import Image, ImageDraw
from pydantic import BaseModel
from torch import Tensor

import refiners.fluxion.layers as fl
from refiners.fluxion.adapters.histogram import (
    ColorLoss,
    HistogramDistance,
    HistogramExtractor,
    HistogramProjection,
    SD1HistogramAdapter,
    histogram_to_histo_channels,
)
from refiners.fluxion.adapters.histogram_auto_encoder import HistogramAutoEncoder
from refiners.fluxion.adapters.palette import PaletteExtractor
from refiners.fluxion.utils import images_to_tensor, save_to_safetensors, tensor_to_image
from refiners.foundationals.clip.text_encoder import CLIPTextEncoderL
from refiners.training_utils.callback import Callback, GradientNormLayerLogging
from refiners.training_utils.config import TrainingConfig
from refiners.training_utils.datasets.palette import (
    ColorDatasetConfig,
    Palette,
    PaletteDataset,
    SamplingByPalette,
    TextEmbeddingPaletteLatentsBatch,
)
from refiners.training_utils.huggingface_datasets import HuggingfaceDatasetConfig
from refiners.training_utils.metrics.palette import BatchHistogramPrompt, BatchHistogramResults, batch_palette_metrics
from refiners.training_utils.trainers.abstract_color_trainer import (
    AbstractColorTrainer,
    ColorTrainerEvaluationConfig,
    GridEvalDataset,
)
from refiners.training_utils.trainers.histogram_auto_encoder import HistogramAutoEncoderConfig
from refiners.training_utils.trainers.latent_diffusion import (
    FinetuneLatentDiffusionBaseConfig,
    TestDiffusionBaseConfig,
)

Color = Tuple[int, int, int]
Histogram = Tensor


class ImageAndHistogram(TypedDict):
    image: Image.Image
    histogram: Histogram
    palette: Palette

class ColorTrainingConfig(TrainingConfig):
    color_loss_weight: float = 1.0

class AdapterConfig(BaseModel):
    embedding_dim: int = 768
    num_tokens: int = 4
    num_layers: int = 0
    feedforward_dim: int = 768

class HistogramLatentDiffusionConfig(FinetuneLatentDiffusionBaseConfig):
    histogram_auto_encoder: HistogramAutoEncoderConfig
    adapter: AdapterConfig
    dataset: ColorDatasetConfig
    evaluation: ColorTrainerEvaluationConfig
    training: ColorTrainingConfig
    eval_dataset: ColorDatasetConfig

class GridEvalHistogramDataset(GridEvalDataset[BatchHistogramPrompt]):
    __prompt_type__ = BatchHistogramPrompt
    
    def __init__(self, 
            db_indexes: list[int], 
            hf_dataset: PaletteDataset, 
            source_prompts: list[str], 
            text_encoder: CLIPTextEncoderL, 
            histogram_extractor: HistogramExtractor,
            palette_extractor: PaletteExtractor
        ) -> None:
        super().__init__(db_indexes=db_indexes, hf_dataset=hf_dataset, source_prompts=source_prompts, text_encoder=text_encoder)
        self.histogram_extractor = histogram_extractor
        self.palette_extractor = palette_extractor
        
    def process_item(self, items: TextEmbeddingPaletteLatentsBatch) -> dict[str, Any]:
        if len(items) != 1:
            raise ValueError("The items must have length 1.")
        
        histograms = self.histogram_extractor.images_to_histograms([item.image for item in items])
        source_palettes = [self.palette_extractor(item.image, size=len(item.palette)) for item in items]

        return {
            "source_palettes": source_palettes,
            "source_histograms": histograms
        }

class HistogramLatentDiffusionTrainer(
    AbstractColorTrainer[BatchHistogramPrompt, BatchHistogramResults, HistogramLatentDiffusionConfig],
):
    
    
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
    
    @cached_property
    def palette_extractor(self) -> PaletteExtractor:
        return PaletteExtractor(size = 8)
    
    def load_dataset(self) -> PaletteDataset:
        return PaletteDataset(
            config=self.config.dataset,
            # use only palette 8 here
            sampling_by_palette = SamplingByPalette(
                sampling={
                    "palette_8": 1.0
                }
            )
		)
    
    def batch_metrics(self, results: BatchHistogramResults, prefix: str = "histogram-img") -> None:
        
        self.log({f"{prefix}/loss": self.histogram_distance(
            results.result_histograms.to(device=self.device),
            results.source_histograms.to(device=self.device)
        )})
        
        
        [red, green, blue] = self.color_loss.image_vs_histo(
                results.result_images.to(device=self.device),
                results.source_histograms.to(device=self.device),
                self.config.histogram_auto_encoder.color_bits
        )
        
        self.log({f"{prefix}/rgb_distance": (red+green+blue).item()})
        self.log({f"{prefix}/red_distance": red.item()})
        self.log({f"{prefix}/green_distance": green.item()})
        self.log({f"{prefix}/blue_distance": blue.item()})

        batch_palette_metrics(self.log, results, prefix)
    
    @cached_property
    def histogram_distance(self) -> HistogramDistance:
        return HistogramDistance(color_bits=self.config.histogram_auto_encoder.color_bits)
    
    @cached_property
    def histogram_auto_encoder(self) -> HistogramAutoEncoder:
        assert self.config.models["histogram_auto_encoder"] is not None, "The config must contain a histogram entry."
        
        autoencoder = HistogramAutoEncoder(
            latent_dim=self.config.histogram_auto_encoder.latent_dim, 
            resnet_sizes=self.config.histogram_auto_encoder.resnet_sizes,
            color_bits=self.config.histogram_auto_encoder.color_bits,
            n_down_samples=self.config.histogram_auto_encoder.n_down_samples,
            device=self.device
        )
        logger.info(f"Building autoencoder with compression rate {autoencoder.compression_rate}")
        return autoencoder

    @cached_property
    def histogram_adapter(self) -> SD1HistogramAdapter[Any]:
        embedding_dim = self.config.adapter.embedding_dim
            
        adapter = SD1HistogramAdapter(target=self.unet, embedding_dim=embedding_dim)
        return adapter
    
    @cached_property
    def histogram_projection(self) -> HistogramProjection:   
        embedding_dim = self.config.adapter.embedding_dim         

        return HistogramProjection(
            embedding_dim=embedding_dim,
            feedforward_dim=self.config.adapter.feedforward_dim,
            num_layers=self.config.adapter.num_layers,
            num_tokens=self.config.adapter.num_tokens,
            device=self.device,
            dtype=self.dtype
        )

    @cached_property
    def color_loss(self) -> ColorLoss:
        return ColorLoss()

    def collate_results(self, batch: list[BatchHistogramResults]) -> BatchHistogramResults:
        return BatchHistogramResults.collate_fn(batch)
    
    def empty(self) -> BatchHistogramResults:
        return BatchHistogramResults.empty()
    
    def collate_prompts(self, batch: list[BatchHistogramPrompt]) -> BatchHistogramPrompt:
        return BatchHistogramPrompt.collate_fn(batch)
        
    def __init__(
        self,
        config: HistogramLatentDiffusionConfig,
        callbacks: "list[Callback[Any]] | None" = None,
    ) -> None:
        super().__init__(config=config, callbacks=callbacks)
        self.callbacks.extend((LoadHistogram(), SaveHistogram(), GradientNormLayerLogging()))

    def load_models(self) -> dict[str, fl.Module]:
        return {
            "unet": self.unet,
            "text_encoder": self.text_encoder,
            "lda": self.lda,
            "histogram_auto_encoder": self.histogram_auto_encoder,
            "histogram_projection": self.histogram_projection
        }
    
    def build_results(self, batch: BatchHistogramPrompt, result_images: Tensor) -> BatchHistogramResults:
        return BatchHistogramResults(
            source_histograms = batch.source_histograms,
            source_prompts = batch.source_prompts,
            source_palettes = batch.source_palettes,
            text_embeddings = batch.text_embeddings,
            result_images = result_images,
            result_histograms = self.histogram_extractor(result_images),
            db_indexes= batch.db_indexes,
            source_images= batch.source_images
        )
    def compute_loss(self, batch: TextEmbeddingPaletteLatentsBatch) -> Tensor:
        
        texts = [item.text for item in batch]
        text_embeddings = self.text_encoder(texts)

        images = [item.image for item in batch]
        latents = self.lda.images_to_latents(images)
        images_tensor = images_to_tensor(images, device=self.device, dtype=self.dtype)
        
        histograms = self.histogram_extractor.images_to_histograms([item.image for item in batch], device = self.device, dtype = self.dtype)
        histogram_embeddings = self.histogram_auto_encoder.encode(histograms)
        histogram_embeddings = histogram_embeddings.reshape(histogram_embeddings.shape[0], 1, -1)
        histogram_embeddings2 = self.histogram_projection(histogram_embeddings)

        timestep = self.sample_timestep()
        noise = self.sample_noise(size=latents.shape, dtype=latents.dtype)
        noisy_latents = self.ddpm_solver.add_noise(x=latents, noise=noise, step=self.current_step)
        self.unet.set_timestep(timestep=timestep)

        self.unet.set_clip_text_embedding(clip_text_embedding=text_embeddings)
        self.histogram_adapter.set_histogram_embedding(histogram_embeddings2)

        prediction = self.unet(noisy_latents)

        loss_1 = self.mse_loss(prediction, noise)
        
        predicted_latents = self.ddpm_solver.remove_noise(
            x=noisy_latents.to(device=self.ddpm_solver.device), 
            noise=prediction.to(device=self.ddpm_solver.device), 
            step=self.current_step
        )

        predicted_decoded = self.lda.decode(x=predicted_latents).to(device=self.device)
        predicted_images_tensor = (predicted_decoded + 1) / 2
        loss_2 = self.color_loss(
            predicted_images_tensor,
            images_tensor
        )
        
        self.log({
            f"losses/color_loss": loss_2.item(),
            f"losses/image": loss_1.item()
        })

        return loss_1 + loss_2 * self.config.training.color_loss_weight
 
    def eval_set_adapter_values(self, batch: BatchHistogramPrompt) -> None:
        cfg_histogram_embedding = self.histogram_auto_encoder.compute_histogram_embedding(
            batch.source_histograms
        )
        cfg_histogram_embedding2 = self.histogram_projection(cfg_histogram_embedding)
        self.histogram_adapter.set_histogram_embedding(cfg_histogram_embedding2)

    def draw_curves(self, res_histo: list[float], src_histo: list[float], color: str, width: int, height: int) -> Image.Image:
        histo_img = Image.new(mode="RGB", size=(width, height))
        
        draw = ImageDraw.Draw(histo_img)
        
        if len(res_histo) != len(src_histo):
            raise ValueError("The histograms must have the same length.")
        
        ratio = width/len(res_histo)
        semi_height = height//2
        
        scale_ratio = 5
                
        draw.line([
            (i*ratio, (1-res_histo[i]*scale_ratio)*semi_height + semi_height) for i in range(len(res_histo))
        ], fill=color, width=4)
        
        draw.line([
            (i*ratio, (1-src_histo[i]*scale_ratio)*semi_height) for i in range(len(src_histo))
        ], fill=color, width=1)
        
        return histo_img
    
    def draw_cover_image(self, batch: BatchHistogramResults) -> Image.Image:
        (batch_size, channels, height, width) = batch.result_images.shape

        vertical_image = batch.result_images.permute(0,2,3,1).reshape(1, height*batch_size, width, channels).permute(0,3,1,2)
        
        results_histograms = batch.result_histograms
        source_histograms = batch.source_histograms
        source_images = batch.source_images

        join_canvas_image: Image.Image = Image.new(
            mode="RGB", size=(width + width//2, height * batch_size)
        )
        res_image = tensor_to_image(vertical_image)
        
        join_canvas_image.paste(res_image, box=(width//2, 0))
        
        res_histo_channels = histogram_to_histo_channels(results_histograms)
        src_histo_channels = histogram_to_histo_channels(source_histograms)
        
        colors = ["red", "green", "blue"]
        
        for i in range(batch_size):
            join_canvas_image.paste(source_images[i].resize((width//2, height//2)), box=(0, height *i))
            
            source_image_palette = self.draw_palette(
                self.palette_extractor.from_histogram(source_histograms[i], color_bits= self.config.histogram_auto_encoder.color_bits, size=len(batch.source_palettes[i])),
                width//2,
                height//16
            )
            join_canvas_image.paste(source_image_palette, box=(0, height *i + height//2))
            
            res_image_palette = self.draw_palette(
                self.palette_extractor.from_histogram(results_histograms[i], color_bits= self.config.histogram_auto_encoder.color_bits, size=len(batch.source_palettes[i])),
                width//2,
                height//16
            )
            
            join_canvas_image.paste(res_image_palette, box=(0, height *i + (15*height)//16))

            for (color_id, color_name) in enumerate(colors):
                image_curve = self.draw_curves(
                    res_histo_channels[color_id][i].cpu().tolist(), # type: ignore
                    src_histo_channels[color_id][i].cpu().tolist(), # type: ignore
                    color_name,
                    width//2,
                    height//8
                )
                join_canvas_image.paste(image_curve, box=(0, height *i + height//2 + ((1+2*color_id)*height)//16))
                
        return join_canvas_image
    



    # def compute_evaluation(self) -> None:
    #     prompts = self.config.evaluation.prompts
    #     num_images_per_prompt = self.config.evaluation.num_images_per_prompt
    #     images_and_histograms: List[ImageAndHistogram] = []
        
    #     if len(prompts) > 0:
    #         images_and_histograms = self.compute_db_samples_evaluation(
    #             prompts, 
    #             num_images_per_prompt
    #         )
    #         self.batch_image_histogram_metrics(images_and_histograms, prefix="histogram-image-edge")


class LoadHistogram(Callback[HistogramLatentDiffusionTrainer]):
    def on_train_begin(self, trainer: HistogramLatentDiffusionTrainer) -> None:
        adapter = trainer.histogram_adapter
        adapter.zero_init()
        adapter.inject()


class SaveHistogram(Callback[HistogramLatentDiffusionTrainer]):
    def on_checkpoint_save(self, trainer: HistogramLatentDiffusionTrainer) -> None:
        tensors: dict[str, Tensor] = {}
        # metadata: dict[str, str] = {}

        model = trainer.unet
        if model.parent is None:
            raise ValueError("The model must have a parent.")
        adapter = model.parent

        tensors = {f"unet.{i}": p for i, p in adapter.named_parameters() if p.requires_grad}
        
        projection = {
            f"histogram_projection.{i}": w for i, w in trainer.histogram_projection.state_dict().items()
        }
        
        tensors.update(projection)

        save_to_safetensors(
            path=trainer.ensure_checkpoints_save_folder / f"step{trainer.clock.step}.safetensors", tensors=tensors
        )
