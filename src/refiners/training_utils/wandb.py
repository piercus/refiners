from abc import ABC
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import wandb
from PIL import Image
from pydantic import BaseModel

from refiners.training_utils.callback import Callback
from refiners.training_utils.config import BaseConfig
from refiners.training_utils.trainer import Trainer

number = float | int
WandbLoggable = number | Image.Image | list[number] | dict[str, list[number]]


def convert_to_wandb(value: WandbLoggable) -> Any:
    match value:
        case Image.Image():
            return convert_to_wandb_image(value=value)
        case list():
            return convert_to_wandb_histogram(value=value)
        case dict():
            return convert_to_wandb_table(value=value)
        case _:
            return value


def convert_to_wandb_image(value: Image.Image) -> wandb.Image:
    return wandb.Image(data_or_path=value)


def convert_to_wandb_histogram(value: list[number]) -> wandb.Histogram:
    return wandb.Histogram(sequence=value)


def convert_to_wandb_table(value: dict[str, list[number]]) -> wandb.Table:
    assert all(
        isinstance(v, list) and len(v) == len(next(iter(value.values()))) for v in value.values()
    ), "Expected a dictionary of lists of the same size"
    columns = list(value.keys())
    data_rows = list(zip(*value.values()))
    return wandb.Table(columns=columns, data=data_rows)


class WandbLogger:
    def __init__(self, init_config: dict[str, Any] = {}) -> None:
        self.wandb_run = wandb.init(**init_config)  # type: ignore

    def log(self, data: dict[str, WandbLoggable], step: int) -> None:
        converted_data = {key: convert_to_wandb(value=value) for key, value in data.items()}
        self.wandb_run.log(converted_data, step=step)  # type: ignore

    def update_summary(self, key: str, value: Any) -> None:
        self.wandb_run.summary[key] = value  # type: ignore

    @property
    def project_name(self) -> str:
        return self.wandb_run.project_name()  # type: ignore

    @property
    def run_name(self) -> str:
        return self.wandb_run.name or ""  # type: ignore

<<<<<<< HEAD
    @property
    def dir(self) -> str:
        return self.wandb_run.dir # type: ignore
=======

class WandbConfig(BaseModel):
    """
    Wandb configuration.

    See https://docs.wandb.ai/ref/python/init for more details.
    """

    mode: Literal["online", "offline", "disabled"] = "disabled"
    project: str
    entity: str | None = None
    save_code: bool | None = None
    name: str | None = None
    tags: list[str] = []
    group: str | None = None
    job_type: str | None = None
    notes: str | None = None
    dir: Path | None = None
    resume: bool | Literal["allow", "must", "never", "auto"] | None = None
    reinit: bool | None = None
    magic: bool | None = None
    anonymous: Literal["never", "allow", "must"] | None = None
    id: str | None = None


AnyTrainer = Trainer[BaseConfig, Any]


class WandbCallback(Callback["TrainerWithWandb"]):
    epoch_losses: list[float]
    iteration_losses: list[float]

    def on_init_begin(self, trainer: "TrainerWithWandb") -> None:
        trainer.load_wandb()

    def on_train_begin(self, trainer: "TrainerWithWandb") -> None:
        self.epoch_losses = []
        self.iteration_losses = []

    def on_compute_loss_end(self, trainer: "TrainerWithWandb") -> None:
        loss_value = trainer.loss.detach().cpu().item()
        self.epoch_losses.append(loss_value)
        self.iteration_losses.append(loss_value)
        trainer.wandb_log(data={"step_loss": loss_value})

    def on_optimizer_step_end(self, trainer: "TrainerWithWandb") -> None:
        avg_iteration_loss = sum(self.iteration_losses) / len(self.iteration_losses)
        trainer.wandb_log(data={"average_iteration_loss": avg_iteration_loss})
        self.iteration_losses = []

    def on_epoch_end(self, trainer: "TrainerWithWandb") -> None:
        avg_epoch_loss = sum(self.epoch_losses) / len(self.epoch_losses)
        trainer.wandb_log(data={"average_epoch_loss": avg_epoch_loss, "epoch": trainer.clock.epoch})
        self.epoch_losses = []

    def on_lr_scheduler_step_end(self, trainer: "TrainerWithWandb") -> None:
        trainer.wandb_log(data={"learning_rate": trainer.optimizer.param_groups[0]["lr"]})

    def on_backward_end(self, trainer: "TrainerWithWandb") -> None:
        trainer.wandb_log(data={"total_grad_norm": trainer.total_gradient_norm})


class WandbMixin(ABC):
    config: Any
    wandb_logger: WandbLogger

    def load_wandb(self) -> None:
        wandb_config = getattr(self.config, "wandb", None)
        assert wandb_config is not None and isinstance(wandb_config, WandbConfig), "Wandb config is not set"
        init_config = {**wandb_config.model_dump(), "config": self.config.model_dump()}
        self.wandb_logger = WandbLogger(init_config=init_config)

    def wandb_log(self, data: dict[str, WandbLoggable]) -> None:
        assert isinstance(self, Trainer), "WandbMixin must be mixed with a Trainer"
        self.wandb_logger.log(data=data, step=self.clock.step)

    @cached_property
    def wandb_callback(self) -> WandbCallback:
        return WandbCallback()


class TrainerWithWandb(AnyTrainer, WandbMixin, ABC):
    pass
>>>>>>> main
