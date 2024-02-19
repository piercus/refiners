from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from refiners.training_utils.config import BaseConfig
    from refiners.training_utils.trainers.trainer import Trainer

T = TypeVar("T", bound="Trainer[BaseConfig, Any]")


class CallbackConfig(BaseModel):
    """
    Base configuration for a callback.

    For your callback to be properly configured, you should inherit from this class and add your own configuration.
    """

    model_config = ConfigDict(extra="forbid")


class Callback(Generic[T]):
    def on_init_begin(self, trainer: T) -> None:
        ...

    def on_init_end(self, trainer: T) -> None:
        ...

    def on_train_begin(self, trainer: T) -> None:
        ...

    def on_train_end(self, trainer: T) -> None:
        ...

    def on_epoch_begin(self, trainer: T) -> None:
        ...

    def on_epoch_end(self, trainer: T) -> None:
        ...

    def on_batch_begin(self, trainer: T) -> None:
        ...

    def on_batch_end(self, trainer: T) -> None:
        ...

    def on_backward_begin(self, trainer: T) -> None:
        ...

    def on_backward_end(self, trainer: T) -> None:
        ...

    def on_optimizer_step_begin(self, trainer: T) -> None:
        ...

    def on_optimizer_step_end(self, trainer: T) -> None:
        ...

    def on_compute_loss_begin(self, trainer: T) -> None:
        ...

    def on_compute_loss_end(self, trainer: T) -> None:
        ...

    def on_evaluate_begin(self, trainer: T) -> None:
        ...

    def on_evaluate_end(self, trainer: T) -> None:
        ...

    def on_lr_scheduler_step_begin(self, trainer: T) -> None:
        ...

    def on_lr_scheduler_step_end(self, trainer: T) -> None:
        ...

class ClockCallback(Callback["Trainer[BaseConfig, Any]"]):
    def on_train_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        trainer.clock.reset()
        logger.info(
            (
                "Starting training for a total of: "
                f"{trainer.clock.num_steps} steps, "
                f"{trainer.clock.num_epochs} epochs, "
                f"{trainer.clock.num_iterations} iterations."
            )
        )
        trainer.clock.start_timer()

    def on_train_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        trainer.clock.stop_timer()
        logger.info(
            (
                "Training took: "
                f"{trainer.clock.time_elapsed} seconds, "
                f"{trainer.clock.iteration} iterations, "
                f"{trainer.clock.epoch} epochs, "
                f"{trainer.clock.step} steps."
            )
        )

    def on_epoch_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        logger.info(f"Epoch {trainer.clock.epoch} started.")

    def on_epoch_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        trainer.clock.epoch += 1
        trainer.clock.num_batches_processed = 0

    def on_batch_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        logger.info(f"Step {trainer.clock.step} started.")
        
    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        trainer.clock.step += 1
        trainer.clock.num_batches_processed += 1
        trainer.clock.num_minibatches_processed += 1

    def on_optimizer_step_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        logger.info(f"Iteration {trainer.clock.iteration} ended.")
        trainer.clock.iteration += 1
        trainer.clock.num_minibatches_processed = 0

    def on_evaluate_begin(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        logger.info("Evaluation started.")

    def on_evaluate_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        logger.info("Evaluation ended.")


class GradientNormClipping(Callback["Trainer[BaseConfig, Any]"]):
    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        clip_norm = trainer.config.training.clip_grad_norm
        if clip_norm is not None:
            clip_gradient_norm(
                parameters=trainer.learnable_parameters, total_norm=trainer.total_gradient_norm, clip_norm=clip_norm
            )


class GradientValueClipping(Callback["Trainer[BaseConfig, Any]"]):
    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        clip_value = trainer.config.training.clip_grad_value
        if clip_value is not None:
            clip_gradient_value(parameters=trainer.learnable_parameters, clip_value=clip_value)

class GradientNormLogging(Callback["Trainer[BaseConfig, Any]"]):
    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        trainer.log(data={"total_grad_norm": trainer.total_gradient_norm})

class GradientNormLayerLogging(Callback["Trainer[BaseConfig, Any]"]):
    def on_backward_end(self, trainer: "Trainer[BaseConfig, Any]") -> None:
        named_gradient_norm = trainer.named_gradient_norm
        for layer_name, norm in named_gradient_norm:
            trainer.log(data={f"layer_grad_norm/{layer_name}": norm})
