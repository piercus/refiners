import random
import time
from abc import ABC, abstractmethod, abstractproperty
from functools import cached_property, wraps
from typing import Any, Callable, Generic, Iterable, TypeVar, cast

import numpy as np
import torch
from loguru import logger
from torch import Tensor, cuda, device as Device, dtype as DType, get_rng_state, set_rng_state, stack
<<<<<<< HEAD:src/refiners/training_utils/trainers/trainer.py
=======
from torch.autograd import backward
>>>>>>> main:src/refiners/training_utils/trainer.py
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LRScheduler,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader, Dataset

from refiners.fluxion import layers as fl
from refiners.fluxion.utils import manual_seed, no_grad
from refiners.training_utils.callback import (
    Callback,
    ClockCallback,
    GradientNormClipping,
    GradientValueClipping,
<<<<<<< HEAD:src/refiners/training_utils/trainers/trainer.py
    MonitorLoss,
    MonitorTime
)
from refiners.training_utils.config import BaseConfig, SchedulerType, TimeUnit, TimeValue
from refiners.training_utils.dropout import DropoutCallback
from refiners.training_utils.sharding_manager import ShardingManager, SimpleShardingManager
from refiners.training_utils.wandb import WandbLoggable, WandbLogger
=======
)
from refiners.training_utils.config import BaseConfig, SchedulerType, TimeUnit, TimeValue
from refiners.training_utils.dropout import DropoutCallback
>>>>>>> main:src/refiners/training_utils/trainer.py

__all__ = ["seed_everything", "scoped_seed", "Trainer"]


def count_learnable_parameters(parameters: Iterable[Parameter]) -> int:
    return sum(p.numel() for p in parameters if p.requires_grad)


def human_readable_number(number: int) -> str:
    float_number = float(number)
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(float_number) < 1000:
            return f"{float_number:.1f}{unit}"
        float_number /= 1000
    return f"{float_number:.1f}E"


def seed_everything(seed: int | None = None) -> None:
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        logger.info(f"Using random seed: {seed}")
    random.seed(a=seed)
    np.random.seed(seed=seed)
    manual_seed(seed=seed)
    cuda.manual_seed_all(seed=seed)


def scoped_seed(seed: int | Callable[..., int] | None = None) -> Callable[..., Callable[..., Any]]:
    """
    Decorator for setting a random seed within the scope of a function.

    This decorator sets the random seed for Python's built-in `random` module,
    `numpy`, and `torch` and `torch.cuda` at the beginning of the decorated function. After the
    function is executed, it restores the state of the random number generators
    to what it was before the function was called. This is useful for ensuring
    reproducibility for specific parts of the code without affecting randomness
    elsewhere.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def inner_wrapper(*args: Any, **kwargs: Any) -> Any:
            random_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = get_rng_state()
            cuda_torch_state = cuda.get_rng_state()
            actual_seed = seed(*args) if callable(seed) else seed
            seed_everything(seed=actual_seed)
            result = func(*args, **kwargs)
            random.setstate(random_state)
            np.random.set_state(numpy_state)
            set_rng_state(torch_state)
            cuda.set_rng_state(cuda_torch_state)
            return result

        return inner_wrapper

    return decorator


class WarmupScheduler(LRScheduler):
    _step_count: int  # defined by LRScheduler

    def __init__(self, optimizer: Optimizer, scheduler: LRScheduler, warmup_scheduler_steps: int = 0) -> None:
        self.warmup_scheduler_steps = warmup_scheduler_steps
        self.scheduler = scheduler
        super().__init__(optimizer=optimizer)

    def get_lr(self) -> list[float] | float:  # type: ignore
        if self._step_count <= self.warmup_scheduler_steps:
            return [base_lr * self._step_count / self.warmup_scheduler_steps for base_lr in self.base_lrs]
        return self.scheduler.get_lr()

    def step(self, epoch: int | None = None) -> None:
        if self._step_count < self.warmup_scheduler_steps:
            super().step()
        else:
            self.scheduler.step(epoch=epoch)
            self._step_count += 1


class TrainingClock:
    def __init__(
        self,
        dataset_length: int,
        batch_size: int,
        training_duration: TimeValue,
        gradient_accumulation: TimeValue,
        evaluation_interval: TimeValue,
        lr_scheduler_interval: TimeValue,
    ) -> None:
        self.dataset_length = dataset_length
        self.batch_size = batch_size
        self.training_duration = training_duration
        self.gradient_accumulation = gradient_accumulation
        self.evaluation_interval = evaluation_interval
        self.lr_scheduler_interval = lr_scheduler_interval
        self.num_batches_per_epoch = dataset_length // batch_size
        self.start_time = None
        self.end_time = None
        self.step = 0
        self.epoch = 0
        self.iteration = 0
        self.num_batches_processed = 0
        self.num_minibatches_processed = 0
        self.loss: Tensor | None = None

    @cached_property
    def unit_to_steps(self) -> dict[TimeUnit, int]:
        iteration_factor = self.num_batches_per_epoch if self.gradient_accumulation["unit"] == TimeUnit.EPOCH else 1
        return {
            TimeUnit.STEP: 1,
            TimeUnit.EPOCH: self.num_batches_per_epoch,
            TimeUnit.ITERATION: self.gradient_accumulation["number"] * iteration_factor,
        }

    def convert_time_unit_to_steps(self, number: int, unit: TimeUnit) -> int:
        return number * self.unit_to_steps[unit]

    def convert_steps_to_time_unit(self, steps: int, unit: TimeUnit) -> int:
        return steps // self.unit_to_steps[unit]

    def convert_time_value(self, time_value: TimeValue, target_unit: TimeUnit) -> int:
        number, unit = time_value["number"], time_value["unit"]
        steps = self.convert_time_unit_to_steps(number=number, unit=unit)
        return self.convert_steps_to_time_unit(steps=steps, unit=target_unit)

    @cached_property
    def num_epochs(self) -> int:
        return self.convert_time_value(time_value=self.training_duration, target_unit=TimeUnit.EPOCH)

    @cached_property
    def num_iterations(self) -> int:
        return self.convert_time_value(time_value=self.training_duration, target_unit=TimeUnit.ITERATION)

    @cached_property
    def num_steps(self) -> int:
        return self.convert_time_value(time_value=self.training_duration, target_unit=TimeUnit.STEP)

    @cached_property
    def num_step_per_iteration(self) -> int:
        return self.convert_time_unit_to_steps(
            number=self.gradient_accumulation["number"], unit=self.gradient_accumulation["unit"]
        )

    @cached_property
    def num_step_per_evaluation(self) -> int:
        return self.convert_time_unit_to_steps(
            number=self.evaluation_interval["number"], unit=self.evaluation_interval["unit"]
        )

    def reset(self) -> None:
        self.start_time = None
        self.end_time = None
        self.step = 0
        self.epoch = 0
        self.iteration = 0
        self.num_batches_processed = 0
        self.num_minibatches_processed = 0

    def start_timer(self) -> None:
        self.start_time = time.time()

    def stop_timer(self) -> None:
        self.end_time = time.time()

    @property
    def time_elapsed(self) -> int:
        assert self.start_time is not None, "Timer has not been started yet."
        return int(time.time() - self.start_time)

    @cached_property
    def evaluation_interval_steps(self) -> int:
        return self.convert_time_unit_to_steps(
            number=self.evaluation_interval["number"], unit=self.evaluation_interval["unit"]
        )

    @cached_property
    def lr_scheduler_interval_steps(self) -> int:
        return self.convert_time_unit_to_steps(
            number=self.lr_scheduler_interval["number"], unit=self.lr_scheduler_interval["unit"]
        )

    @property
    def is_optimizer_step(self) -> bool:
        return self.num_minibatches_processed == self.num_step_per_iteration

    @property
    def is_lr_scheduler_step(self) -> bool:
        return self.step % self.lr_scheduler_interval_steps == 0

    @property
    def done(self) -> bool:
        return self.step >= self.num_steps

    @property
    def is_evaluation_step(self) -> bool:
        return self.step % self.evaluation_interval_steps == 0


def compute_grad_norm(parameters: Iterable[Parameter], device: Device) -> float:
    """
    Computes the gradient norm of the parameters of a given model similar to `clip_grad_norm_` returned value.
    """
    gradients: list[Tensor] = [p.grad.detach() for p in parameters if p.grad is not None]
    assert gradients, "The model has no gradients to compute the norm."
    total_norm = stack(tensors=[gradient.norm().to(device=device) for gradient in gradients]).norm().item()  # type: ignore
    return total_norm  # type: ignore


Batch = TypeVar("Batch")
ConfigType = TypeVar("ConfigType", bound=BaseConfig)


class _Dataset(Dataset[Batch]):
    """
    A wrapper around the `get_item` method to create a [`torch.utils.data.Dataset`][torch.utils.data.Dataset].
    """

    def __init__(self, get_item: Callable[[int], Batch], length: int) -> None:
        assert length > 0, "Dataset length must be greater than 0."
        self.length = length
        self.get_item = get_item

    def __getitem__(self, index: int) -> Batch:
        return self.get_item(index)

    def __len__(self) -> int:
        return self.length


class Trainer(Generic[ConfigType, Batch], ABC):
    def __init__(self, config: ConfigType, callbacks: list[Callback[Any]] | None = None) -> None:
        self.config = config
        self.clock = TrainingClock(
            dataset_length=self.dataset_length,
            batch_size=config.training.batch_size,
            training_duration=config.training.duration,
            evaluation_interval=config.training.evaluation_interval,
            gradient_accumulation=config.training.gradient_accumulation,
            lr_scheduler_interval=config.scheduler.update_interval,
        )
        self.callbacks = callbacks or []
        self.callbacks += self.default_callbacks()
        self._call_callbacks(event_name="on_init_begin")
        self.load_models()
        self.prepare_models()
        self._call_callbacks(event_name="on_init_end")

    def default_callbacks(self) -> list[Callback[Any]]:
        callbacks: list[Callback[Any]] = [
            ClockCallback(),
            GradientValueClipping(),
            GradientNormClipping(),
            DropoutCallback(),
            MonitorTime()
        ]

        # look for any Callback that might be a property of the Trainer
        for attr_name in dir(self):
            if "__" in attr_name:
                continue

            try:
                attr = getattr(self, attr_name)
            except AssertionError:
                continue
            if isinstance(attr, Callback):
                callbacks.append(cast(Callback[Any], attr))
        return callbacks

    @cached_property
    def device(self) -> Device:
<<<<<<< HEAD:src/refiners/training_utils/trainers/trainer.py
        return self.sharding_manager.device

    @cached_property
    def dtype(self) -> DType:
        return self.sharding_manager.dtype
=======
        selected_device = Device(self.config.training.device)
        logger.info(f"Using device: {selected_device}")
        return selected_device
>>>>>>> main:src/refiners/training_utils/trainer.py

    @cached_property
    def dtype(self) -> DType:
        dtype = getattr(torch, self.config.training.dtype, None)
        assert isinstance(dtype, DType), f"Unknown dtype: {self.config.training.dtype}"
        logger.info(f"Using dtype: {dtype}")
        return dtype

    @property
    def parameters(self) -> list[Parameter]:
        """Returns a list of all parameters in all models"""
        return [param for model in self.models.values() for param in model.parameters()]

    @property
    def named_parameters(self) -> list[tuple[str, Parameter]]:
        """Returns a list of all parameters in all models"""
        return [
            (f"{model_name}.{param[0]}", param[1])
            for model_name in self.models
            for param in self.models[model_name].named_parameters()
        ]

    @property
    def named_learnable_parameters(self) -> list[tuple[str, Parameter]]:
        return [named_param for named_param in self.named_parameters if named_param[1].requires_grad]

    @property
    def model_learnable_parameters(self) -> list[tuple[str, list[Parameter]]]:
        """Returns a list of learnable parameters with the model name"""
        results: list[tuple[str, list[Parameter]]] = []
        for model_name in self.models:
            params: list[Parameter] = [param for param in self.models[model_name].parameters() if param.requires_grad]
            if len(params) > 0:
                results.append((model_name, params))
        return results

    @property
    def learnable_parameters(self) -> list[Parameter]:
        """Returns a list of learnable parameters in all models"""
        return [param for model in self.models.values() for param in model.parameters() if param.requires_grad]

    @property
    def optimizer_parameters(self) -> list[dict[str, Any]]:
        """
        Returns a optimizer compatible list of param dict
        See https://pytorch.org/docs/stable/optim.html#per-parameter-options for more details
        """

        params: list[dict[str, Any]] = []
        for model_name, model in self.models.items():
            model_params = [param for param in model.parameters() if param.requires_grad]
            model_config = self.config.models[model_name]
            model_optim_conf: dict[str, Any] = {}

            if model_config.learning_rate is not None:
                model_optim_conf["lr"] = model_config.learning_rate
            if model_config.weight_decay is not None:
                model_optim_conf["weight_decay"] = model_config.learning_rate
            if model_config.betas is not None:
                model_optim_conf["betas"] = model_config.learning_rate
            if model_config.eps is not None:
                model_optim_conf["eps"] = model_config.learning_rate

            for param in model_params:
                optim_params_dict = {"params": param, **model_optim_conf}
                params.append(optim_params_dict)

        return params

    @property
    def learnable_parameter_count(self) -> int:
        """Returns the number of learnable parameters in all models"""
        return count_learnable_parameters(parameters=self.learnable_parameters)

    @property
    def gradients(self) -> list[Tensor]:
        """Returns a list of detached gradients for all learnable parameters in all models"""
        return [
            param.grad.detach()
            for model in self.models.values()
            for param in model.parameters()
            if param.grad is not None
        ]

    @property
    def total_gradient_norm(self) -> float:
        """Returns the total gradient norm for all learnable parameters in all models"""
        return compute_grad_norm(parameters=self.parameters, device=self.device)

    @property
    def named_gradient_norm(self) -> list[tuple[str, float]]:
        """Returns the total gradient norm for all learnable parameters in all models"""
        list: list[tuple[str, float]] = []
        for param in self.named_learnable_parameters:  
            if param[1].grad is not None:
                list.append((param[0], compute_grad_norm(parameters=[param[1]], device=self.device)))
            # else:
            #     print(f"Gradient for {param[0]} is None")
        return list

    @cached_property
    def optimizer(self) -> Optimizer:
        formatted_param_count = human_readable_number(number=self.learnable_parameter_count)
        logger.info(f"Total number of learnable parameters in the model(s): {formatted_param_count}")
        model_learnable_parameters = self.model_learnable_parameters
        model_formatted_param_count = [
            (model_name, human_readable_number(count_learnable_parameters(parameters=param_list)))
            for model_name, param_list in model_learnable_parameters
        ]
        for model_name, param_count in model_formatted_param_count:
            logger.info(f"Number of learnable parameters in model `{model_name}`: {param_count}")
        optimizer = self.config.optimizer.get(params=self.optimizer_parameters)
        return optimizer

    @cached_property
    def lr_scheduler(self) -> LRScheduler:
        config = self.config.scheduler
        scheduler_step_size = config.update_interval["number"]

        match config.scheduler_type:
            case SchedulerType.CONSTANT_LR:
                lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lambda _: 1.0)
            case SchedulerType.STEP_LR:
                lr_scheduler = StepLR(optimizer=self.optimizer, step_size=scheduler_step_size, gamma=config.gamma)
            case SchedulerType.EXPONENTIAL_LR:
                lr_scheduler = ExponentialLR(optimizer=self.optimizer, gamma=config.gamma)
            case SchedulerType.COSINE_ANNEALING_LR:
                lr_scheduler = CosineAnnealingLR(
                    optimizer=self.optimizer, T_max=scheduler_step_size, eta_min=config.eta_min
                )
            case SchedulerType.REDUCE_LR_ON_PLATEAU:
                lr_scheduler = cast(
                    LRScheduler,
                    ReduceLROnPlateau(
                        optimizer=self.optimizer,
                        mode=config.mode,
                        factor=config.factor,
                        patience=config.patience,
                        threshold=config.threshold,
                        cooldown=config.cooldown,
                        min_lr=config.min_lr,
                    ),
                )
            case SchedulerType.LAMBDA_LR:
                assert config.lr_lambda is not None, "lr_lambda must be specified to use LambdaLR"
                lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=config.lr_lambda)
            case SchedulerType.ONE_CYCLE_LR:
                lr_scheduler = OneCycleLR(
                    optimizer=self.optimizer, max_lr=config.max_lr, total_steps=scheduler_step_size
                )
            case SchedulerType.MULTIPLICATIVE_LR:
                assert config.lr_lambda is not None, "lr_lambda must be specified to use MultiplicativeLR"
                lr_scheduler = MultiplicativeLR(optimizer=self.optimizer, lr_lambda=config.lr_lambda)
            case SchedulerType.COSINE_ANNEALING_WARM_RESTARTS:
                lr_scheduler = CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=500)
            case SchedulerType.CYCLIC_LR:
                lr_scheduler = CyclicLR(optimizer=self.optimizer, base_lr=config.base_lr, max_lr=config.max_lr)
            case SchedulerType.MULTI_STEP_LR:
                lr_scheduler = MultiStepLR(optimizer=self.optimizer, milestones=config.milestones, gamma=config.gamma)
            case _:
                raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")

        warmup_scheduler_steps = self.clock.convert_time_value(config.warmup, config.update_interval["unit"])
        if warmup_scheduler_steps > 0:
            lr_scheduler = WarmupScheduler(
                optimizer=self.optimizer,
                scheduler=lr_scheduler,
                warmup_scheduler_steps=warmup_scheduler_steps,
            )

        return lr_scheduler

    @cached_property
    def sharding_manager(self) -> ShardingManager:
        # TODO : implement accelerate and fabric sharding manager
        return SimpleShardingManager(self.config.training)

    @cached_property
    def models(self) -> dict[str, fl.Module]:
        return self.load_models()

    def set_models_to_train_mode(self) -> None:
        for model in self.models.values():
            model.train()

    def set_models_to_eval_mode(self) -> None:
        for model in self.models.values():
            model.eval()

    def prepare_model(self, model_name: str) -> None:
        model = self.models[model_name]
        
        ## hardcore fix for #226
        # dropout_config = self.config.dropout
        # chain_models = [model for model in self.models.values() if isinstance(model, fl.Chain)]
        # for model in chain_models:
        #     dropout_config.apply_dropout(model=model)
                
        if (checkpoint := self.config.models[model_name].checkpoint) is not None:
            model.load_from_safetensors(tensors_path=checkpoint)
        else:
            logger.info(f"No checkpoint found. Initializing model `{model_name}` from scratch.")
        model.requires_grad_(requires_grad=self.config.models[model_name].train)
        model.zero_grad()
        self.sharding_manager.setup_model(model=model, config=self.config.models[model_name])

    def prepare_models(self) -> None:
        assert self.models, "No models found."
        for model_name in self.models:
            self.prepare_model(model_name=model_name)

<<<<<<< HEAD:src/refiners/training_utils/trainers/trainer.py
    def prepare_checkpointing(self) -> None:
        if self.config.checkpointing.save_folder is not None:
            assert self.config.checkpointing.save_folder.is_dir()
            self.checkpoints_save_folder = (
                self.config.checkpointing.save_folder / self.wandb.project_name / self.wandb.run_name
            )
            self.checkpoints_save_folder.mkdir(parents=True, exist_ok=False)
            logger.info(f"Checkpointing enabled: {self.checkpoints_save_folder}")
        if self.config.checkpointing.use_wandb:            
            self.checkpoints_save_folder = Path(self.wandb.dir)
            logger.info(f"Checkpointing enabled: {self.checkpoints_save_folder}")            
        else:
            self.checkpoints_save_folder = None
            logger.info("Checkpointing disabled: configure `save_folder` to turn it on.")
        
=======
>>>>>>> main:src/refiners/training_utils/trainer.py
    @abstractmethod
    def load_models(self) -> dict[str, fl.Module]:
        ...

    @abstractmethod
    def get_item(self, index: int) -> Batch:
        """
        Returns a batch of data.

        This function is used by the dataloader to fetch a batch of data.
        """
        ...

    @abstractproperty
    def dataset_length(self) -> int:
        """
        Returns the length of the dataset.

        This is used to compute the number of batches per epoch.
        """
        ...

    @abstractmethod
    def collate_fn(self, batch: list[Batch]) -> Batch:
        """
        Collate function for the dataloader.

        This function is used to tell the dataloader how to combine a list of
        batches into a single batch.
        """
        ...

    @cached_property
    def dataset(self) -> Dataset[Batch]:
        """
        Returns the dataset constructed with the `get_item` method.
        """
        return _Dataset(get_item=self.get_item, length=self.dataset_length)

    @cached_property
    def dataloader(self) -> DataLoader[Any]:
        return DataLoader(
<<<<<<< HEAD:src/refiners/training_utils/trainers/trainer.py
            dataset=self.dataset, batch_size=self.config.training.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self.config.training.num_workers
=======
            dataset=self.dataset, batch_size=self.config.training.batch_size, shuffle=True, collate_fn=self.collate_fn
>>>>>>> main:src/refiners/training_utils/trainer.py
        )

    @abstractmethod
    def compute_loss(self, batch: Batch) -> Tensor:
        ...
    
    def eval(self) -> None:
        """Perform a single epoch."""
        for batch in self.eval_dataloader:
            if not self.clock.done:
                self._call_callbacks(event_name="on_batch_begin")
                self.step(batch=batch)
                self._call_callbacks(event_name="on_batch_end")
            else:
                break

    def compute_evaluation(self) -> None:
        pass

    def backward(self) -> None:
        """Backward pass on the loss."""
        self._call_callbacks(event_name="on_backward_begin")
        scaled_loss = self.loss / self.clock.num_step_per_iteration
        self.sharding_manager.backward(scaled_loss)
        self._call_callbacks(event_name="on_backward_end")
        if self.clock.is_optimizer_step:
            self._call_callbacks(event_name="on_optimizer_step_begin")
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._call_callbacks(event_name="on_optimizer_step_end")
        if self.clock.is_lr_scheduler_step:
            self._call_callbacks(event_name="on_lr_scheduler_step_begin")
            self.lr_scheduler.step()
            self._call_callbacks(event_name="on_lr_scheduler_step_end")
        if self.clock.is_evaluation_step:
            self.evaluate()

    def step(self, batch: Batch) -> None:
        """Perform a single training step."""
        self._call_callbacks(event_name="on_compute_loss_begin")
        loss = self.compute_loss(batch=batch)
        self.loss = loss
        self._call_callbacks(event_name="on_compute_loss_end")
        self.backward()

    def epoch(self) -> None:
        """Perform a single epoch."""
        for batch in self.dataloader:
            if self.clock.done:
                break
            self._call_callbacks(event_name="on_batch_begin")
            self.step(batch=batch)
            self._call_callbacks(event_name="on_batch_end")
            
    @staticmethod
    def get_training_seed(instance: "Trainer[BaseConfig, Any]") -> int:
        return instance.config.training.seed

    @scoped_seed(seed=get_training_seed)
    def train(self) -> None:
        """Train the model."""
        self.set_models_to_train_mode()
        self._call_callbacks(event_name="on_train_begin")
        assert self.learnable_parameters, "There are no learnable parameters in the models."
        self.evaluate()
        while not self.clock.done:
            self._call_callbacks(event_name="on_epoch_begin")
            self.epoch()
            self._call_callbacks(event_name="on_epoch_end")
        self._call_callbacks(event_name="on_train_end")

    @staticmethod
    def get_evaluation_seed(instance: "Trainer[BaseConfig, Any]") -> int:
        return instance.config.training.evaluation_seed

    @no_grad()
    @scoped_seed(seed=get_evaluation_seed)
    def evaluate(self) -> None:
        """Evaluate the model."""
        self.set_models_to_eval_mode()
        self._call_callbacks(event_name="on_evaluate_begin")
        self.compute_evaluation()
        self._call_callbacks(event_name="on_evaluate_end")
        self.set_models_to_train_mode()

    def _call_callbacks(self, event_name: str) -> None:
        for callback in self.callbacks:
            getattr(callback, event_name)(self)


