# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default experiment configuration.
"""

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING, DictConfig, OmegaConf

@dataclass
class ClassInitSpecification:
    """Class initialization specification
    """
    # Absolute path to the class
    cls_path: str = MISSING

    # Arguments used in the class constructor
    args: list = field(default_factory=list)

    # Named arguments used in the class constructor
    kwargs: dict = field(default_factory=dict)

@dataclass
class SystemConfig:
    """System configuration
    """
        # Whether training is distributed
    distributed: bool = True
    # Backend to use for distributed training
    distributed_backend: str = 'nccl'
    # Number of CPUs in system (for dataset loading multi-processing)
    num_cpu: int = 16
    # Number of GPUs in system (for dataset loading distributed multi-processing)
    num_gpu: int = 1

@dataclass
class GradNormClipConfig:
    """Gradient norm clipping configuration
    """
    max_norm: float = 10.0
    norm_type: float = 2.0

@dataclass
class TrainerConfig:
    """Trainer configuration
    """
    # Training batch size
    batch_size: int = 16
    # Whether to train with mixed precision
    mixed_precision: bool = True

    total_iterations: int = 100000
    # Training iterations between evaluations
    test_every: int = 10000
    # Training iterations between log events
    log_every: int = 50
    # Training iterations between image logging
    log_metric_every: int = 500
    # Training iterations between "latest" checkpointing
    checkpoint_every: int = 5000
    # Training iterations between persistent checkpointing
    persistent_checkpoint_every: int = 50000
    # Log with wandb
    use_wandb: bool = True
    # Log with lpip
    log_lpips: bool = True
    # Log with fid
    log_fid: bool = True
    # Optimizer related: TODO: integrate
    lr: float = 1e-4
    lr_warmup_steps: int = 500
    num_training_timesteps: int = 1000
    # Training objective. Supported values: "diffusion" and "flow_matching".
    objective: str = "diffusion"
    # Shift used by the flow-matching scheduler when objective == "flow_matching".
    flow_shift: float = 1.0
    # Min-SNR-gamma loss weighting (https://arxiv.org/abs/2303.09556). None disables it.
    min_snr_gamma: Optional[float] = None
    # Gradient accumulation depth used only for the first `warmup_grad_accum_iters`
    # training iterations. 1 (default) disables accumulation.
    warmup_grad_accum_steps: int = 1
    # Number of initial training iterations (global_step) over which to apply the
    # warmup gradient accumulation. 0 (default) disables the warmup.
    warmup_grad_accum_iters: int = 0
    # Betas determine the distribution of noise seen during training
    beta_a: float = 1
    beta_b: float = 2.5
    compile: bool = False
    # Prob for CFG
    p_cfg: float = 0.0
    # Weights for the latent regularization
    alpha_reg: float = 0.1
    vae_warmup_steps: int = 0
    # Identity Augmentation Frequency
    augment_every: int=-1

    # The path to the trainer
    cls_path: str = MISSING
    # Extra trainer kwargs
    trainer_kwargs: dict = field(default_factory=dict)

    # If not None, activates EMA with the given weight
    ema_mu: Optional[float] = None

    # Optimizer config
    optimizer: ClassInitSpecification = field(default_factory=ClassInitSpecification)
    # Scheduler config
    scheduler: Optional[ClassInitSpecification] = None
    # Gradient clipping config
    grad_clipping: Optional[GradNormClipConfig] = None

    def __post_init__(self):
        # Ensure correct initialization of all nested dataclasses via omegaconf
        if isinstance(self.optimizer, DictConfig):
            self.optimizer = ClassInitSpecification(**OmegaConf.to_container(self.optimizer))
        elif isinstance(self.optimizer, dict):
            self.optimizer = ClassInitSpecification(**self.optimizer) # pylint: disable=E1134

        if isinstance(self.scheduler, DictConfig):
            self.scheduler = ClassInitSpecification(**OmegaConf.to_container(self.scheduler))
        elif isinstance(self.scheduler, dict):
            self.scheduler = ClassInitSpecification(**self.scheduler)

        if isinstance(self.grad_clipping, DictConfig):
            self.grad_clipping = GradNormClipConfig(**OmegaConf.to_container(self.scheduler))
        elif isinstance(self.grad_clipping, dict):
            self.grad_clipping = GradNormClipConfig(**self.grad_clipping)

@dataclass
class DatasetConfig:
    """Dataset configuration
    """
    collection: ClassInitSpecification = field(default_factory=ClassInitSpecification)
    # Directory to cache dataset in
    data_dir: str = "dataset/"

    # List of training dataset sources
    train_sets: list = field(default_factory=list)

    # List of test dataset sources
    test_sets: list = field(default_factory=list)

    def __post_init__(self):
        # Ensure correct initialization of all nested dataclasses via omegaconf
        if isinstance(self.collection, DictConfig):
            self.collection = ClassInitSpecification(
                **OmegaConf.to_container(self.collection)
            )
        elif isinstance(self.collection, dict):
            self.collection = ClassInitSpecification(**self.collection) # pylint: disable=E1134

@dataclass
class ExperimentConfig:
    """Full experiment config
    """
    checkpoint: Optional[str] = None
    # Enable profiling
    enable_profiling: bool = False
    # The root output directory where experiment outputs are saved
    exp_root_dir: str = "experiments"
    # Project name (used for logging)
    project_name: str = "your_project_name"
    # Project experiment subgroup (used for logging)
    project_subgroup: str = "your_project_name_base"
    # System specification
    system: SystemConfig = field(default_factory=SystemConfig)
    # Model specification
    model: ClassInitSpecification = field(default_factory=ClassInitSpecification)
    # Loss specification
    loss: ClassInitSpecification = field(default_factory=ClassInitSpecification)
    # Trainer specification
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    # Dataset specification
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def __post_init__(self):
        # Ensure correct initialization of all nested dataclasses via omegaconf
        if isinstance(self.system, DictConfig):
            self.system = SystemConfig(**OmegaConf.to_container(self.system))
        if isinstance(self.model, DictConfig):
            self.model = ClassInitSpecification(**OmegaConf.to_container(self.model))
        if isinstance(self.loss, DictConfig):
            self.loss = ClassInitSpecification(**OmegaConf.to_container(self.loss))
        if isinstance(self.trainer, DictConfig):
            self.trainer = TrainerConfig(**OmegaConf.to_container(self.trainer))
        if isinstance(self.dataset, DictConfig):
            self.dataset = DatasetConfig(**OmegaConf.to_container(self.dataset))
