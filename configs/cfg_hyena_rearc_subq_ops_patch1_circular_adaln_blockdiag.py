"""Hyena ResNet for ARC-AGI — patch=1 + circular padding + AdaLN-Zero + block-diagonal learnable-ω₀ kernel.

Builds on cfg_hyena_rearc_subq_ops_patch1_circular_adaln.py by swapping the
per-block Hyena kernel for the ImageNet-tuned block-diagonal learnable-ω₀ variant:

  - kernel: SIRENKernelND → BlockDiagonalLearnableOmegaSIRENKernelND
            (num_blocks=8, ω₀ ∈ [1, 12] linear schedule, off_block_scale=0.1,
             per-row scale clamp [1e-2, 2], apply_lr_scale=True)
  - mask:   Identity → BlockAlignedGaussianModulationND

patch_size=1 → seq_len = 64×64 = 4096 tokens (VARC --image-size 64).
L_cache=64 to match the actual VARC input resolution.

Stored in VARC/configs/ (not nvSubquadratic-private) so it is immune to branch changes.
"""

import math

import torch

from experiments.datamodules.arc import ARCDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.arc_wrapper import ARCWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import BlockDiagonalLearnableOmegaSIRENKernelND
from nvsubquadratic.modules.masks_nd import BlockAlignedGaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import Patchify, Unpatchify
from nvsubquadratic.modules.residual_block import AdaLNZeroResidualBlock
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.arc_resnet import ARCResNet
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


# ── Architecture ──────────────────────────────────────────────────────────────
EMBED_DIM = 384
NUM_BLOCKS = 12
PATCH_SIZE = 1
MAX_SIZE = 32
NUM_COLORS = 12
NUM_TASKS = 400

# ── Hyena ─────────────────────────────────────────────────────────────────────
DATA_DIM = 2
PATCHED_RESOLUTION = 64  # VARC uses --image-size 64; L_cache matches actual input
FFT_PADDING = "circular"
GRID_TYPE = "single"
DROPOUT = 0.1

# ── Block-diagonal learnable-ω₀ kernel ────────────────────────────────────────
KERNEL_MLP_HIDDEN_DIM = 64
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 64
KERNEL_HIDDEN_OMEGA_0 = 1.0

KERNEL_BLOCK_DIAG_NUM_BLOCKS = 8
KERNEL_BLOCK_DIAG_OMEGA_0_MIN = 1.0
KERNEL_BLOCK_DIAG_OMEGA_0_MAX = 12.0
KERNEL_BLOCK_DIAG_SCHEDULE = "linear"
KERNEL_BLOCK_DIAG_OFF_BLOCK_SCALE = 0.1

KERNEL_LEARNABLE_OMEGA_0_SCALE_MIN = 1e-2
KERNEL_LEARNABLE_OMEGA_0_SCALE_MAX = 2.0
KERNEL_LEARNABLE_OMEGA_APPLY_LR_SCALE = True

# ── Training placeholders (overridden by VARC offline_train_ARC.py CLI args) ──
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
PLACEHOLDER = None
BATCH_SIZE = 16
GRAD_ACCUM = 2
NUM_GPUS = 8
NUM_TRAINING_SAMPLES_REARC = 413_020


def get_config() -> ExperimentConfig:
    """Hyena ResNet ARC: patch=1 + circular + AdaLN-Zero + block-diagonal learnable-ω₀."""
    training_iterations = math.ceil(NUM_EPOCHS * NUM_TRAINING_SAMPLES_REARC / (BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))

    config = ExperimentConfig()
    config.debug = False
    config.seed = 42

    config.dataset = LazyConfig(ARCDataModule)(
        data_dir="/home/dwessel/code/VARC/raw_data/ARC-AGI/data",
        rearc_dir="/home/dwessel/code/VARC/raw_data/re_arc",
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        seed=config.seed,
        max_size=MAX_SIZE,
        num_color_permutations=9,
        rearc_num_color_permutations=0,
        val_task_split="training",
        val_subset="test",
    )

    config.lightning_wrapper_class = LazyConfig(ARCWrapper)()
    config.optimizer = LazyConfig(torch.optim.AdamW)(params=PLACEHOLDER, lr=LEARNING_RATE, weight_decay=0.0)
    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=training_iterations,
        grad_clip=1.0,
        accumulate_grad_steps=GRAD_ACCUM,
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=0.05,
        total_iterations="${train.iterations}",
        mode="max",
    )
    config.trainer.checkpoint_monitor = "val/exact_match"
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.compile_compatible_fftconv = True
    config.trainer.precision = "bf16-mixed"

    norm_cfg = LazyConfig(RMSNorm)(dim=EMBED_DIM)

    resnet_cfg = LazyConfig(ResidualNetwork)(
        in_channels=EMBED_DIM,
        out_channels=NUM_COLORS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=EMBED_DIM,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(Patchify)(
            in_features=EMBED_DIM,
            out_features=EMBED_DIM,
            data_dim=DATA_DIM,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        ),
        out_proj_cfg=LazyConfig(Unpatchify)(
            in_features=EMBED_DIM,
            out_features=NUM_COLORS,
            data_dim=DATA_DIM,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        ),
        norm_cfg=norm_cfg,
        block_cfg=LazyConfig(AdaLNZeroResidualBlock)(
            hidden_dim=EMBED_DIM,
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim=EMBED_DIM,
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim=DATA_DIM,
                        hidden_dim=EMBED_DIM,
                        fft_padding=FFT_PADDING,
                        use_fp16_fft=False,
                        fft_backend="torch_fft",
                        kernel_cfg=LazyConfig(BlockDiagonalLearnableOmegaSIRENKernelND)(
                            data_dim=DATA_DIM,
                            out_dim=EMBED_DIM,
                            mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                            num_layers=KERNEL_NUM_LAYERS,
                            embedding_dim=KERNEL_EMBEDDING_DIM,
                            L_cache=PATCHED_RESOLUTION,  # 64
                            use_bias=True,
                            hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                            num_blocks=KERNEL_BLOCK_DIAG_NUM_BLOCKS,
                            omega_0_min=KERNEL_BLOCK_DIAG_OMEGA_0_MIN,
                            omega_0_max=KERNEL_BLOCK_DIAG_OMEGA_0_MAX,
                            schedule=KERNEL_BLOCK_DIAG_SCHEDULE,
                            off_block_scale=KERNEL_BLOCK_DIAG_OFF_BLOCK_SCALE,
                            omega_0_scale_min=KERNEL_LEARNABLE_OMEGA_0_SCALE_MIN,
                            omega_0_scale_max=KERNEL_LEARNABLE_OMEGA_0_SCALE_MAX,
                            apply_lr_scale=KERNEL_LEARNABLE_OMEGA_APPLY_LR_SCALE,
                        ),
                        mask_cfg=LazyConfig(BlockAlignedGaussianModulationND)(
                            data_dim=DATA_DIM,
                            num_channels=EMBED_DIM,
                            min_attenuation_at_step=0.1,
                            max_attenuation_at_limit=0.95,
                            init_extent=1.0,
                            parametrization="direct",
                        ),
                        grid_type=GRID_TYPE,
                    ),
                    short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                        in_channels=3 * EMBED_DIM,
                        out_channels=3 * EMBED_DIM,
                        kernel_size=3,
                        groups=3 * EMBED_DIM,
                        padding=1,
                        bias=False,
                    ),
                    gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
                    gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
                    pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=EMBED_DIM),
                    output_norm_cfg=LazyConfig(RMSNorm)(dim=EMBED_DIM),
                    qk_norm_cfg=LazyConfig(L2Norm)(),
                ),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            sequence_mixer_norm_cfg=norm_cfg,
            condition_norm_cfg=norm_cfg,
            mlp_cfg=LazyConfig(MLP)(
                dim=EMBED_DIM,
                activation="glu",
                expansion_factor=1.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            mlp_norm_cfg=norm_cfg,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
    )

    config.net = LazyConfig(ARCResNet)(
        num_tasks=NUM_TASKS,
        num_colors=NUM_COLORS,
        hidden_dim=EMBED_DIM,
        resnet_cfg=resnet_cfg,
        task_injection="film",
    )

    config.wandb = WandbConfig(entity="implicit-long-convs", project="nvsubquadratic", job_group="arc_patch_ablation")

    return config
