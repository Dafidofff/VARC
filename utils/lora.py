"""Low-rank (LoRA) adapters for test-time training on the Hyena mixer projections.

Backlog Phase 3: the under-fit diagnosis (see ttt_autoresearch.md) implicates the
*spatial mixer* — TTT never reshapes the qkv/out projections enough to fit the support
transform on the harder tasks, yet letting the full projections train (the unfrozen
baseline) does not help either (freeze-mixer == baseline; the full update is inert/noisy).

LoRA adds *constrained* test-time fitting capacity to exactly those projections: the base
weight is frozen and a rank-r delta (B @ A, B zero-initialised so the adapter starts as a
no-op) is the only thing that adapts the mixer. This is a controllable version of what the
FiLM checkpoint achieved structurally.

Injection targets the ``qkv_proj`` and ``out_proj`` ``nn.Linear`` layers inside every Hyena
sequence-mixer block (see nvSubquadratic-private sequence_mixer.py).
"""
import math
from typing import Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an ``nn.Linear`` with a frozen base weight + a trainable rank-r delta.

    forward(x) = base(x) + scaling * B(A(x)), with scaling = alpha / rank and B zero-init,
    so the wrapped layer is numerically identical to the base layer at TTT step 0.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_features, out_features = base.in_features, base.out_features
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    target_suffixes: Tuple[str, ...] = ("qkv_proj", "out_proj"),
) -> int:
    """Replace every ``nn.Linear`` whose attribute name is in ``target_suffixes`` with a
    ``LoRALinear``. Returns the number of layers wrapped.

    Matches on the leaf attribute name (e.g. ``...mixer.qkv_proj``), so only the Hyena
    mixer projections are touched — not the readout/embedding/AdaLN ``Linear`` layers.
    """
    wrapped = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if child_name in target_suffixes and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                wrapped += 1
    return wrapped
