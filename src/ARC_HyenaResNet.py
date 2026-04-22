"""Wrapper that makes nvSubquadratic ARCResNet compatible with the VARC training loop.

The VARC loop calls:  model(pixel_values, task_ids, attention_mask=...)
ARCResNet expects:    model({"input": pixel_values, "condition": {"task_id": task_ids}})

This module instantiates the model from a config file path and exposes it
with the VARC interface so it can be dropped in via --architecture hyena.
"""

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

NVSUBQ_ROOT = Path("/home/dwessel/code/nvSubquadratic-private")


def _ensure_nvsubq_on_path() -> None:
    if str(NVSUBQ_ROOT) not in sys.path:
        sys.path.insert(0, str(NVSUBQ_ROOT))


def build_hyena_arc_resnet(config_path: str, num_tasks: int) -> nn.Module:
    """Instantiate ARCResNet from a nvSubquadratic config file.

    Args:
        config_path: Absolute path to a cfg_*.py that exposes get_config().
        num_tasks:   Override the number of task tokens (matches dataset).
    """
    _ensure_nvsubq_on_path()

    from nvsubquadratic.lazy_config import instantiate

    spec = importlib.util.spec_from_file_location("_hyena_cfg", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)

    config = cfg_module.get_config()

    # Patch num_tasks in case the config hardcodes a different value.
    # config.net is an OmegaConf DictConfig with top-level keys (not nested under kwargs).
    config.net.num_tasks = num_tasks

    return instantiate(config.net)


class HyenaResNetVARCWrapper(nn.Module):
    """Wraps ARCResNet to match the VARC model interface."""

    def __init__(self, arc_resnet: nn.Module) -> None:
        super().__init__()
        self.arc_resnet = arc_resnet

    def forward(
        self,
        pixel_values: torch.Tensor,
        task_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values:    [B, H, W] integer colour grid.
            task_ids:        [B] integer task indices.
            attention_mask:  [B, H, W] mask (unused by ARCResNet; loss handles padding).
        Returns:
            logits: [B, num_colors, H, W]
        """
        out = self.arc_resnet({"input": pixel_values, "condition": {"task_id": task_ids}})
        return out["logits"]
