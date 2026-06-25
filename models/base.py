"""Base model class for all TorchaVerse models.

All model implementations inherit from :class:`BaseModel`, which provides
a common interface for configuration, parameter counting, and weight
serialization.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for all models in TorchaVerse.

    Args:
        config: Optional configuration dictionary.  Subclasses can
            access ``self.config`` to retrieve architecture hyperparameters.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config: dict[str, Any] = config or {}

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the total number of parameters.

        Args:
            trainable_only: If ``True``, count only parameters with
                ``requires_grad=True``.

        Returns:
            The parameter count.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def num_parameters_human(self, trainable_only: bool = True) -> str:
        """Return a human-readable parameter count string."""
        n = self.num_parameters(trainable_only)
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.2f}B"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        if n >= 1_000:
            return f"{n / 1_000:.2f}K"
        return str(n)

    def save(self, path: str) -> None:
        """Save model weights to *path* in safetensors format."""
        state_dict = self.state_dict()
        try:
            from safetensors.torch import save_file
            save_file(state_dict, path)
        except ImportError:
            torch.save(state_dict, path)

    def load(self, path: str, strict: bool = True) -> None:
        """Load model weights from *path*.

        Prefers safetensors (safe format).  Falls back to
        ``torch.load(..., weights_only=True)`` which refuses pickle
        execution.  Use :meth:`load_unsafe` to opt in to pickle loading
        for legacy self-produced checkpoints.
        """
        try:
            from safetensors.torch import load_file
            state_dict = load_file(path)
        except ImportError:
            # safetensors unavailable; use torch.load with the safe
            # default.  The previous ``weights_only=False`` fallback
            # allowed pickle RCE; we refuse that path here.
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict, strict=strict)

    def load_unsafe(self, path: str, strict: bool = True) -> None:
        """Load model weights from a pickle checkpoint.

        This is **insecure** (allows arbitrary code execution) and
        should only be used to load self-produced checkpoints from a
        trusted location.  Prefer :meth:`load` (safetensors) whenever
        possible.
        """
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state_dict, strict=strict)
