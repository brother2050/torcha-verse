"""ControlNet paper adapter (v0.9.0).

Project-internal ControlNet adapter (Zhang et al., 2023 --
arXiv:2302.05543) that mirrors the architectural surface of
the diffusers :class:`diffusers.ControlNetModel` and the
ComfyUI ``ControlNetApply`` node, while keeping the runtime
footprint limited to ``torch`` and the project-internal
:mod:`core.checkpoint_loader`.

Design notes
------------

* **Source of truth.**  diffusers ``ControlNetModel`` is the
  architectural reference; the ComfyUI ``ControlNetApply``
  node is the integration reference -- it returns a *list* of
  residual feature maps (one per down-block) that the caller
  adds back into the UNet's skip connections.  We expose the
  same list-of-residuals contract from
  :meth:`ControlNetAdapter.forward`.

* **Tiny footprint.**  :class:`ControlNetConfig` defaults to
  ``num_layers=4`` with ``block_out_channels=(64, 128, 256,
  512)`` -- a "Tiny" preset that fits inside a 256x256 latent
  grid.

* **Integration with** :class:`core.offload.ModelPatcher**.
  ControlNet is a *patch* on the base UNet: it adds residuals,
  it does not replace parameters.  When the host UNet is
  wrapped in a :class:`ModelPatcher`, the ControlNet weights
  live on the same module graph and are subject to the same
  offload policy (see ``enable_model_cpu_offload`` /
  ``enable_sequential_cpu_offload`` in :mod:`core.offload`).
  There is no separate weight copy; the patcher sees the
  control network as just another :class:`nn.Module` in the
  tree.

Public surface: :class:`ControlNetConfig`,
:class:`ControlNetAdapter`, :data:`CONTROLNET_KEY_MAP`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from core.checkpoint_loader import (
    load_safetensors,
    load_state_dict_with_renames,
    save_safetensors,
)

__all__ = ["ControlNetConfig", "ControlNetAdapter", "CONTROLNET_KEY_MAP"]


# Key rename table: diffusers -> local naming.
# Consumed by :meth:`ControlNetAdapter.from_pretrained` (delegates to
# :func:`core.checkpoint_loader.load_state_dict_with_renames`).  The
# diffusers ``ControlNetModel`` publishes ``controlnet_cond_embedding``
# (the hint encoder), ``down_blocks`` / ``mid_block`` (the encoder
# half of the UNet), and ``controlnet_down_blocks`` /
# ``controlnet_mid_block`` (the ControlNet-specific zero-conv
# outputs).  We collapse ``controlnet_cond_embedding`` into
# ``hint_in`` and ``controlnet_down_blocks.{i}`` into ``zero_conv.{i}``;
# the rest is 1:1.
CONTROLNET_KEY_MAP: Dict[str, str] = {
    # controlnet_cond_embedding (the hint encoder)
    "controlnet_cond_embedding.conv_in.weight":  "hint_in.0.weight",
    "controlnet_cond_embedding.conv_in.bias":    "hint_in.0.bias",
    "controlnet_cond_embedding.conv_out.weight": "hint_in.2.weight",
    "controlnet_cond_embedding.conv_out.bias":   "hint_in.2.bias",
    # down_blocks.{i}.resnets.{j}  (1:1 to local naming)
    "down_blocks.{i}.resnets.{j}.norm1.weight": "down_blocks.{i}.resnets.{j}.norm1.weight",
    "down_blocks.{i}.resnets.{j}.norm1.bias":   "down_blocks.{i}.resnets.{j}.norm1.bias",
    "down_blocks.{i}.resnets.{j}.conv1.weight": "down_blocks.{i}.resnets.{j}.conv1.weight",
    "down_blocks.{i}.resnets.{j}.conv1.bias":   "down_blocks.{i}.resnets.{j}.conv1.bias",
    "down_blocks.{i}.resnets.{j}.conv2.weight": "down_blocks.{i}.resnets.{j}.conv2.weight",
    "down_blocks.{i}.resnets.{j}.conv2.bias":   "down_blocks.{i}.resnets.{j}.conv2.bias",
    # down_blocks.{i}.downsamplers.0.conv  (1:1)
    "down_blocks.{i}.downsamplers.0.conv.weight": "down_blocks.{i}.downsamplers.0.conv.weight",
    "down_blocks.{i}.downsamplers.0.conv.bias":   "down_blocks.{i}.downsamplers.0.conv.bias",
    # mid_block.resnets.0  (one resnet in the tiny clone)
    "mid_block.resnets.0.norm1.weight": "mid_block.resnets.0.norm1.weight",
    "mid_block.resnets.0.norm1.bias":   "mid_block.resnets.0.norm1.bias",
    "mid_block.resnets.0.conv1.weight": "mid_block.resnets.0.conv1.weight",
    "mid_block.resnets.0.conv1.bias":   "mid_block.resnets.0.conv1.bias",
    # controlnet zero-conv outputs (the "control" in ControlNet)
    "controlnet_down_blocks.{i}.weight": "zero_conv.{i}.weight",
    "controlnet_down_blocks.{i}.bias":   "zero_conv.{i}.bias",
    "controlnet_mid_block.weight":       "zero_conv_mid.weight",
    "controlnet_mid_block.bias":         "zero_conv_mid.bias",
}


def _materialise_per_block_map(num_layers: int) -> Dict[str, str]:
    """Expand ``{i}`` / ``{j}`` placeholders in
    :data:`CONTROLNET_KEY_MAP` for the load helpers."""
    out: Dict[str, str] = {}
    for k, v in CONTROLNET_KEY_MAP.items():
        if "{i}" in k or "{j}" in k:
            for i in range(num_layers):
                for j in range(2):  # each diffusers down-block has 2 resnets
                    out[k.format(i=i, j=j)] = v.format(i=i, j=j)
        else:
            out[k] = v
    return out


# Config


@dataclass
class ControlNetConfig:
    """Architectural configuration for :class:`ControlNetAdapter`.

    Attributes:
        in_channels: Latent channel count (matches the host UNet).
        hint_channels: Channels in the *control hint* (canny=1,
            depth=1, RGB=3, ...).
        cross_attention_dim: Latent dim of the text conditioning.
        num_layers: Number of downsample stages (== number of
            zero-conv outputs).  Defaults to ``4`` (Tiny).
        block_out_channels: Output channels of the downsample
            stack.  Length must equal ``num_layers``.
        downsample: Whether each down-block (after the first)
            halves the spatial resolution.
    """

    in_channels: int = 3
    hint_channels: int = 3
    cross_attention_dim: int = 768
    num_layers: int = 4
    block_out_channels: Tuple[int, ...] = (64, 128, 256, 512)
    downsample: bool = True

    def __post_init__(self) -> None:
        # Coerce tuples; do not silently truncate -- mismatched
        # lengths are a hard error.
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0; got {self.num_layers!r}")
        if len(self.block_out_channels) != self.num_layers:
            raise ValueError(
                f"block_out_channels must have length == num_layers "
                f"({self.num_layers}); got {len(self.block_out_channels)}"
            )
        if any(c <= 0 for c in self.block_out_channels):
            raise ValueError(
                f"block_out_channels must be all-positive; "
                f"got {tuple(self.block_out_channels)!r}"
            )
        if self.hint_channels <= 0:
            raise ValueError(f"hint_channels must be > 0; got {self.hint_channels!r}")
        self.block_out_channels = tuple(int(c) for c in self.block_out_channels)

    @classmethod
    def tiny(cls) -> "ControlNetConfig":
        """Return the Tiny preset (4 layers, 64..512 channels)."""
        return cls(
            in_channels=3, hint_channels=3, cross_attention_dim=768,
            num_layers=4, block_out_channels=(64, 128, 256, 512),
            downsample=True,
        )

    @property
    def hint_in_channels(self) -> int:
        """Channels of the first conv in the hint encoder."""
        return int(self.hint_channels)

    @property
    def latent_channels(self) -> int:
        """Alias for :attr:`in_channels` (diffusers-style naming)."""
        return int(self.in_channels)


# Building blocks


class _ZeroConv2d(nn.Module):
    """A 1x1 conv whose weight and bias are initialised to zero.

    The zero initialisation is the *defining* trick of
    ControlNet (Zhang et al., 2023, sec. 4): at the start of
    training every residual is exactly zero, so the
    ControlNet-patched UNet is bit-for-bit identical to the
    base UNet.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.conv = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=1, padding=0, bias=True,
        )
        # Zero-initialise.  No reset_parameters afterwards --
        # the zero assignment is the only init step.
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _HintBlock(nn.Module):
    """A single downsample stage in the hint encoder.

    Mirrors the diffusers ``controlnet_cond_embedding`` block:
    one 3x3 conv (optionally stride 2) followed by a SiLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=True,
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class _DownBlock(nn.Module):
    """A single down-block in the control network's main path.

    Tiny copy of the diffusers UNet down-block: one 3x3 conv
    (optionally stride 2) + GroupNorm + SiLU, then a residual
    3x3 conv.  No cross-attention -- the project-internal
    clone keeps the cross-attention in the host UNet.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=True,
        )
        self.norm1 = nn.GroupNorm(
            num_groups=min(32, out_channels), num_channels=out_channels,
            eps=1e-6, affine=True,
        )
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=True,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=min(32, out_channels), num_channels=out_channels,
            eps=1e-6, affine=True,
        )
        self.act2 = nn.SiLU()
        # Residual 1x1 projection when channels / stride change.
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=stride, padding=0, bias=True,
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        return h + residual

# Adapter


class ControlNetAdapter(nn.Module):
    """Project-internal ControlNet adapter (Zhang et al., 2023).

    Ingests a *control hint* (canny / depth / RGB pose / ...)
    and emits a list of residual feature maps -- one per
    down-block resolution -- that the caller adds back into
    the host UNet's skip connections (ComfyUI's
    ``ControlNetApply`` contract).

    :meth:`forward` returns the raw list of residuals;
    :meth:`apply` is a single-tensor convenience that adds
    the residuals to a flat conditioning tensor.

    The adapter is intentionally small: a hint encoder + N
    down-blocks + N zero-conv outputs.  The cross-attention
    dim is stored on the config for symmetry with diffusers
    but is **not** consumed by the project-internal clone.
    """

    def __init__(self, config: ControlNetConfig) -> None:
        super().__init__()
        self.config: ControlNetConfig = config
        # Hint encoder: a single 3x3 conv (stride 1) + SiLU + one
        # _HintBlock.  Mirrors ``controlnet_cond_embedding`` in
        # diffusers.
        self.hint_in = nn.Sequential(
            nn.Conv2d(
                config.hint_channels, config.block_out_channels[0],
                kernel_size=3, stride=1, padding=1, bias=True,
            ),
            nn.SiLU(),
            _HintBlock(
                config.block_out_channels[0],
                config.block_out_channels[0],
                downsample=False,
            ),
        )
        # Main downsample stack.  The first down-block keeps the
        # resolution; the rest halve it.  Matches the diffusers
        # convention where the *first* zero-conv is at the input
        # resolution.
        self.down_blocks = nn.ModuleList()
        in_ch = config.block_out_channels[0]
        for i, out_ch in enumerate(config.block_out_channels):
            downsample_first = bool(config.downsample) and i > 0
            self.down_blocks.append(
                _DownBlock(in_ch, out_ch, downsample=downsample_first)
            )
            in_ch = out_ch
        # One zero-conv per down-block (the residuals) + a mid-block zero-conv.
        self.zero_conv = nn.ModuleList(
            _ZeroConv2d(out_ch, out_ch)
            for out_ch in config.block_out_channels
        )
        self.zero_conv_mid = _ZeroConv2d(in_ch, in_ch)

    @property
    def num_residuals(self) -> int:
        """Number of per-block residuals :meth:`forward` returns.

        Equals :attr:`ControlNetConfig.num_layers`; surfaced as
        a property so callers can introspect the adapter
        without reading the config.
        """
        return int(self.config.num_layers)

    def forward(
        self,
        latent_model_input: torch.Tensor,
        control_image: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run the control network on ``control_image``.

        ``latent_model_input``, ``t`` and
        ``encoder_hidden_states`` are accepted for signature
        compatibility with the diffusers ``ControlNetModel``
        contract; only ``control_image`` is consumed.

        Returns:
            A list of length ``num_layers + 1``: the first
            ``num_layers`` entries are the zero-conv outputs
            at each down-block resolution; the final entry is
            the mid-block residual (``zero_conv_mid``).
        """
        del latent_model_input, t, encoder_hidden_states  # signature compat
        residuals: List[torch.Tensor] = []
        h = self.hint_in(control_image)
        for block, zero in zip(self.down_blocks, self.zero_conv):
            h = block(h)
            residuals.append(zero(h))
        residuals.append(self.zero_conv_mid(h))
        return residuals

    def apply(
        self,
        conditioning: torch.Tensor,
        control_image: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """High-level helper: add the residuals to ``conditioning``.

        *Single-tensor* convenience wrapper around
        :meth:`forward`.  Each per-block residual is
        bilinearly resized to ``conditioning``'s spatial shape
        and projected to ``conditioning``'s channel count via
        a 1x1 conv (zero-initialised -- the patcher is
        effectively a no-op until the convs are trained), then
        summed and added back in (scaled by ``strength``).
        The mid-block residual is dropped (it sits at the
        bottleneck resolution, not the conditioning
        resolution).

        Args:
            conditioning: ``[B, C, H, W]`` (or ``[C, H, W]``).
                Spatial size and channel count are the
                broadcast targets.
            control_image: The control hint.  Forwarded to
                :meth:`forward`.
            strength: Per-call scaling factor (``1.0`` = full
                effect, ``0.0`` = identity).

        Returns:
            ``conditioning + strength * residuals_sum``.
        """
        residuals = self.forward(
            latent_model_input=conditioning,
            control_image=control_image,
            t=torch.zeros(1),
            encoder_hidden_states=torch.zeros(1),
        )
        target_h = int(conditioning.shape[-2])
        target_w = int(conditioning.shape[-1])
        target_c = int(conditioning.shape[-3])
        per_block = residuals[: self.num_residuals]
        accum = torch.zeros_like(conditioning)
        for r in per_block:
            r_resized = nn.functional.interpolate(
                r, size=(target_h, target_w),
                mode="bilinear", align_corners=False,
            )
            # Project channels with a 1x1 conv (zero
            # kernel -- same trick as the down-block
            # zero-conv outputs).
            if r_resized.shape[1] != target_c:
                kernel = torch.zeros(
                    target_c, r_resized.shape[1], 1, 1,
                    device=r_resized.device,
                    dtype=r_resized.dtype,
                )
                r_resized = nn.functional.conv2d(r_resized, kernel)
            accum = accum + r_resized
        return conditioning + float(strength) * accum

    @classmethod
    def from_pretrained(
        cls,
        weights_path: Union[str, Path],
        **kwargs: Any,
    ) -> "ControlNetAdapter":
        """Load a :class:`ControlNetAdapter` from a safetensors file.

        The load goes through :mod:`core.checkpoint_loader`.
        Recognised kwargs: ``config`` (a
        :class:`ControlNetConfig`, default Tiny),
        ``torch_dtype``, ``device``, ``strict``.  The
        ``weights_path`` may be a diffusers-style state dict
        (re-keyed via :data:`CONTROLNET_KEY_MAP`) or a
        local-style state dict (no re-keying).
        """
        config: ControlNetConfig = kwargs.pop(
            "config", ControlNetConfig.tiny()
        )
        torch_dtype: Optional[torch.dtype] = kwargs.pop("torch_dtype", None)
        device: Union[str, torch.device, None] = kwargs.pop("device", None)
        strict: bool = bool(kwargs.pop("strict", False))
        raw = load_safetensors(weights_path)
        # Re-key the diffusers -> local naming.  When the file
        # is already in local naming the rename is a no-op.
        key_map = _materialise_per_block_map(config.num_layers)
        # Instantiate the adapter and apply the state dict.
        adapter = cls(config)
        load_state_dict_with_renames(
            adapter, raw, key_map=key_map, strict=strict,
        )
        if torch_dtype is not None:
            adapter = adapter.to(torch_dtype)
        if device is not None:
            adapter = adapter.to(device)
        adapter.eval()
        return adapter

    def save_pretrained(self, save_path: Union[str, Path]) -> Path:
        """Save the adapter's state dict to a safetensors file.

        Output matches what :meth:`from_pretrained` consumes:
        a flat state dict with the *local* naming scheme.
        Round-tripping ``adapter -> save -> load`` is a no-op.
        Parent directories of ``save_path`` are created on
        demand.
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # ``save_safetensors`` signature is
        # ``(state_dict, path)`` -- reverse of the
        # ``from_pretrained`` flow.
        save_safetensors(self.state_dict(), str(path))
        return path.resolve()

    def extra_repr(self) -> str:
        """Compact repr for ``print(adapter)``."""
        return (
            f"num_layers={self.config.num_layers}, "
            f"hint_channels={self.config.hint_channels}, "
            f"block_out_channels={self.config.block_out_channels}"
        )


# Smoke test

if __name__ == "__main__":
    # Minimal import-path check.  No unit test here -- the goal
    # is to confirm that the module can be imported and that the
    # public surface is wired up.  Run with
    # ``python -m papers.adapters.controlnet`` from the project
    # root, or any other entry point that has the ``core`` and
    # ``papers`` packages on ``sys.path``.
    print("[controlnet] module path: papers.adapters.controlnet")
    print(f"[controlnet] public surface: {sorted(__all__)}")
    print(f"[controlnet] CONTROLNET_KEY_MAP entries: "
          f"{len(CONTROLNET_KEY_MAP)}")
    print("[controlnet] smoke test done.")
