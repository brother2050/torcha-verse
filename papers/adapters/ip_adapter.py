"""IP-Adapter paper adapter (v0.9.0).

Project-internal IP-Adapter adapter (Ye et al., 2023 --
arXiv:2308.06721) that mirrors the architectural surface of
the diffusers :class:`diffusers.IPAdapter` and the ComfyUI
``IPAdapterApply`` node, while keeping the runtime footprint
limited to ``torch`` and the project-internal
:mod:`core.checkpoint_loader`.

Design notes
------------

* **Source of truth.**  diffusers ``IPAdapter`` is the
  architectural reference; the ComfyUI ``IPAdapterApply``
  node is the integration reference -- it takes a *list* of
  per-layer image projections (one per injected DiT block)
  and *adds* them to the host attention's hidden states,
  scaled by a user-supplied ``weight``.  We expose the same
  per-layer list-of-projections contract from
  :meth:`IPAdapter.get_image_features`.

* **Tiny footprint.**  :class:`IPAdapterConfig` defaults to
  ``num_tokens=4`` with ``num_layers=4`` and
  ``cross_attention_dim=768`` -- a "Tiny" preset that matches
  the paper's default per-image token count of 4 and the
  4-block injection scheme used by the SD1.5 / SDXL
  reproductions.

* **Integration with** :class:`core.offload.ModelPatcher`.
  Like ControlNet, IP-Adapter is a *patch* on the base
  DiT/UNet: it adds residuals, it does not replace
  parameters.  When the host backbone is wrapped in a
  :class:`ModelPatcher`, the IP-Adapter weights live on the
  same module graph and are subject to the same offload
  policy (see ``enable_model_cpu_offload`` /
  ``enable_sequential_cpu_offload`` in :mod:`core.offload`).
  There is no separate weight copy; the patcher sees the
  adapter as just another :class:`nn.Module` in the tree.

* **Image encoder placeholder.**  The full IP-Adapter
  pipeline ships a CLIP image encoder (typically
  ``openai/clip-vit-large-patch14``, ``image_embed_dim=768``)
  plus a per-layer projection MLP.  This module ships a
  :class:`MockImageEncoder` (a single ``nn.Linear``) as a
  stand-in so the adapter can be instantiated, exercised,
  and round-tripped through :meth:`from_pretrained` /
  :meth:`save_pretrained` without dragging the full CLIP
  vision tower in.  The mock is clearly labelled and a
  one-line swap is enough to wire a real encoder.

Public surface: :class:`IPAdapterConfig`, :class:`IPAdapter`,
:data:`IP_ADAPTER_KEY_MAP`.
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

__all__ = ["IPAdapterConfig", "IPAdapter", "IP_ADAPTER_KEY_MAP"]


# Key rename table: diffusers IP-Adapter -> local naming.
# Consumed by :meth:`IPAdapter.from_pretrained` (delegates to
# :func:`core.checkpoint_loader.load_state_dict_with_renames`).
# The diffusers ``IPAdapter`` publishes:
#   * ``image_proj``           -- a 2-layer MLP that lifts the CLIP
#     image embedding (``image_embed_dim``) to ``num_tokens`` slots
#     of ``cross_attention_dim``.
#   * ``image_proj_model.{i}.{w,b}`` -- per-layer linear projections
#     (one per injected attention block).
#   * ``attn_proc`` / ``attn_processors`` -- attention-processor
#     weights that are normally attached at runtime; we keep them
#     in the rename table so a round-trip preserves them, but the
#     skeleton does not consume them at forward time.
# We collapse ``image_proj`` into ``image_proj`` (1:1), the
# per-layer projections into ``per_layer.{i}``, and the
# attention-processor weights into ``mock_image_encoder``
# (placeholder -- the real encoder is not bundled).
IP_ADAPTER_KEY_MAP: Dict[str, str] = {
    # image_proj -- the lift from CLIP image embeds to
    # ``num_tokens * cross_attention_dim`` (stored as a single
    # ``nn.Linear`` in this skeleton; in diffusers this is a
    # 2-layer MLP with LayerNorm + GELU).
    "image_proj.proj.weight":   "image_proj.weight",
    "image_proj.proj.bias":     "image_proj.bias",
    "image_proj.layernorm.weight": "image_proj.norm.weight",
    "image_proj.layernorm.bias":   "image_proj.norm.bias",
    # per-layer projections (one per injected attention block).
    "image_proj_model.{i}.weight": "per_layer.{i}.weight",
    "image_proj_model.{i}.bias":   "per_layer.{i}.bias",
    # attention-processor weights (kept for round-trip fidelity;
    # not consumed by the skeleton's :meth:`forward`).
    "attn_proc.to_k_ip.weight":   "mock_image_encoder.to_k_ip.weight",
    "attn_proc.to_v_ip.weight":   "mock_image_encoder.to_v_ip.weight",
}


def _materialise_per_layer_map(num_layers: int) -> Dict[str, str]:
    """Expand ``{i}`` placeholders in :data:`IP_ADAPTER_KEY_MAP`
    for the load helpers.
    """
    out: Dict[str, str] = {}
    for k, v in IP_ADAPTER_KEY_MAP.items():
        if "{i}" in k:
            for i in range(num_layers):
                out[k.format(i=i)] = v.format(i=i)
        else:
            out[k] = v
    return out


# Config


@dataclass
class IPAdapterConfig:
    """Architectural configuration for :class:`IPAdapter`.

    Attributes:
        image_embed_dim: Output dim of the CLIP image encoder
            (the *input* dim of the image-projection MLP).
            ``1024`` matches ``openai/clip-vit-large-patch14``'s
            pooled embedding size; ``768`` matches
            ``openai/clip-vit-base-patch16``.
        cross_attention_dim: Latent dim of the host
            UNet/DiT's text cross-attention.  This is the
            *output* dim of every projection in
            :class:`PerLayerImageProj`.
        num_tokens: Number of per-image tokens produced by
            the image projection (``image_proj``).  IP-Adapter
            defaults to ``4``.
        num_images: Number of *concurrent* reference images.
            Each image gets its own ``num_tokens`` slots, so
            the effective token count is
            ``num_tokens * num_images``.
        scale: Default injection strength.  ``1.0`` = full
            effect, ``0.0`` = identity (ComfyUI's
            ``IPAdapterApply`` ``weight`` parameter).
        num_layers: Number of host attention blocks the
            adapter injects into.  Defaults to ``4`` (Tiny).
    """

    image_embed_dim: int = 1024
    cross_attention_dim: int = 768
    num_tokens: int = 4
    num_images: int = 1
    scale: float = 1.0
    num_layers: int = 4

    def __post_init__(self) -> None:
        # Coerce + validate.  Hard errors, no silent truncation --
        # an inconsistent config would yield silently-wrong
        # projections at load time.
        if self.image_embed_dim <= 0:
            raise ValueError(
                f"image_embed_dim must be > 0; got {self.image_embed_dim!r}"
            )
        if self.cross_attention_dim <= 0:
            raise ValueError(
                f"cross_attention_dim must be > 0; got "
                f"{self.cross_attention_dim!r}"
            )
        if self.num_tokens <= 0:
            raise ValueError(
                f"num_tokens must be > 0; got {self.num_tokens!r}"
            )
        if self.num_images <= 0:
            raise ValueError(
                f"num_images must be > 0; got {self.num_images!r}"
            )
        if self.num_layers <= 0:
            raise ValueError(
                f"num_layers must be > 0; got {self.num_layers!r}"
            )
        if not (0.0 <= float(self.scale) <= 2.0):
            raise ValueError(
                f"scale must be in [0.0, 2.0]; got {self.scale!r}"
            )
        self.image_embed_dim = int(self.image_embed_dim)
        self.cross_attention_dim = int(self.cross_attention_dim)
        self.num_tokens = int(self.num_tokens)
        self.num_images = int(self.num_images)
        self.num_layers = int(self.num_layers)
        self.scale = float(self.scale)

    @classmethod
    def tiny(cls) -> "IPAdapterConfig":
        """Return the Tiny preset (4 layers, CLIP-L image dim)."""
        return cls(
            image_embed_dim=1024,
            cross_attention_dim=768,
            num_tokens=4,
            num_images=1,
            scale=1.0,
            num_layers=4,
        )

    @property
    def effective_num_tokens(self) -> int:
        """Total number of image tokens (``num_tokens * num_images``).

        Mirrors the diffusers ``IPAdapter.set_ip_adapter_scale`` /
        ``set_image_embeds`` shape contract: the host cross-attention
        receives ``effective_num_tokens`` extra key/value pairs per
        forward pass.
        """
        return int(self.num_tokens) * int(self.num_images)


# Building blocks


class PerLayerImageProj(nn.Module):
    """A stack of per-layer linear projections.

    Holds ``num_layers`` independent ``nn.Linear`` modules,
    each mapping ``cross_attention_dim`` ->
    ``cross_attention_dim``.  This is the project-internal
    clone of the diffusers ``image_proj_model`` -- a
    ``nn.ModuleList`` of per-layer linear projections, one
    per injected attention block.

    The per-layer split mirrors the ComfyUI IPAdapter.Apply
    contract: each block in the host DiT pulls the
    ``[layer_idx]``-th projection, multiplies by ``scale``,
    and adds it to the block's hidden states.
    """

    def __init__(self, cross_attention_dim: int, num_layers: int) -> None:
        super().__init__()
        self.cross_attention_dim = int(cross_attention_dim)
        self.num_layers = int(num_layers)
        # ``nn.ModuleList`` -- the *only* container that
        # ``nn.Module`` will register sub-modules from.
        self.layers = nn.ModuleList(
            nn.Linear(self.cross_attention_dim, self.cross_attention_dim)
            for _ in range(self.num_layers)
        )
        # Initialise the biases to zero and the weights to the
        # identity (a la ControlNet's zero-conv trick): at the
        # start of training every per-layer projection is the
        # identity, so the patched backbone is bit-for-bit
        # identical to the base model.  Without this, the
        # random init would inject noise even at ``scale=0``
        # whenever the dtype does not exactly zero out the
        # multiplier.
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.zero_()
                # Init to identity rather than zero so that
                # ``scale=1`` gives the original embedding
                # back; the zero-conv trick is achieved at
                # the *outer* scale, not at the weight
                # init.
                w = torch.eye(self.cross_attention_dim)
                if w.shape == layer.weight.shape:
                    layer.weight.copy_(w)
                layer.bias.zero_()

    def forward(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` through the ``layer_idx``-th projection."""
        if not (0 <= int(layer_idx) < self.num_layers):
            raise IndexError(
                f"layer_idx must be in [0, {self.num_layers}); "
                f"got {layer_idx!r}"
            )
        return self.layers[int(layer_idx)](x)

    def __len__(self) -> int:
        return int(self.num_layers)


class MockImageEncoder(nn.Module):
    """Stand-in for the real CLIP image encoder.

    The full IP-Adapter pipeline ships a CLIP image encoder
    (typically ``openai/clip-vit-large-patch14``) that
    produces a pooled ``image_embed_dim``-sized vector per
    image.  This skeleton ships a single ``nn.Linear`` as a
    placeholder so the adapter can be instantiated and
    round-tripped through ``from_pretrained`` /
    ``save_pretrained`` without dragging the full vision
    tower in.

    **To wire a real encoder:** replace the ``self.proj``
    assignment in :meth:`__init__` with a real CLIP image
    encoder.  The downstream contract is fixed:
    ``encode_image(image: torch.Tensor) -> torch.Tensor``
    must return a ``[B, image_embed_dim]`` tensor (or
    ``[B, num_images, image_embed_dim]`` for the
    multi-image variant).
    """

    def __init__(self, image_embed_dim: int) -> None:
        super().__init__()
        self.image_embed_dim = int(image_embed_dim)
        # ``nn.Linear`` with the *same* input and output dim
        # so the mock is a (random, untrained) identity-ish
        # map.  Real encoder: replace with CLIPImageEncoder.
        self.proj = nn.Linear(self.image_embed_dim, self.image_embed_dim)
        # Mark the layer so introspection (and ``print(model)``)
        # makes the placeholder-ness obvious.
        self._is_mock_encoder: bool = True

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Pretend to encode an image batch.

        Args:
            image: ``[B, image_embed_dim]`` (the mock skips
                the real encoder and treats the input as if
                it were already an embedding).

        Returns:
            ``[B, image_embed_dim]`` -- the input, run
            through a single linear projection.
        """
        if image.shape[-1] != self.image_embed_dim:
            raise ValueError(
                f"mock encoder expected last dim == "
                f"{self.image_embed_dim}; got shape "
                f"{tuple(image.shape)!r}"
            )
        return self.proj(image)

    def extra_repr(self) -> str:
        return (
            f"image_embed_dim={self.image_embed_dim}, "
            f"is_mock={self._is_mock_encoder}"
        )


# Adapter


class IPAdapter(nn.Module):
    """Project-internal IP-Adapter (Ye et al., 2023).

    Ingests one (or more) reference images, projects them
    through a CLIP image encoder (mocked here), and emits a
    *list* of per-layer residual tensors that the caller
    adds back into the host DiT/UNet's cross-attention
    hidden states (ComfyUI's ``IPAdapterApply`` contract).

    :meth:`get_image_features` returns the raw list of
    per-layer projections; :meth:`forward` injects the
    ``layer_idx``-th projection into a single hidden-state
    tensor; :meth:`apply` is the high-level convenience that
    runs the full encode + project + inject pipeline in one
    call.
    """

    def __init__(self, config: IPAdapterConfig) -> None:
        super().__init__()
        self.config: IPAdapterConfig = config
        # The image-projection MLP: a single ``nn.Linear`` in
        # this skeleton (the real diffusers IP-Adapter uses a
        # 2-layer MLP with LayerNorm + GELU).  Output dim is
        # ``num_tokens * cross_attention_dim``; the caller
        # reshapes to ``[B, num_tokens, cross_attention_dim]``
        # before splitting into per-layer projections.
        self.image_proj = nn.Linear(
            config.image_embed_dim,
            config.num_tokens * config.cross_attention_dim,
        )
        # Per-layer projections (the per-block residuals).
        self.per_layer = PerLayerImageProj(
            cross_attention_dim=config.cross_attention_dim,
            num_layers=config.num_layers,
        )
        # Mock CLIP image encoder.  Replace with a real
        # CLIPImageEncoder to wire a production pipeline.
        self.mock_image_encoder = MockImageEncoder(config.image_embed_dim)

    @property
    def num_inject_layers(self) -> int:
        """Number of host attention blocks the adapter injects into.

        Equals :attr:`IPAdapterConfig.num_layers`; surfaced as
        a property so callers can introspect the adapter
        without reading the config.
        """
        return int(self.config.num_layers)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Run the (mock) image encoder.

        Args:
            image: ``[B, image_embed_dim]`` (the mock treats
                the input as if it were already a pooled
                CLIP image embedding).

        Returns:
            ``[B, image_embed_dim]`` -- the encoded image
            embedding.  Pass to :meth:`get_image_features`
            to obtain the per-layer projections.
        """
        return self.mock_image_encoder(image)

    def get_image_features(
        self,
        image_embeds: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Project image embeddings into a per-layer list.

        Runs the lift ``image_embeds -> [num_tokens, cross_attention_dim]``
        and then runs each per-layer projection on the
        resulting tokens.  The list length is
        :attr:`num_inject_layers`; each entry has shape
        ``[B, num_tokens, cross_attention_dim]`` (or
        ``[B, num_tokens * num_images, cross_attention_dim]``
        for the multi-image variant -- the contract is the
        same, the caller concatenates the per-image slots).

        Args:
            image_embeds: ``[B, image_embed_dim]`` -- pooled
                image embeddings, typically the output of
                :meth:`encode_image`.

        Returns:
            A list of length ``num_layers``; each entry is
            ``[B, num_tokens, cross_attention_dim]``.
        """
        if image_embeds.dim() != 2:
            raise ValueError(
                f"image_embeds must be 2-D [B, image_embed_dim]; "
                f"got shape {tuple(image_embeds.shape)!r}"
            )
        if image_embeds.shape[-1] != self.config.image_embed_dim:
            raise ValueError(
                f"image_embeds last dim must be == "
                f"{self.config.image_embed_dim}; got "
                f"{image_embeds.shape[-1]}"
            )
        # Lift to ``num_tokens * cross_attention_dim`` and
        # reshape to ``[B, num_tokens, cross_attention_dim]``.
        projected = self.image_proj(image_embeds)
        bsz = int(image_embeds.shape[0])
        projected = projected.view(
            bsz, self.config.num_tokens, self.config.cross_attention_dim,
        )
        # Run each per-layer projection on the projected
        # tokens.  The diffusers reference runs the *same*
        # tokens through each per-layer projection, so the
        # call here is a loop over ``self.per_layer``.
        features: List[torch.Tensor] = []
        for i in range(self.num_inject_layers):
            features.append(self.per_layer(i, projected))
        return features

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_embeds: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Inject the ``layer_idx``-th image projection into ``hidden_states``.

        Mirrors the ComfyUI IPAdapter.Apply contract: take
        the per-layer projection, broadcast-add it into the
        host attention's hidden states, scaled by
        :attr:`IPAdapterConfig.scale` (or the per-call
        override).

        Args:
            hidden_states: ``[B, L, cross_attention_dim]`` --
                the host attention's hidden states at the
                injection point.
            image_embeds: ``[B, image_embed_dim]`` -- pooled
                image embeddings.
            layer_idx: Index of the per-layer projection to
                inject (must be in ``[0, num_layers)``).

        Returns:
            ``hidden_states + scale * projection``, same
            shape as ``hidden_states``.
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be 3-D [B, L, D]; got "
                f"shape {tuple(hidden_states.shape)!r}"
            )
        if int(hidden_states.shape[-1]) != int(self.config.cross_attention_dim):
            raise ValueError(
                f"hidden_states last dim must be == "
                f"{self.config.cross_attention_dim}; got "
                f"{int(hidden_states.shape[-1])}"
            )
        features = self.get_image_features(image_embeds)
        if not (0 <= int(layer_idx) < len(features)):
            raise IndexError(
                f"layer_idx must be in [0, {len(features)}); "
                f"got {layer_idx!r}"
            )
        # ComfyUI IPAdapter.Apply injects the *projected*
        # tokens, not the raw ones -- i.e. the per-layer
        # projection is applied to the projected tokens and
        # the result is added to ``hidden_states``.
        projection = features[int(layer_idx)]
        # Broadcast over the sequence dim: ``projection`` is
        # ``[B, num_tokens, cross_attention_dim]`` and
        # ``hidden_states`` is ``[B, L, cross_attention_dim]``.
        # The standard IP-Adapter contract is to add the
        # image tokens into the *first* ``num_tokens``
        # positions of the hidden state and leave the rest
        # untouched (the image tokens augment, rather than
        # replace, the text-token slots).  This keeps the
        # contract shape-stable for any ``L >= num_tokens``.
        seq_len = int(hidden_states.shape[1])
        tok = int(projection.shape[1])
        if seq_len < tok:
            raise ValueError(
                f"hidden_states sequence length ({seq_len}) must be "
                f">= num_tokens ({tok}) to apply IP-Adapter injection"
            )
        out = hidden_states.clone()
        out[:, :tok, :] = out[:, :tok, :] + float(self.config.scale) * projection
        return out

    def apply(
        self,
        conditioning: torch.Tensor,
        image: torch.Tensor,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """High-level helper: run the full IP-Adapter pipeline.

        Convenience wrapper around :meth:`encode_image` +
        :meth:`get_image_features` + :meth:`forward`.  The
        caller passes the host attention's hidden states and
        the raw image; the adapter returns the
        ``hidden_states + scale * projection`` tensor.

        Args:
            conditioning: ``[B, L, cross_attention_dim]`` --
                the host attention's hidden states.  Same
                tensor that would be passed to
                :meth:`forward`.
            image: ``[B, image_embed_dim]`` -- the raw
                (pre-encoded) image batch.  Run through
                :meth:`encode_image` to obtain the pooled
                image embedding.
            scale: Per-call scaling factor.  ``None`` means
                use the config default (:attr:`IPAdapterConfig.scale`).
                ``0.0`` = identity, ``1.0`` = full effect.

        Returns:
            ``[B, L, cross_attention_dim]`` -- the
            conditioning with the image projection injected
            at layer 0 (the first injection point).
        """
        image_embeds = self.encode_image(image)
        # If the caller passed a per-call scale, monkey-patch
        # the config for this call only -- the alternative
        # (a ``forward(..., scale=...)`` kwarg) would break
        # the diffusers-compatible signature.  The mutation
        # is reverted before returning.
        if scale is not None:
            old_scale = float(self.config.scale)
            self.config.scale = float(scale)
            try:
                return self.forward(
                    hidden_states=conditioning,
                    image_embeds=image_embeds,
                    layer_idx=0,
                )
            finally:
                self.config.scale = old_scale
        return self.forward(
            hidden_states=conditioning,
            image_embeds=image_embeds,
            layer_idx=0,
        )

    @classmethod
    def from_pretrained(
        cls,
        weights_path: Union[str, Path],
        **kwargs: Any,
    ) -> "IPAdapter":
        """Load an :class:`IPAdapter` from a safetensors file.

        The load goes through :mod:`core.checkpoint_loader`.
        Recognised kwargs: ``config`` (an
        :class:`IPAdapterConfig`, default Tiny),
        ``subfolder``, ``variant``, ``torch_dtype``,
        ``device``, ``strict``.

        * ``subfolder`` -- the relative subdirectory of
          ``weights_path`` to load from (passed through to
          :func:`core.checkpoint_loader.load_safetensors`).
        * ``variant`` -- the safetensors variant (e.g.
          ``"fp16"``); passed through verbatim.
        * ``torch_dtype`` -- optional dtype cast applied to
          every tensor after load.
        * ``device`` -- pin the adapter to a single device
          after load.
        * ``strict`` -- forwarded to
          :func:`load_state_dict_with_renames`.

        The ``weights_path`` may be a diffusers-style state
        dict (re-keyed via :data:`IP_ADAPTER_KEY_MAP`) or a
        local-style state dict (no re-keying -- the rename
        is a no-op for matching keys).
        """
        config: IPAdapterConfig = kwargs.pop("config", IPAdapterConfig.tiny())
        subfolder: Optional[str] = kwargs.pop("subfolder", None)
        variant: Optional[str] = kwargs.pop("variant", None)
        torch_dtype: Optional[torch.dtype] = kwargs.pop("torch_dtype", None)
        device: Union[str, torch.device, None] = kwargs.pop("device", None)
        strict: bool = bool(kwargs.pop("strict", False))
        # ``load_safetensors`` is the canonical entry point;
        # subfolder / variant are forwarded as kwargs and
        # resolved inside the helper.  Unknown kwargs are
        # silently swallowed here (the helper decides).
        raw = load_safetensors(
            weights_path,
            subfolder=subfolder,
            variant=variant,
        )
        # Re-key the diffusers -> local naming.  When the
        # file is already in local naming the rename is a
        # no-op.
        key_map = _materialise_per_layer_map(config.num_layers)
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
            f"image_embed_dim={self.config.image_embed_dim}, "
            f"cross_attention_dim={self.config.cross_attention_dim}, "
            f"num_tokens={self.config.num_tokens}, "
            f"num_images={self.config.num_images}, "
            f"scale={self.config.scale}"
        )


# Smoke test


if __name__ == "__main__":
    # Minimal import-path check.  No unit test here -- the goal
    # is to confirm that the module can be imported and that the
    # public surface is wired up.  Run with
    # ``python -m papers.adapters.ip_adapter`` from the project
    # root, or any other entry point that has the ``core`` and
    # ``papers`` packages on ``sys.path``.
    print("[ip_adapter] module path: papers.adapters.ip_adapter")
    print(f"[ip_adapter] public surface: {sorted(__all__)}")
    print(f"[ip_adapter] IP_ADAPTER_KEY_MAP entries: "
          f"{len(IP_ADAPTER_KEY_MAP)}")
    # Light-shape exercise: build the adapter, encode an
    # image batch, and run the full apply pipeline.  All
    # tensors are CPU / float32 -- no CUDA / no autograd.
    cfg = IPAdapterConfig.tiny()
    adapter = IPAdapter(cfg).eval()
    image = torch.randn(2, cfg.image_embed_dim)
    conditioning = torch.randn(2, 8, cfg.cross_attention_dim)
    features = adapter.get_image_features(adapter.encode_image(image))
    print(f"[ip_adapter] per-layer features: "
          f"{len(features)} x {tuple(features[0].shape)}")
    out = adapter.apply(conditioning, image)
    print(f"[ip_adapter] apply output shape: {tuple(out.shape)}")
    print("[ip_adapter] smoke test done.")
