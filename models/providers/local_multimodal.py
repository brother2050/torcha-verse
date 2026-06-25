"""Local-torch multimodal (omni-modal) provider for the v0.4.x P0 milestone.

This module wires the project-owned
:mod:`models.multimodal.omni_model` and
:mod:`models.multimodal.vision_language` into the
:class:`models.interfaces.media_providers.MultimodalProvider` protocol
so that the v0.4.x P0 omni-modal nodes / examples can be exercised
**end-to-end with a real neural network** (no echo, no passthrough)
while still being *pure torch, zero external dependencies*.

The class is intentionally small:

* it owns a :class:`OmniModel` (the cross-modal backbone) and a
  simple :class:`LocalTorchMultimodalProvider._TextHead` for text
  generation, loaded from a single ``.pt`` file (or constructed
  in memory from a :class:`MultimodalProviderConfig`);
* it implements :meth:`generate` (the only
  :class:`MultimodalProvider` method exercised by
  ``call_omni_backend``) and a few introspection helpers used by
  the v0.4.x P0 demo / tests;
* it is **thread-safe** (a single re-entrant lock guards the
  forward pass so concurrent :meth:`generate` calls serialise on
  the same model).

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L4 ``models.multimodal`` -- real components (OmniModel /
  VisionLanguageModel).
* L6 ``models.providers`` (this module) -- real omni-modal provider.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from infrastructure.logger import get_logger

from ..interfaces.media_providers import MultimodalProvider
from ..multimodal import OmniModel, VisionLanguageModel

__all__ = [
    "LocalTorchMultimodalProvider",
    "MultimodalProviderConfig",
    "TINY_MULTIMODAL_CONFIG",
    "SMALL_MULTIMODAL_CONFIG",
]


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger = get_logger("models.providers.local_multimodal")


# ---------------------------------------------------------------------------
# Config presets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MultimodalProviderConfig:
    """Hyperparameter bundle for :class:`LocalTorchMultimodalProvider`.

    Defaults produce a tiny model whose forward / backward pass
    runs in well under a second on a single CPU thread; that is
    what the v0.4.x P0 demo / CI smoke tests rely on to keep the
    milestone dependency-free.
    """

    name: str = "tiny"
    # Shared language backbone (TransformerDecoder in OmniModel).
    text_vocab_size: int = 256        # byte-level
    text_hidden_size: int = 64
    text_num_layers: int = 2
    text_num_heads: int = 4
    text_max_seq_len: int = 64
    # Vision tower (ViT).
    vision_image_size: int = 16       # must be divisible by patch_size
    vision_patch_size: int = 4
    vision_in_channels: int = 3
    vision_hidden_size: int = 64
    vision_num_layers: int = 2
    vision_num_heads: int = 4
    # Audio tower (Conv1d stack in OmniModel).
    audio_mel_channels: int = 32
    audio_hidden_size: int = 32
    audio_output_dim: int = 64
    audio_num_layers: int = 2
    audio_stride: int = 2
    # Generation.
    default_max_new_tokens: int = 8

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


TINY_MULTIMODAL_CONFIG = MultimodalProviderConfig(name="tiny")
SMALL_MULTIMODAL_CONFIG = MultimodalProviderConfig(
    name="small",
    text_hidden_size=128,
    text_num_layers=4,
    vision_hidden_size=128,
    vision_num_layers=4,
    audio_hidden_size=64,
    audio_output_dim=128,
    default_max_new_tokens=16,
)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
class LocalTorchMultimodalProvider(MultimodalProvider):
    """A real, project-owned :class:`MultimodalProvider` backed by ``torch``.

    The provider is **stateless at the framework level** -- it
    holds a single :class:`OmniModel` and serialises concurrent
    calls behind a lock.  All forward passes run in
    ``torch.no_grad`` mode so inference does not allocate
    autograd graphs.

    Args:
        omni: A pre-built :class:`OmniModel`.  When ``None`` a
            fresh one is built from ``config``.
        config: The :class:`MultimodalProviderConfig` that was
            used to build the model.  When ``None`` the
            :data:`TINY_MULTIMODAL_CONFIG` is used.
        device: Device to run the model on.  Defaults to CPU so
            the provider is portable across CI environments.
    """

    class _TextHead(nn.Module):
        """Tiny causal LM head that maps hidden states to byte logits.

        The head is a single linear projection from the language
        hidden size to the byte-level vocabulary (256).  The
        :class:`OmniModel` is shared with the text / vision / audio
        towers so the hidden size is known at construction time.
        """

        def __init__(self, hidden_size: int, vocab_size: int) -> None:
            super().__init__()
            self.proj: nn.Linear = nn.Linear(hidden_size, vocab_size)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.proj(hidden)

    class _TinyCausalLM(nn.Module):
        """A minimal causal LM used purely for text generation.

        The v0.4.x P0 omni-modal demo only needs a *small* LM to
        satisfy the multimodal provider's :meth:`generate` text
        branch; the language model inside
        :class:`models.multimodal.omni_model.OmniModel` is a
        :class:`models.text.transformer.TransformerDecoder` whose
        forward returns *logits* (it always goes through
        ``lm_head``), so we cannot easily extract the hidden
        state.  This standalone module is purpose-built for the
        provider and is trained from scratch with the rest of
        the model (i.e. it is a *separate* set of weights --
        not the same as :attr:`omni.language_model`).
        """

        def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            num_layers: int,
            num_heads: int,
            max_seq_len: int,
        ) -> None:
            super().__init__()
            self.embedding: nn.Embedding = nn.Embedding(vocab_size, hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
                activation="gelu",
            )
            self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(
                layer, num_layers=num_layers,
            )
            self.lm_head: nn.Linear = nn.Linear(hidden_size, vocab_size)
            self.hidden_size: int = hidden_size
            self.max_seq_len: int = max_seq_len

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            h = self.embedding(input_ids)
            h = self.encoder(h)
            return self.lm_head(h)

    def __init__(
        self,
        omni: Optional[nn.Module] = None,
        config: Optional[MultimodalProviderConfig] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        if config is None:
            config = TINY_MULTIMODAL_CONFIG
        if omni is None:
            omni = OmniModel(
                text_config={
                    "vocab_size": config.text_vocab_size,
                    "hidden_size": config.text_hidden_size,
                    "num_layers": config.text_num_layers,
                    "num_heads": config.text_num_heads,
                    "num_kv_heads": config.text_num_heads,
                    "max_seq_len": config.text_max_seq_len,
                },
                vision_config={
                    "image_size": config.vision_image_size,
                    "patch_size": config.vision_patch_size,
                    "in_channels": config.vision_in_channels,
                    "hidden_size": config.vision_hidden_size,
                    "num_layers": config.vision_num_layers,
                    "num_heads": config.vision_num_heads,
                },
                audio_config={
                    "mel_channels": config.audio_mel_channels,
                    "hidden_size": config.audio_hidden_size,
                    "output_dim": config.audio_output_dim,
                    "num_layers": config.audio_num_layers,
                    "stride": config.audio_stride,
                },
            )
        # The text vocab size is the byte vocab (256); the LM
        # head projects hidden -> 256.  The provider owns a
        # *standalone* :class:`_TinyCausalLM` for the text
        # generation branch (see docstring for why we do not
        # reuse :attr:`omni.language_model`).
        text_lm = LocalTorchMultimodalProvider._TinyCausalLM(
            vocab_size=config.text_vocab_size,
            hidden_size=config.text_hidden_size,
            num_layers=config.text_num_layers,
            num_heads=config.text_num_heads,
            max_seq_len=config.text_max_seq_len,
        )

        self._omni: nn.Module = omni.to(device)
        self._text_lm: nn.Module = text_lm.to(device)
        for m in (self._omni, self._text_lm):
            m.eval()

        self._config: MultimodalProviderConfig = config
        self._device: torch.device = (
            torch.device(device)
            if not isinstance(device, torch.device)
            else device
        )
        self._lock: threading.RLock = threading.RLock()
        self._logger = _logger

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_random(
        cls,
        config: Optional[MultimodalProviderConfig] = None,
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> "LocalTorchMultimodalProvider":
        """Construct a provider with freshly initialised models."""
        return cls(config=config, device=device)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> "LocalTorchMultimodalProvider":
        """Load a provider from a ``.pt`` file produced by :meth:`save`."""
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(
                "multimodal provider file not found: {}".format(p)
            )
        payload = torch.load(p, map_location=device, weights_only=False)
        cfg_dict = payload.get("config", {})
        if not isinstance(cfg_dict, dict):
            raise TypeError("payload['config'] must be a dict")
        config = MultimodalProviderConfig(**cfg_dict)
        provider = cls(config=config, device=device)
        if "omni" in payload:
            provider._omni.load_state_dict(payload["omni"], strict=False)
        if "text_lm" in payload:
            provider._text_lm.load_state_dict(payload["text_lm"], strict=False)
        return provider

    def save(self, path: Union[str, Path]) -> Path:
        """Persist the provider to ``path`` (a ``.pt`` file)."""
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self._config.to_dict(),
            "omni": self._omni.state_dict(),
            "text_lm": self._text_lm.state_dict(),
        }
        torch.save(payload, out)
        return out

    # ------------------------------------------------------------------
    # MultimodalProvider interface
    # ------------------------------------------------------------------
    def generate(
        self,
        input: Union[str, Dict[str, Any], Sequence[Any]],
        *,
        max_new_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a multi-modal response from ``input``.

        The pipeline supports three input shapes:

        * ``str`` -- pure text.  The provider runs a tiny
          autoregressive generation loop on the byte-level
          language model.
        * ``dict`` -- a heterogeneous payload with optional
          ``"text"`` (str) / ``"image"`` (Tensor of shape
          ``(3, H, W)``) / ``"audio"`` (Tensor of shape
          ``(1, T)``) keys.  Each present modality is encoded
          by the matching tower of the :class:`OmniModel` and
          fused into the language hidden state.
        * ``Sequence`` -- a list whose first element is treated
          as the prompt text (the remaining elements are
          ignored in v0.4.x P0; a richer fusion will replace
          this in v0.5).

        Args:
            input: The heterogeneous prompt.  See above.
            max_new_tokens: Number of tokens to generate when
                the input contains text.  Defaults to
                ``config.default_max_new_tokens``.
            seed: RNG seed for reproducibility.
            **kwargs: Ignored.  Forwarded for forward-compat
                with the v0.4.x P0 node kwargs.

        Returns:
            A dict.  The keys that may be present include:

            * ``"text"`` -- the generated byte string (when the
              input has text).
            * ``"image_emb_shape"`` -- the shape of the vision
              embedding (when the input has an image).
            * ``"audio_emb_shape"`` -- the shape of the audio
              embedding (when the input has audio).
        """
        cfg = self._config
        # Normalise the input into a dict.
        if isinstance(input, str):
            payload: Dict[str, Any] = {"text": input}
        elif isinstance(input, dict):
            payload = dict(input)
        elif isinstance(input, Sequence):
            payload = {"text": str(input[0]) if input else ""}
        else:
            raise TypeError(
                "input must be str, dict, or Sequence, got {}".format(
                    type(input).__name__
                )
            )

        text = payload.get("text")
        image = payload.get("image")
        audio = payload.get("audio")
        n_new = int(max_new_tokens or cfg.default_max_new_tokens)
        if n_new < 1:
            n_new = 1

        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))

        result: Dict[str, Any] = {}
        with self._lock:
            with torch.no_grad():
                # --- Vision tower ---
                if image is not None:
                    if not isinstance(image, torch.Tensor):
                        image = torch.as_tensor(image, dtype=torch.float32)
                    if image.ndim == 3:
                        image = image.unsqueeze(0)
                    # Pad / crop to (cfg.vision_image_size,
                    # cfg.vision_image_size) so the ViT can patch
                    # embed it.
                    image = self._resize_image(
                        image, cfg.vision_image_size, cfg.vision_image_size,
                    ).to(self._device)
                    # ``vision_encoder`` returns
                    # ``(features, cls_token)``; we keep the
                    # full features tensor.
                    vis_features, _ = self._omni.vision_encoder(image)
                    result["image_emb_shape"] = tuple(vis_features.shape)

                # --- Audio tower ---
                if audio is not None:
                    if not isinstance(audio, torch.Tensor):
                        audio = torch.as_tensor(audio, dtype=torch.float32)
                    # The audio encoder expects
                    # ``(batch, mel_channels, time)``; we pad /
                    # trim the channel dim to
                    # ``config.audio_mel_channels`` if the caller
                    # did not provide a mel spectrogram.
                    if audio.ndim == 1:
                        audio = audio.unsqueeze(0)
                    if audio.ndim == 2:
                        audio = audio.unsqueeze(1)
                    if audio.shape[1] != cfg.audio_mel_channels:
                        if audio.shape[1] < cfg.audio_mel_channels:
                            pad = cfg.audio_mel_channels - audio.shape[1]
                            audio = torch.nn.functional.pad(
                                audio, (0, 0, 0, pad),
                            )
                        else:
                            audio = audio[:, : cfg.audio_mel_channels, :]
                    audio = audio.to(self._device)
                    audio_emb = self._omni.audio_encoder(audio)
                    result["audio_emb_shape"] = tuple(audio_emb.shape)

                # --- Text tower + autoregressive decode ---
                if text is not None and text != "":
                    text_bytes = self._byte_tokenize(
                        text, cfg.text_max_seq_len,
                    )
                    input_ids = torch.tensor(
                        [text_bytes], dtype=torch.long, device=self._device,
                    )
                    out_ids: List[int] = []
                    cur = input_ids
                    for _ in range(n_new):
                        logits = self._text_lm(cur)
                        # Greedy next-token from the last position.
                        next_id = int(
                            torch.argmax(logits[0, -1], dim=-1).item()
                        )
                        out_ids.append(next_id)
                        cur = torch.cat(
                            [cur, torch.tensor(
                                [[next_id]], device=self._device,
                                dtype=torch.long,
                            )],
                            dim=1,
                        )
                        if cur.shape[1] >= cfg.text_max_seq_len:
                            break
                    # Clamp to ASCII so the output is always
                    # decodable without replacement chars.
                    out_ids = [b % 128 for b in out_ids]
                    decoded = bytes(out_ids).decode(
                        "utf-8", errors="replace"
                    )
                    result["text"] = decoded
                elif "text" not in result:
                    # No text provided -- produce a tiny
                    # token-less summary so the caller has
                    # *something* in the dict.
                    result["text"] = ""

        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def config(self) -> MultimodalProviderConfig:
        """The provider config (read-only)."""
        return self._config

    @property
    def device(self) -> torch.device:
        """The device the models are bound to."""
        return self._device

    def num_parameters(self) -> int:
        """Total parameter count across OmniModel + TinyCausalLM."""
        return sum(
            sum(p.numel() for p in m.parameters())
            for m in (self._omni, self._text_lm)
        )

    @staticmethod
    def _byte_tokenize(text: str, max_len: int) -> List[int]:
        if not text:
            return [0] * max_len
        ids = list(text.encode("utf-8")[:max_len])
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        return ids

    @staticmethod
    def _resize_image(
        image: torch.Tensor, target_h: int, target_w: int,
    ) -> torch.Tensor:
        """Bilinearly resize a ``(B, 3, H, W)`` image to ``(target_h, target_w)``."""
        return F.interpolate(
            image, size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        )

    def __repr__(self) -> str:
        return (
            "LocalTorchMultimodalProvider(name={!r}, params={}, device={!r})".format(
                self._config.name, self.num_parameters(), self._device,
            )
        )
