"""Tiny Transformer LM for the v0.4.0 P0 milestone (pure-torch, no
external dependencies).

This module defines a *configurable, lightweight* decoder-only
Transformer that wraps the project's existing
:class:`models.text.transformer.TransformerDecoder` and adds:

* a single-source-of-truth :class:`TinyTransformerConfig` dataclass
  that documents the **~10M parameter** "small" preset (6 layers,
  hidden_size 384, vocab 260) and supports the **~2M parameter**
  "tiny" preset for CI smoke tests;
* a **byte-level tokenizer** :class:`ByteTokenizer` with PAD/BOS/EOS
  ids 0/1/2, byte ids 3..258 (256 bytes), mask id 259;
* a serialisation format that round-trips a config + state-dict
  through ``torch.save`` (a single ``.pt`` file) so the model can be
  cached under :mod:`models.source` like any other model;
* :func:`load_tiny_transformer` and :func:`save_tiny_transformer`
  helpers used by the provider layer.

The whole module is **pure torch + standard library** -- no
``transformers``, no ``tokenizers``, no ``safetensors``.  This is
the constraint that makes the v0.4.0 P0 "real model" milestone
self-contained: the model is *built, trained, serialised and
served* with only the project's own code.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``models.providers`` (this module) -- project-owned model
  factory used to demonstrate end-to-end real-model coverage of
  the 30-node L4 capability layer.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from infrastructure.logger import get_logger

__all__ = [
    "ByteTokenizer",
    "TinyTransformerConfig",
    "SMALL_CONFIG",
    "TINY_CONFIG",
    "save_tiny_transformer",
    "load_tiny_transformer",
    "build_tiny_transformer",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Module-level logger.
_logger = get_logger("models.providers.tiny_transformer")

#: Reserved special-token ids in :class:`ByteTokenizer`.
_PAD_ID: int = 0
_BOS_ID: int = 1
_EOS_ID: int = 2

#: First byte id (after the three special tokens).  Bytes 0..255 are
#: shifted up by this offset.
_BYTE_OFFSET: int = 3

#: Number of distinct byte values (0..255).
_NUM_BYTES: int = 256

#: Mask id (used to signal an invalid / unknown byte).
_MASK_ID: int = _BYTE_OFFSET + _NUM_BYTES  # 259

#: Total vocabulary size for :class:`ByteTokenizer`.
#: 3 special + 256 bytes + 1 mask = 260.
DEFAULT_VOCAB_SIZE: int = _MASK_ID + 1  # 260


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TinyTransformerConfig:
    """Single-source-of-truth configuration for the tiny Transformer.

    Defaults are the **~10M parameter "small" preset** that v0.4.0 P0
    uses for end-to-end real-model coverage.  For CI smoke tests set
    :data:`TINY_CONFIG` instead.

    Attributes:
        vocab_size: Vocabulary size (must match the tokenizer).
        hidden_size: Model dimension.
        num_layers: Number of Transformer blocks.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (GQA).  When ``None``
            it defaults to ``num_heads`` (MHA mode).
        intermediate_size: MLP intermediate dimension.
        max_seq_len: Maximum sequence length for RoPE.
        rope_theta: RoPE base frequency.
        norm_type: ``"rmsnorm"`` (LLaMA-style) or ``"layernorm"``.
        activation: ``"swiglu"`` (LLaMA-style), ``"geglu"`` or
            ``"mlp"``.
        tie_word_embeddings: When ``True`` the LM head shares its
            weight with the input embedding (saves a lot of memory
            at the cost of slightly slower convergence).
        attention_dropout: Attention dropout probability.
        name: Human-readable name (used for logging / display only).
    """

    vocab_size: int = DEFAULT_VOCAB_SIZE
    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    num_kv_heads: Optional[int] = 6
    intermediate_size: int = 1024
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_type: str = "rmsnorm"
    activation: str = "swiglu"
    tie_word_embeddings: bool = True
    attention_dropout: float = 0.0
    name: str = "tiny-transformer-small"

    def as_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TinyTransformerConfig":
        """Reconstruct from a serialised dictionary (unknown keys ignored)."""
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_model_kwargs(self) -> Dict[str, Any]:
        """Return a kwargs dict suitable for
        :class:`models.text.transformer.TransformerDecoder`."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads if self.num_kv_heads is not None else self.num_heads,
            "intermediate_size": self.intermediate_size,
            "max_seq_len": self.max_seq_len,
            "rope_theta": self.rope_theta,
            "norm_type": self.norm_type,
            "activation": self.activation,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_dropout": self.attention_dropout,
        }

    def __repr__(self) -> str:
        return (
            "TinyTransformerConfig(name={!r}, hidden_size={}, num_layers={}, "
            "num_heads={}, num_kv_heads={}, intermediate_size={}, "
            "vocab_size={}, params~={}M)".format(
                self.name,
                self.hidden_size,
                self.num_layers,
                self.num_heads,
                self.num_kv_heads,
                self.intermediate_size,
                self.vocab_size,
                self.approx_params_m(),
            )
        )

    def approx_params_m(self) -> float:
        """Return a rough upper-bound parameter count in millions.

        The bound is the per-block cost (``12 * h^2 + 2 * h * i``)
        multiplied by ``num_layers`` plus the embedding / LM head
        (``vocab_size * h``).  Tied embeddings are not double-counted.
        Good enough for choosing a preset.
        """
        h = self.hidden_size
        i = self.intermediate_size
        n_kv = self.num_kv_heads if self.num_kv_heads is not None else self.num_heads
        # Attention: q proj (h*h) + k proj (h*head_dim*n_kv) + v proj
        # (h*head_dim*n_kv) + o proj (h*h).  head_dim == h / num_heads.
        head_dim = h // self.num_heads
        per_block = 4 * h * h + 2 * h * head_dim * n_kv + 2 * h * i
        embed = self.vocab_size * h
        # The LM head is tied to the embedding when
        # ``tie_word_embeddings`` is True -- do not double-count.
        total = per_block * self.num_layers + embed
        if not self.tie_word_embeddings:
            total += self.vocab_size * h
        return total / 1_000_000.0


#: The **~2M parameter** "tiny" preset used by CI smoke tests.
#: Trains in a few seconds on a single CPU thread.
TINY_CONFIG: TinyTransformerConfig = TinyTransformerConfig(
    hidden_size=128,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    intermediate_size=256,
    max_seq_len=256,
    tie_word_embeddings=True,
    name="tiny-transformer-tiny",
)

#: The **~10M parameter** "small" preset for the v0.4.0 P0 demo.
SMALL_CONFIG: TinyTransformerConfig = TinyTransformerConfig(
    hidden_size=384,
    num_layers=6,
    num_heads=6,
    num_kv_heads=6,
    intermediate_size=1024,
    max_seq_len=512,
    tie_word_embeddings=True,
    name="tiny-transformer-small",
)


# ---------------------------------------------------------------------------
# Byte-level tokenizer
# ---------------------------------------------------------------------------
class ByteTokenizer:
    """A minimal, dependency-free byte-level tokenizer.

    Maps every byte (0..255) to a token id in ``[3, 259)``.  Three
    special ids are reserved:

    * ``0`` -- PAD
    * ``1`` -- BOS
    * ``2`` -- EOS
    * ``259`` -- MASK (a single reserved id for future use)

    The class satisfies the :class:`training.dataset.BaseTokenizer`
    protocol (duck-typed) so it can be used anywhere a
    ``BaseTokenizer`` is accepted.

    The class is intentionally **stateless and immutable**; one
    instance can be shared across threads.  Decoding is the inverse
    of encoding; ids that fall outside ``[3, 259)`` are dropped
    (special tokens are not emitted as characters).
    """

    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE) -> None:
        if vocab_size < DEFAULT_VOCAB_SIZE:
            raise ValueError(
                "vocab_size must be at least {} (got {})".format(
                    DEFAULT_VOCAB_SIZE, vocab_size,
                )
            )
        self.vocab_size: int = int(vocab_size)
        self.pad_token_id: int = _PAD_ID
        self.bos_token_id: int = _BOS_ID
        self.eos_token_id: int = _EOS_ID
        self.mask_token_id: int = _MASK_ID

    # ------------------------------------------------------------------
    def encode(
        self, text: str, *, add_bos: bool = True, add_eos: bool = True
    ) -> List[int]:
        """Encode ``text`` to a list of token ids.

        Args:
            text: The input string.  Encoded as UTF-8 bytes; bytes
                that fall outside the supported range (i.e. id
                ``>= vocab_size``) are clamped to ``mask_token_id``.
            add_bos: Whether to prepend ``bos_token_id``.
            add_eos: Whether to append ``eos_token_id``.

        Returns:
            A list of non-negative token ids.
        """
        if text is None:
            text = ""
        raw = text.encode("utf-8", errors="replace")
        ids: List[int] = []
        for byte in raw:
            tid = byte + _BYTE_OFFSET
            if tid >= self.vocab_size:
                tid = self.mask_token_id
            ids.append(tid)
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str:
        """Decode a list / tensor of token ids back to a string.

        Args:
            ids: Iterable of token ids.
            skip_special: When ``True`` (default) special tokens
                (PAD / BOS / EOS / MASK) are dropped from the output.

        Returns:
            The decoded string.  Invalid bytes are replaced with
            U+FFFD (the standard UTF-8 replacement character).
        """
        out_bytes = bytearray()
        for raw in ids:
            i = int(raw)
            if skip_special and i in (_PAD_ID, _BOS_ID, _EOS_ID, _MASK_ID):
                continue
            if i < _BYTE_OFFSET or i >= _BYTE_OFFSET + _NUM_BYTES:
                # Unknown id (e.g. from a mismatched vocab).
                continue
            out_bytes.append(i - _BYTE_OFFSET)
        return out_bytes.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, int]:
        """Return a JSON-friendly snapshot of the tokeniser config."""
        return {
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "mask_token_id": self.mask_token_id,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, int]) -> "ByteTokenizer":
        """Reconstruct a :class:`ByteTokenizer` from a snapshot."""
        return cls(vocab_size=int(d.get("vocab_size", DEFAULT_VOCAB_SIZE)))

    def __repr__(self) -> str:
        return "ByteTokenizer(vocab_size={})".format(self.vocab_size)


# ---------------------------------------------------------------------------
# Build / save / load
# ---------------------------------------------------------------------------
def build_tiny_transformer(
    config: Optional[TinyTransformerConfig] = None,
) -> Tuple[Any, ByteTokenizer]:
    """Construct a fresh :class:`TransformerDecoder` + tokeniser.

    The import of :class:`TransformerDecoder` is local to avoid a
    top-of-module import cycle (the text subpackage is heavy and
    should not be loaded until the provider layer is actually
    needed).

    Args:
        config: Optional :class:`TinyTransformerConfig`.  Defaults
            to :data:`SMALL_CONFIG`.

    Returns:
        ``(model, tokenizer)`` -- a freshly initialised
        :class:`TransformerDecoder` in ``eval()`` mode and a
        :class:`ByteTokenizer` matched to the model's vocabulary.
    """
    from models.text.transformer import TransformerDecoder

    cfg = config or SMALL_CONFIG
    model = TransformerDecoder(**cfg.to_model_kwargs())
    model.eval()
    tokenizer = ByteTokenizer(vocab_size=cfg.vocab_size)
    return model, tokenizer


def save_tiny_transformer(
    model: nn.Module,
    tokenizer: ByteTokenizer,
    path: Union[str, Path],
    config: Optional[TinyTransformerConfig] = None,
) -> Path:
    """Serialise a tiny Transformer + tokeniser to a single ``.pt`` file.

    The file format is a plain ``torch.save`` payload:

    .. code-block:: python

        {
            "format_version": 1,
            "config": <TinyTransformerConfig.as_dict()>,
            "tokenizer": <ByteTokenizer.state_dict()>,
            "state_dict": <model.state_dict()>,
        }

    The file is written atomically (tempfile + ``os.replace``) so a
    crash mid-write never leaves a half-written file in the cache.

    Args:
        model: The :class:`TransformerDecoder` to serialise.
        tokenizer: The matching :class:`ByteTokenizer`.
        path: Destination path.  Parent directories are created.
        config: Optional config to persist alongside the state-dict.
            When ``None`` the function inspects the model to infer
            a best-effort :class:`TinyTransformerConfig` (it cannot
            always recover every field, so callers should pass one
            explicitly when possible).

    Returns:
        The absolute path that was written.
    """
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if config is None:
        # Best-effort inference from the live model.  The
        # ``name`` field cannot be recovered so it falls back to
        # a generic sentinel.
        cfg = TinyTransformerConfig(
            vocab_size=getattr(model, "vocab_size", DEFAULT_VOCAB_SIZE),
            hidden_size=getattr(model, "hidden_size", 0),
            num_layers=len(getattr(model, "layers", [])),
            num_heads=getattr(model, "num_heads", 0),
            num_kv_heads=getattr(model, "num_kv_heads", None),
            intermediate_size=0,  # not directly recoverable
            max_seq_len=getattr(model, "max_seq_len", 0),
            name="tiny-transformer-unknown",
        )
    else:
        cfg = config

    payload = {
        "format_version": 1,
        "config": cfg.as_dict(),
        "tokenizer": tokenizer.state_dict(),
        "state_dict": model.state_dict(),
    }

    # Atomic write: serialise to a buffer, write to a sibling
    # tempfile, then os.replace.  This is the same pattern used by
    # models.source.cache.
    import os
    import tempfile

    fd, tmp_name = tempfile.mkstemp(
        prefix=out.name + ".", suffix=".tmp", dir=str(out.parent)
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, out)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError as exc:
            _logger.debug("Failed to remove temp file %s: %s", tmp_name, exc)
        raise

    _logger.info(
        "Saved tiny transformer (%d params, %.1fM) to %s",
        sum(p.numel() for p in model.parameters()),
        cfg.approx_params_m(),
        out,
    )
    return out


def load_tiny_transformer(
    path: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Tuple[Any, ByteTokenizer, TinyTransformerConfig]:
    """Load a tiny Transformer from a ``.pt`` file.

    Args:
        path: The file written by :func:`save_tiny_transformer`.
        device: Device to map the loaded tensors onto.
        strict: When ``True`` (default) an unknown / missing key
            in the state-dict raises.  Set to ``False`` to load
            partial weights (e.g. a finetuned adapter).

    Returns:
        ``(model, tokenizer, config)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file's format-version is unrecognised.
        RuntimeError: If the state-dict does not match the model
            (``strict=True``).
    """
    from models.text.transformer import TransformerDecoder

    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError("tiny transformer file not found: {}".format(p))

    payload = torch.load(str(p), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(
            "unexpected payload type {} in {}".format(type(payload), p)
        )
    fmt = int(payload.get("format_version", 0))
    if fmt != 1:
        raise ValueError(
            "unsupported format_version {} in {}".format(fmt, p)
        )

    cfg = TinyTransformerConfig.from_dict(payload.get("config", {}))
    tok = ByteTokenizer.from_state_dict(payload.get("tokenizer", {}))
    model = TransformerDecoder(**cfg.to_model_kwargs())
    state_dict = payload.get("state_dict", {})
    # ``strict=False`` so that loading a partial adapter works
    # without erroring out -- callers that need a hard guarantee
    # should check ``unexpected_keys`` / ``missing_keys``.
    result = model.load_state_dict(state_dict, strict=strict)
    if strict and (result.missing_keys or result.unexpected_keys):
        raise RuntimeError(
            "state-dict mismatch loading {}: missing={}, unexpected={}".format(
                p, result.missing_keys, result.unexpected_keys,
            )
        )
    model = model.to(device)
    model.eval()
    _logger.info(
        "Loaded tiny transformer from %s (params=%.1fM)",
        p, cfg.approx_params_m(),
    )
    return model, tok, cfg
