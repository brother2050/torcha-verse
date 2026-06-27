"""Local-torch text provider for the v0.4.0 P0 milestone (pure-torch).

This module wires a project-owned tiny Transformer LM
(:mod:`models.providers.tiny_transformer`) into the
:class:`models.interfaces.llm_provider.LLMProvider` protocol so that
the 30-node L4 capability layer can be exercised **end-to-end with
a real neural network** (no echo, no passthrough) while still being
*pure torch, zero external dependencies*.

The class is intentionally small:

* it owns a :class:`TransformerDecoder` + :class:`ByteTokenizer`
  loaded from a single ``.pt`` file (or constructed in memory from
  a :class:`TinyTransformerConfig`);
* it implements :meth:`generate` (the only LLMProvider method
  exercised by ``call_text_backend``) and a few chat-shaped
  helpers (:meth:`chat`, :meth:`complete`) that the v0.4.0 P0
  demo / tests use to verify the contract;
* it is **thread-safe** (a single re-entrant lock guards the
  forward pass so concurrent :meth:`generate` calls serialise on
  the same model).

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``models.providers`` (this module) -- real text provider.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from infrastructure.logger import get_logger

from ..interfaces.llm_provider import LLMProvider
from .tiny_transformer import (
    ByteTokenizer,
    SMALL_CONFIG,
    TINY_CONFIG,
    TinyTransformerConfig,
    build_tiny_transformer,
    load_tiny_transformer,
)

__all__ = [
    "LocalTorchTextProvider",
    "GenerationConfig",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Module-level logger.
_logger = get_logger("models.providers.local_text")


# ---------------------------------------------------------------------------
# GenerationConfig
# ---------------------------------------------------------------------------
class GenerationConfig:
    """Sampling parameters for :meth:`LocalTorchTextProvider.generate`.

    Defaults are picked so the output is reproducible (greedy
    decoding) -- callers that want variety bump ``temperature`` and
    set ``do_sample=True``.  The class mirrors the kwargs of
    :meth:`TransformerDecoder.generate` so the two stay in sync.
    """

    __slots__ = (
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
        "do_sample",
        "repetition_penalty",
        "stop_token_ids",
    )

    def __init__(
        self,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
        stop_token_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.max_new_tokens: int = int(max_new_tokens)
        self.temperature: float = float(temperature)
        self.top_k: int = int(top_k)
        self.top_p: float = float(top_p)
        self.do_sample: bool = bool(do_sample)
        self.repetition_penalty: float = float(repetition_penalty)
        self.stop_token_ids: Tuple[int, ...] = tuple(stop_token_ids or ())

    def to_kwargs(self) -> Dict[str, Any]:
        """Return a kwargs dict for :meth:`TransformerDecoder.generate`."""
        kw: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.stop_token_ids:
            kw["stop_token_ids"] = list(self.stop_token_ids)
        return kw

    def __repr__(self) -> str:
        return (
            "GenerationConfig(max_new_tokens={}, temperature={}, top_k={}, "
            "top_p={}, do_sample={}, repetition_penalty={})".format(
                self.max_new_tokens, self.temperature, self.top_k,
                self.top_p, self.do_sample, self.repetition_penalty,
            )
        )


# ---------------------------------------------------------------------------
# LocalTorchTextProvider
# ---------------------------------------------------------------------------
class LocalTorchTextProvider(LLMProvider):
    """A real, project-owned :class:`LLMProvider` backed by ``torch``.

    The provider is **stateless at the framework level** -- it
    holds a single :class:`TransformerDecoder` + :class:`ByteTokenizer`
    pair and serialises concurrent calls behind a lock.  The
    model's :meth:`generate` is invoked in ``torch.no_grad`` mode
    so inference does not allocate autograd graphs.

    Args:
        model: A pre-built :class:`TransformerDecoder`.  When
            ``None`` a fresh :func:`build_tiny_transformer` is
            called.
        tokenizer: A matching :class:`ByteTokenizer`.  When
            ``None`` a default ``ByteTokenizer`` for the config's
            ``vocab_size`` is used.
        config: The :class:`TinyTransformerConfig` that was used
            to build the model.  When ``None`` a default
            :data:`TINY_CONFIG` is used.
        device: Device to run the model on.  Defaults to CPU so
            the provider is portable across CI environments.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[ByteTokenizer] = None,
        config: Optional[TinyTransformerConfig] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        if model is None or tokenizer is None or config is None:
            default = config or TINY_CONFIG
            built_model, built_tok = build_tiny_transformer(default)
            model = model or built_model
            tokenizer = tokenizer or built_tok
            config = config or default
        if not isinstance(tokenizer, ByteTokenizer):
            raise TypeError(
                "`tokenizer` must be a ByteTokenizer (got {})".format(
                    type(tokenizer).__name__
                )
            )
        self._model: nn.Module = model.to(device)
        self._model.eval()
        self._tokenizer: ByteTokenizer = tokenizer
        self._config: TinyTransformerConfig = config
        self._device: torch.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        self._lock: threading.RLock = threading.RLock()
        self._logger = _logger

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> "LocalTorchTextProvider":
        """Load a provider from a ``.pt`` file (see
        :func:`models.providers.tiny_transformer.load_tiny_transformer`)."""
        model, tok, cfg = load_tiny_transformer(path, device=device)
        return cls(model=model, tokenizer=tok, config=cfg, device=device)

    @classmethod
    def from_random(
        cls,
        config: Optional[TinyTransformerConfig] = None,
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> "LocalTorchTextProvider":
        """Construct a provider with a freshly initialised model.

        Useful for CI smoke tests and the v0.4.0 P0 demo when no
        pre-trained checkpoint is available -- the model will
        produce mostly random text but the contract (input ids in,
        text out) is fully exercised.
        """
        cfg = config or TINY_CONFIG
        model, tok = build_tiny_transformer(cfg)
        return cls(model=model, tokenizer=tok, config=cfg, device=device)

    @classmethod
    def from_wrapped_model(
        cls,
        bundle,
        *,
        device: Union[str, torch.device] = "cpu",
    ) -> "LocalTorchTextProvider":
        """Build a provider on top of a :func:`load_model_and_tokenizer`
        bundle (real user-selected weights).

        The ``bundle`` is a 3-tuple ``(model, tokenizer, family)`` as
        returned by :func:`models.runtime.load_model_and_tokenizer`.
        The factory passes the upstream model + tokenizer through
        directly.  When the upstream model exposes
        :meth:`generate(prompt, **kwargs) -> str` (the case for
        :class:`ModelMixin`-backed architectures) we call it
        verbatim; otherwise we use the standard
        tokenize-→forward-→decode path with a uniform sampling
        loop.

        The returned provider is **stateless w.r.t. the
        architecture**: callers should set ``self._architecture``
        if they need to know which one is loaded.
        """
        if isinstance(bundle, tuple) and len(bundle) == 3:
            model, tokenizer, family = bundle
        elif isinstance(bundle, tuple) and len(bundle) == 2:
            model, tokenizer = bundle
            family = None
        else:
            model, tokenizer = bundle, None
        # Pass through the existing constructor when the bundle
        # already matches the micro-transformer shape.  Otherwise
        # we attach the model + tokenizer manually and mark the
        # provider as "user-loaded" so the generate/embed paths
        # route to the real weights.
        if (
            isinstance(tokenizer, ByteTokenizer)
            and hasattr(model, "config")
            and isinstance(getattr(model, "config", None), TinyTransformerConfig)
        ):
            return cls(
                model=model, tokenizer=tokenizer,
                config=getattr(model, "config"),
                device=device,
            )
        # Generic user-loaded path.
        instance = cls.__new__(cls)
        instance._model = model.to(device)
        instance._model.eval()
        instance._tokenizer = tokenizer
        instance._config = getattr(model, "config", None)
        instance._device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        instance._lock = threading.RLock()
        instance._logger = _logger
        instance._user_loaded = True
        instance._family = family
        return instance

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------
    def embed(self, text: str) -> List[float]:
        """Project ``text`` to a dense vector (mean-pooled last hidden state).

        Used by the RAG stack to project natural-language queries
        and documents into the vector-store space.  The result is
        an L2-normalised float list with length
        ``config.d_model`` (typically 64 / 128 / 256 depending on
        the configured tiny model).

        Implementation: encode ``text`` with the underlying
        :class:`TransformerDecoder`, mean-pool across the time
        axis (ignoring the leading SOS / final EOS if any), and
        L2-normalise so cosine similarity == inner product.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Project a batch of texts to a matrix of L2-normalised vectors.

        Returns one float list per input string.  Empty input
        returns an empty list.  The implementation pads the
        batch internally to the longest sequence length and runs
        a single forward pass.
        """
        texts = list(texts)
        if not texts:
            return []
        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(
                    f"texts[{i}] must be str, got {type(t).__name__}"
                )
        with self._lock:
            with torch.no_grad():
                encoded = [self._tokenizer.encode(t) for t in texts]
                max_len = max(len(ids) for ids in encoded)
                # Pad with the tokenizer's pad id (0 if not set).
                pad_id = getattr(self._tokenizer, "pad_id", 0) or 0
                batch = torch.full(
                    (len(encoded), max_len), pad_id, dtype=torch.long
                )
                for i, ids in enumerate(encoded):
                    batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                batch = batch.to(self._device)
                # ``TransformerDecoder`` accepts (B, T) int64 token
                # ids; if a different signature is used the
                # fallback path below kicks in.
                try:
                    hidden = self._model(batch)  # type: ignore[arg-type]
                except TypeError:
                    # Some decoders take ``inputs`` / ``tokens`` kwargs.
                    hidden = self._model(inputs=batch)  # type: ignore[call-arg]
                # ``hidden`` may be a tensor (B, T, D) or an object
                # exposing ``last_hidden_state``.
                if hasattr(hidden, "last_hidden_state"):
                    hidden = hidden.last_hidden_state
                if not torch.is_tensor(hidden):
                    raise RuntimeError(
                        "LocalTorchTextProvider.embed_batch: model output is not a "
                        "tensor; cannot mean-pool."
                    )
                # Build a mask (1 for real tokens, 0 for padding).
                mask = (batch != pad_id).unsqueeze(-1).to(hidden.dtype)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp_min(1.0)
                pooled = summed / counts
                # L2-normalise.
                norms = pooled.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                pooled = pooled / norms
                return pooled.detach().cpu().tolist()

    def generate(
        self,
        prompt: Union[str, Sequence[int]],
        *,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = False,
        **kwargs: Any,
    ) -> str:
        """Generate a continuation of ``prompt`` and return the decoded text.

        Args:
            prompt: A string (encoded with the :class:`ByteTokenizer`)
                or a pre-tokenised id sequence.
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k truncation; ``0`` disables.
            top_p: Top-p (nucleus) truncation; ``1.0`` disables.
            do_sample: When ``True`` sample, otherwise greedy.
            **kwargs: Forwarded to :meth:`TransformerDecoder.generate`
                (e.g. ``stop_token_ids``).

        Returns:
            The decoded string.  The original prompt is *not*
            included in the output (it is consumed during the
            forward pass).  When ``stop_token_ids`` is supplied
            the model stops at the first match and the stop token
            itself is dropped from the output.
        """
        # User-loaded path: defer to whatever the upstream model
        # exposes.  ``ModelMixin`` provides ``generate(prompt, **kw)``
        # returning a string, which is the preferred surface; we
        # only fall back to the bytes / token-id path when the
        # model has a ``generate`` that expects a tensor.
        if getattr(self, "_user_loaded", False):
            with self._lock:
                with torch.no_grad():
                    if isinstance(prompt, str) and hasattr(
                        self._model, "generate"
                    ):
                        try:
                            out = self._model.generate(
                                prompt,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                do_sample=do_sample,
                                **kwargs,
                            )
                            if isinstance(out, str):
                                return out
                        except TypeError:
                            # The model.generate signature is
                            # tensor-shaped; fall through to the
                            # standard path below using the
                            # attached tokenizer.
                            pass
                    # Fallback: tokenize via the bundle's
                    # ``TokenizerBundle`` (project-aware) then
                    # forward to ``self._model.generate``.
                    if self._tokenizer is None:
                        raise RuntimeError(
                            "LocalTorchTextProvider.generate: user-loaded "
                            "model has no tokenizer attached; cannot decode."
                        )
                    if isinstance(prompt, str):
                        if hasattr(self._tokenizer, "encode"):
                            ids = self._tokenizer.encode(
                                prompt, add_bos=True, add_eos=False,
                            )
                        else:
                            # ``TokenizerBundle`` from
                            # :mod:`models.runtime` falls back
                            # to byte-level encoding when no
                            # vocab is present.
                            ids = [b + 3 for b in prompt.encode(
                                "utf-8", errors="ignore",
                            )[: 254]]
                            ids = [1] + ids + [2]
                    else:
                        ids = list(prompt)
                    if not ids:
                        ids = [1]
                    input_ids = torch.tensor(
                        [ids], dtype=torch.long, device=self._device,
                    )
                    output = self._model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=do_sample,
                        **kwargs,
                    )
                    out_ids = (
                        output[0].tolist()
                        if hasattr(output, "shape")
                        else list(output[0])
                    )
                    if hasattr(self._tokenizer, "decode"):
                        return self._tokenizer.decode(out_ids, skip_special=True)
                    return "".join(
                        chr(i) for i in out_ids
                        if 32 <= i < 0x110000
                    )[: 256]

        # Encode + ensure tensor shape (1, T).
        if isinstance(prompt, str):
            ids = self._tokenizer.encode(prompt, add_bos=True, add_eos=False)
        else:
            ids = list(prompt)
        if not ids:
            ids = [self._tokenizer.bos_token_id]
        input_ids = torch.tensor(
            [ids], dtype=torch.long, device=self._device,
        )

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )
        kw = gen_cfg.to_kwargs()
        # Forward any explicit kwargs the caller passed (e.g.
        # ``stop_token_ids``), but do not let them silently
        # override the sampling knobs -- merge with our defaults
        # so an explicit ``do_sample=False`` wins, etc.
        for k, v in kwargs.items():
            kw.setdefault(k, v)

        with self._lock:
            with torch.no_grad():
                output = self._model.generate(input_ids, **kw)

        out_ids = output[0].tolist()
        raw = self._tokenizer.decode(out_ids, skip_special=True)
        # v0.10.5: sanitise the raw byte-level output.  The
        # ByteTokenizer's ``decode`` already emits full U+FFFD
        # sequences for out-of-vocab ids (no more half-character
        # truncation), but the *raw* result can still be polluted
        # by ASCII control characters leaked from id samples in
        # ``[0, 32)`` and by long FFFD runs at the tail of the
        # sampling loop.  We keep the raw string as
        # ``self._last_raw`` for diagnostics and return the
        # sanitised version as the user-visible result.
        self._last_raw = raw
        try:
            from models.providers._text_sanitiser import (
                sanitise_generation,
                garble_assessment,
            )
            level, reason = garble_assessment(raw)
            self._last_garble_level = level
            self._last_garble_reason = reason
            return sanitise_generation(raw)
        except Exception:  # noqa: BLE001 - never break generate
            return raw

    # ------------------------------------------------------------------
    # Chat-shaped helpers (used by the v0.4.0 P0 demo + tests)
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        max_new_tokens: int = 64,
        do_sample: bool = False,
    ) -> str:
        """Format ``messages`` as ``"user: ... assistant: "`` and generate.

        The model is a small byte-level LM; the chat template is a
        minimal prompt-format string -- it is *not* a full
        ChatML / Llama-3 format.  That is intentional: the goal
        of v0.4.0 P0 is to prove the end-to-end pipeline works,
        not to ship a SOTA conversational model.

        Args:
            messages: Sequence of ``{"role": ..., "content": ...}``
                dicts.  Unknown roles are passed through verbatim.
            max_new_tokens: New-token budget for the reply.
            do_sample: Whether to sample.

        Returns:
            The decoded assistant reply (without the prompt).
        """
        if not messages:
            raise ValueError("`messages` must be non-empty")
        prompt_parts: List[str] = []
        for m in messages:
            role = str(m.get("role", "user")).strip() or "user"
            content = str(m.get("content", "")).strip()
            prompt_parts.append("{}: {}".format(role, content))
        prompt_parts.append("assistant:")
        prompt = "\n".join(prompt_parts)
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    def complete(self, prefix: str, **kwargs: Any) -> str:
        """Generate a completion following ``prefix`` (no chat template)."""
        return self.generate(prefix, **kwargs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def model(self) -> nn.Module:
        """The underlying :class:`TransformerDecoder` (read-only)."""
        return self._model

    @property
    def tokenizer(self) -> ByteTokenizer:
        """The :class:`ByteTokenizer` (read-only)."""
        return self._tokenizer

    @property
    def config(self) -> TinyTransformerConfig:
        """The :class:`TinyTransformerConfig` (read-only)."""
        return self._config

    @property
    def device(self) -> torch.device:
        """The device the model is bound to."""
        return self._device

    def num_parameters(self) -> int:
        """Return the total number of parameters in the model."""
        return sum(p.numel() for p in self._model.parameters())

    def __repr__(self) -> str:
        return (
            "LocalTorchTextProvider(name={!r}, params={}, vocab_size={}, "
            "device={!r})".format(
                self._config.name,
                self.num_parameters(),
                self._config.vocab_size,
                self._device,
            )
        )
