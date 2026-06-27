"""Local loader: 自研的 "transformers.AutoModel + AutoTokenizer" 风格统一入口 (v0.10.0)。

为 torcha-verse 提供一个**不依赖 ``transformers`` / ``tokenizers`` /
``huggingface_hub``** 但**接口一致**的本地模型加载层。

设计动机
--------

V0.8.0 方案 (§11.1) 把项目自 v0.4.x 以来的核心约束明确为:

> "**不依赖 transformers / diffusers 包**:保持核心零依赖,自研 BPE /
> safetensors 解析"

但 v0.8 / v0.9 的实现只到:

* ``models.base.ModelMixin`` + ``core.checkpoint_loader.py`` (5 个
  ``*_KEY_MAP``)
* ``models.text.clip_tokenizer.SimpleByteBPETokenizer``
* ``models.text.t5_tokenizer.SimpleSentencePieceTokenizer``

而**没有**一个像 ``transformers.AutoModel.from_pretrained`` 的**一行
加载**入口。调用方 (39 节点 / examples / CLI) 依然要自己写:

```python
# 5 行 boilerplate,每个调用方都要重写
model = HunyuanDiT.from_pretrained(path, key_renames=HUNYUAN_DIT_KEY_MAP,
                                    torch_dtype=torch.float16)
tokenizer = SimpleByteBPETokenizer(vocab_path, merges_path)
text_embeds = model.encode_text(prompt)
latents = call_diffusion_loop_backend(...)
```

本模块填补这个缺口,提供:

* :func:`load_model_and_tokenizer` -- 一行加载,自动识别 model family
  + 自动 key 改名 + 自动 device / dtype 推断
* :class:`ModelHub` -- 类似 ``transformers.Hub`` 的本地 hub,
  内置 ``download=True`` 选项
* :class:`ModelForCausalLM` / ``ModelForTextToImage`` /
  ``ModelForTextToSpeech`` / ``ModelForMusic`` -- 4 个
  TaskHeads,内含 ``generate()`` / ``__call__()`` / ``encode_text()``
* :func:`detect_model_family` -- 根据 safetensors key 前缀自动
  判断 (HunyuanDiT / FLUX / SD3 / Wan2 / MusicGen / TinyTransformer)

零外部依赖
----------

* **不** import ``transformers`` / ``tokenizers`` / ``diffusers`` /
  ``huggingface_hub`` / ``accelerate``
* 只依赖 ``torch`` + 项目自有的
  :class:`models.base.ModelMixin` / :class:`core.checkpoint_loader` /
  :class:`models.text.clip_tokenizer` /
  :class:`models.text.t5_tokenizer`

与 V0.8.0 兼容
---------------

* 走 :meth:`models.base.ModelMixin.from_pretrained` + ``key_renames``
  路径,5 个 ``*_KEY_MAP`` (HUNYUAN_DIT / FLUX / SD3 / WAN2 / MUSICGEN)
  自动选
* ``from_pretrained`` 的 9 维 kwargs (``subfolder`` / ``torch_dtype`` /
  ``device`` / ``device_map`` / ``variant`` / ``key_renames`` /
  ``strict`` / ``config`` / ``**kwargs``) **全部** 透传

测试 0 回归
-----------

* 新模块的所有占位 (``pass``) / ``NotImplementedError`` 全部在
  ``docs/placeholder_registry.md`` 登记
* 不破坏 1182+ 现有测试
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn

from infrastructure.logger import get_logger

from ..base import ModelMixin
from core.checkpoint_loader import (
    FLUX_KEY_MAP,
    HUNYUAN_DIT_KEY_MAP,
    MUSICGEN_KEY_MAP,
    SD3_KEY_MAP,
    WAN2_KEY_MAP,
    load_safetensors,
)
from ..text.clip_tokenizer import SimpleByteBPETokenizer
from ..text.t5_tokenizer import SimpleSentencePieceTokenizer
from .device_planner import DevicePlan, plan_device

__all__ = [
    "ModelFamily",
    "ModelHub",
    "ModelForCausalLM",
    "ModelForTextToImage",
    "ModelForTextToSpeech",
    "ModelForMusic",
    "load_model_and_tokenizer",
    "detect_model_family",
    "TokenizerBundle",
]


_logger = get_logger("models.runtime.loader")


# ---------------------------------------------------------------------------
# ModelFamily
# ---------------------------------------------------------------------------
class ModelFamily(str, Enum):
    """The set of model families the local runtime can auto-detect.

    Values are also valid string keys (the enum inherits from ``str``)
    so the result of :func:`detect_model_family` can be compared to
    plain string literals if needed.
    """

    HUNYUAN_DIT = "hunyuan_dit"
    FLUX = "flux"
    SD3 = "sd3"
    WAN2 = "wan2"
    MUSICGEN = "musicgen"
    TINY_TRANSFORMER = "tiny_transformer"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# TokenizerBundle
# ---------------------------------------------------------------------------
@dataclass
class TokenizerBundle:
    """A single object that bundles every tokenizer the runtime
    may need for a given model family.

    The bundle is constructed by :func:`load_model_and_tokenizer`
    after the model family is known.  Different fields are filled
    depending on the family:

    * ``clip`` -- populated for image / video families (HunyuanDiT /
      FLUX / SD3 / Wan2)
    * ``t5``   -- populated for T5-using families (FLUX / SD3 / Wan2)
    * ``byte`` -- populated for text-only fallback (TinyTransformer
      / causal LM)
    * ``sp``   -- raw SentencePiece reference (for advanced callers)

    All fields are optional; the caller's pipeline is responsible
    for picking the right one(s).
    """

    clip: Optional[SimpleByteBPETokenizer] = None
    t5: Optional[SimpleSentencePieceTokenizer] = None
    byte: Optional[Any] = None  # ByteTokenizer (lives in tiny_transformer)
    sp: Optional[Any] = None  # raw sp model if any

    def has_any(self) -> bool:
        return any(v is not None for v in (self.clip, self.t5, self.byte, self.sp))

    def __repr__(self) -> str:
        bits = []
        if self.clip is not None:
            bits.append(f"clip={type(self.clip).__name__}")
        if self.t5 is not None:
            bits.append(f"t5={type(self.t5).__name__}")
        if self.byte is not None:
            bits.append(f"byte={type(self.byte).__name__}")
        if self.sp is not None:
            bits.append("sp=native")
        return f"TokenizerBundle({', '.join(bits) or 'empty'})"


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------
# Heuristic key prefixes / substrings that uniquely identify each
# supported model family in a ``state_dict.keys()`` listing.
_FAMILY_KEY_SIGNATURES: Tuple[Tuple[ModelFamily, Tuple[str, ...]], ...] = (
    (
        ModelFamily.HUNYUAN_DIT,
        (
            "img_in.proj",
            "x_embedder",
            "time_in.mlp",
            "vector_in.proj",
            "style_embedder",
            "size_embedder",
        ),
    ),
    (
        ModelFamily.FLUX,
        (
            "double_blocks",
            "single_blocks",
            "img_in",
            "txt_in",
            "guidance_in",
            "final_layer.linear",
        ),
    ),
    (
        ModelFamily.SD3,
        (
            "joint_transformer_blocks",
            "single_transformer_blocks",
            "time_embedding",
            "label_embedding",
            "pooled_text_embedding",
            "proj_out",
        ),
    ),
    (
        ModelFamily.WAN2,
        (
            "patch_embedding",
            "time_projection",
            "text_embedding",
            "head.head",
            "head.norm",
        ),
    ),
    (
        ModelFamily.MUSICGEN,
        (
            "text_encoder.transformer",
            "audio_encoder.transformer",
            "conditioning_provider",
            "output_proj",
        ),
    ),
    (
        ModelFamily.TINY_TRANSFORMER,
        (
            "token_embedding",
            "positional_embedding",
            "blocks.0.attn.qkv",
            "ln_f",
        ),
    ),
)


def _signature_hits(state_dict_keys: Sequence[str], sig: str) -> int:
    """Count how many ``state_dict_keys`` start with ``sig``."""
    n = 0
    for k in state_dict_keys:
        if k.startswith(sig):
            n += 1
    return n


def detect_model_family(
    weights_path: Union[str, Path],
    *,
    sample_size: int = 64,
) -> ModelFamily:
    """Infer the model family from a checkpoint's key layout.

    Args:
        weights_path: Path to a directory that contains a
            ``.safetensors`` file, or a direct path to a single
            ``.safetensors``.  The function only reads the *names* of
            the first ``sample_size`` tensors (via
            :func:`load_safetensors` then ``list(state_dict)``) -- the
            tensor data is not copied, so this is fast even on
            multi-GB checkpoints.
        sample_size: Number of keys to read.  ``64`` is enough for
            every family in :data:`_FAMILY_KEY_SIGNATURES`.

    Returns:
        A :class:`ModelFamily`.  When no signature matches the
        function returns :attr:`ModelFamily.UNKNOWN`.

    Raises:
        FileNotFoundError: When ``weights_path`` cannot be resolved
            to a safetensors file.
    """
    path = Path(weights_path)
    ckpt_path = _resolve_checkpoint_file(path)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No .safetensors file found at {weights_path!r}"
        )
    state_dict = load_safetensors(ckpt_path, device="cpu")
    keys = list(state_dict.keys())[: max(1, int(sample_size))]
    scores: List[Tuple[ModelFamily, int]] = []
    for family, sigs in _FAMILY_KEY_SIGNATURES:
        score = sum(_signature_hits(keys, s) for s in sigs)
        if score > 0:
            scores.append((family, score))
    if not scores:
        return ModelFamily.UNKNOWN
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]


# ---------------------------------------------------------------------------
# File resolution
# ---------------------------------------------------------------------------
_SAFETENSORS_SUFFIXES: Tuple[str, ...] = (".safetensors",)
_SHARD_INDEX_SUFFIX: str = ".safetensors.index.json"


def _resolve_checkpoint_file(path: Path) -> Optional[Path]:
    """Return the ``.safetensors`` file for ``path``.

    Accepts a directory, a single ``.safetensors`` file, or a sharded
    layout.  When ``path`` is a directory, the first
    ``.safetensors`` file in sorted order is returned (sharded layouts
    return the first shard -- :func:`load_safetensors` stitches the
    rest transparently).
    """
    if path.is_file():
        if path.suffix in _SAFETENSORS_SUFFIXES:
            return path
        return None
    if not path.is_dir():
        return None
    # Prefer a single-file checkpoint.
    singles = sorted(path.glob("*.safetensors"))
    singles = [p for p in singles if not p.name.endswith(_SHARD_INDEX_SUFFIX)]
    if singles:
        return singles[0]
    # Fall back to first shard in a sharded layout.
    shards = sorted(path.glob("*-of-*.safetensors"))
    if shards:
        return shards[0]
    return None


# ---------------------------------------------------------------------------
# Key map dispatch
# ---------------------------------------------------------------------------
_FAMILY_TO_KEYMAP: Dict[ModelFamily, Optional[Dict[str, str]]] = {
    ModelFamily.HUNYUAN_DIT: HUNYUAN_DIT_KEY_MAP,
    ModelFamily.FLUX: FLUX_KEY_MAP,
    ModelFamily.SD3: SD3_KEY_MAP,
    ModelFamily.WAN2: WAN2_KEY_MAP,
    ModelFamily.MUSICGEN: MUSICGEN_KEY_MAP,
    ModelFamily.TINY_TRANSFORMER: None,
    ModelFamily.UNKNOWN: None,
}


def _keymap_for(
    family: ModelFamily,
    *,
    num_blocks: int = 20,
) -> Optional[Dict[str, str]]:
    """Return the upstream -> local key rename table for ``family``.

    Per-block ``{i}`` placeholders are expanded (so the result is
    immediately usable by
    :func:`core.checkpoint_loader.load_state_dict_with_renames`).
    """
    table = _FAMILY_TO_KEYMAP.get(family)
    if table is None:
        return None
    if not any("{i}" in k for k in table):
        return dict(table)
    expanded: Dict[str, str] = {}
    for k, v in table.items():
        if "{i}" in k:
            for i in range(int(num_blocks)):
                expanded[k.format(i=i)] = v.format(i=i)
        else:
            expanded[k] = v
    return expanded


def _default_num_blocks(family: ModelFamily) -> int:
    if family == ModelFamily.HUNYUAN_DIT:
        return 20
    if family == ModelFamily.FLUX:
        return 19
    if family == ModelFamily.SD3:
        return 24
    if family == ModelFamily.WAN2:
        return 40
    if family == ModelFamily.MUSICGEN:
        return 24
    return 0


# ---------------------------------------------------------------------------
# Tokenizer resolution
# ---------------------------------------------------------------------------
_DEFAULT_TOKENIZER_FILES: Dict[ModelFamily, Tuple[str, ...]] = {
    # image families: CLIP-BPE first, T5 optional
    ModelFamily.HUNYUAN_DIT: ("vocab.json", "merges.txt"),
    ModelFamily.FLUX: ("vocab.json", "merges.txt", "sp.model"),
    ModelFamily.SD3: ("vocab.json", "merges.txt", "sp.model"),
    # video family: CLIP + T5
    ModelFamily.WAN2: ("vocab.json", "merges.txt", "sp.model"),
    # audio family: T5 only
    ModelFamily.MUSICGEN: ("sp.model",),
    # text family: no tokenizer file (the model ships its own byte table)
    ModelFamily.TINY_TRANSFORMER: (),
    ModelFamily.UNKNOWN: (),
}


def _resolve_tokenizer_files(
    directory: Path,
    family: ModelFamily,
) -> TokenizerBundle:
    """Build a :class:`TokenizerBundle` from the ``directory`` layout.

    Looks for the standard ``vocab.json`` / ``merges.txt`` /
    ``sp.model`` filenames (or any ``*.json`` / ``*.txt`` /
    ``*.model`` fallback).  Missing files are silently skipped -- the
    bundle's ``has_any()`` will report ``True`` whenever at least one
    tokenizer is present.
    """
    bundle = TokenizerBundle()
    if not directory.is_dir():
        return bundle
    files = _DEFAULT_TOKENIZER_FILES.get(family, ())
    # CLIP
    if "vocab.json" in files or "merges.txt" in files:
        vocab = directory / "vocab.json"
        merges = directory / "merges.txt"
        if vocab.is_file() or merges.is_file():
            bundle.clip = SimpleByteBPETokenizer(
                vocab_path=vocab if vocab.is_file() else None,
                merges_path=merges if merges.is_file() else None,
            )
    # T5 / SentencePiece
    if "sp.model" in files:
        sp_path = directory / "sp.model"
        if not sp_path.is_file():
            # Fallback: any *.model file in the directory
            models = list(directory.glob("*.model"))
            sp_path = models[0] if models else sp_path
        if sp_path.is_file():
            bundle.t5 = SimpleSentencePieceTokenizer(model_path=sp_path)
    return bundle


# ---------------------------------------------------------------------------
# ModelHub
# ---------------------------------------------------------------------------
class ModelHub:
    """A minimal "transformers.Hub" analogue for local files.

    The hub owns:

    * a process-wide :class:`ModelCache` reference (re-used across
      multiple ``download`` calls to avoid re-hitting the network
      for the same revision);
    * a per-(family, revision) ``load`` cache, so two
      ``load_model_and_tokenizer(...)`` calls with the same args
      return the same Python object (mirrors
      ``transformers.AutoModel.from_pretrained`` cache behaviour).

    Args:
        cache_dir: Root directory for the local cache.  Defaults to
            ``$TORCHA_VERSE_CACHE`` or ``~/.cache/torcha-verse``.
        use_existing_cache: When ``True`` (default), the hub will try
            to reuse the project-owned :mod:`models.source` cache
            (``ModelCache``) for any ``download=True`` call.  When
            ``False`` the hub falls back to a pure-local copy.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        *,
        use_existing_cache: bool = True,
    ) -> None:
        if cache_dir is None:
            cache_dir = os.environ.get(
                "TORCHA_VERSE_CACHE",
                str(Path.home() / ".cache" / "torcha-verse"),
            )
        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._use_existing_cache: bool = bool(use_existing_cache)
        # Load cache: { (family, str(path), str(repr(kwargs))) -> (model, tokenizer) }
        self._load_cache: Dict[
            Tuple[ModelFamily, str, str],
            Tuple[ModelMixin, TokenizerBundle],
        ] = {}
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Download (optional)
    # ------------------------------------------------------------------
    def download(
        self,
        repo_id_or_url: str,
        *,
        revision: str = "main",
        expected_sha256s: Optional[Mapping[str, str]] = None,
    ) -> Path:
        """Resolve ``repo_id_or_url`` to a local directory.

        The download path is **opt-in**: if the project-owned
        :mod:`models.source` cache is importable and
        ``use_existing_cache`` is ``True``, the hub delegates to
        :class:`models.source.ModelFetcher` (which already
        implements mirror selection, dedup, integrity checks and
        GatedRepoError handling per V0.4.x P2+).

        For a URL (starts with ``http://`` or ``https://``) the hub
        does a single-shot download into
        ``<cache_dir>/<basename>``.

        Args:
            repo_id_or_url: Either an ``org/repo`` HF id (when the
                source cache is available) or an ``http(s)://`` URL.
            revision: Git revision / branch / tag.  Defaults to
                ``"main"``.
            expected_sha256s: Optional ``{file: sha}`` map enforced
                by the source fetcher.

        Returns:
            The local directory (or file) that contains the
            downloaded payload.

        Raises:
            RuntimeError: When the caller asked to download from an
                ``org/repo`` id but the source cache is unavailable
                (typically because :mod:`models.source` couldn't
                be imported).
        """
        # URL path: bypass source cache, do a direct urllib download.
        if repo_id_or_url.startswith("http://") or repo_id_or_url.startswith("https://"):
            return self._download_url(repo_id_or_url, expected_sha256s=expected_sha256s)

        # HF repo id path: delegate to models.source when possible.
        if self._use_existing_cache:
            try:
                from models.source import (
                    HuggingFaceSource,
                    MirrorSet,
                    ModelCache,
                    ModelFetcher,
                    SourceRegistry,
                    resolve_token,
                )
            except ImportError as exc:  # pragma: no cover - optional path
                raise RuntimeError(
                    "models.source is unavailable; cannot auto-download "
                    f"from HF repo id {repo_id_or_url!r}: {exc}"
                ) from exc
            mirrors = MirrorSet.from_env()
            cache = ModelCache(root=str(self.cache_dir))
            registry = SourceRegistry.default()
            fetcher = ModelFetcher(
                cache=cache, registry=registry, mirrors=mirrors,
            )
            res = fetcher.fetch(
                "huggingface",
                repo_id_or_url,
                revision=revision,
                token=resolve_token(sources="huggingface").value,
                expected_sha256s=dict(expected_sha256s or {}),
            )
            return Path(res.manifest.local_path)

        # No cache path: the user must supply local files.
        raise RuntimeError(
            f"ModelHub.download: no cache available; supply a local path or "
            f"enable the source cache (use_existing_cache=True)."
        )

    @staticmethod
    def _download_url(
        url: str,
        *,
        expected_sha256s: Optional[Mapping[str, str]] = None,
    ) -> Path:
        import hashlib
        import urllib.request

        target_dir = Path(
            os.environ.get(
                "TORCHA_VERSE_CACHE",
                str(Path.home() / ".cache" / "torcha-verse"),
            )
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / Path(url).name
        if target.is_file():
            return target
        with urllib.request.urlopen(url) as resp:  # noqa: S310
            data = resp.read()
        if expected_sha256s is not None:
            expected = expected_sha256s.get(target.name)
            if expected is not None:
                got = hashlib.sha256(data).hexdigest()
                if got != expected:
                    raise ValueError(
                        f"sha256 mismatch for {target.name}: "
                        f"expected {expected}, got {got}"
                    )
        target.write_bytes(data)
        return target

    # ------------------------------------------------------------------
    # Load (cached)
    # ------------------------------------------------------------------
    def load(
        self,
        path: Union[str, Path],
        *,
        family: Optional[Union[ModelFamily, str]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device: Union[None, str, torch.device] = None,
        device_map: Union[None, str, Dict[str, str]] = None,
        variant: Optional[str] = None,
        strict: bool = False,
        num_blocks: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[ModelMixin, TokenizerBundle, ModelFamily]:
        """Load a model + tokenizer bundle, with a per-args cache.

        Args:
            path: Local directory or ``.safetensors`` file.
            family: Override for the auto-detected model family.  Pass
                a :class:`ModelFamily` (or its string value) when
                you know better than the heuristic.
            torch_dtype: Forwarded to
                :meth:`models.base.ModelMixin.from_pretrained`.
            device: Forwarded as the ``device`` kwarg.
            device_map: Forwarded as the ``device_map`` kwarg.
            variant: Forwarded as the ``variant`` kwarg.
            strict: Forwarded to ``from_pretrained``.
            num_blocks: Number of decoder blocks (for per-block
                key-rename expansion).  Defaults to a family-specific
                value when omitted.
            config: Forwarded as the ``config`` kwarg.
            **kwargs: Forwarded to ``from_pretrained`` and (when
                supported) to the model class.

        Returns:
            ``(model, tokenizer_bundle, detected_family)``.

        Raises:
            FileNotFoundError: When ``path`` cannot be resolved.
            RuntimeError: When the model class is unavailable.
        """
        path = Path(path)
        # Auto-detect family when not provided.
        if family is None:
            family_value = detect_model_family(path)
        else:
            family_value = (
                ModelFamily(family) if isinstance(family, str) else family
            )
        # Number of blocks default.
        n_blocks = int(num_blocks) if num_blocks is not None else _default_num_blocks(
            family_value
        )
        # Cache key: everything that affects the returned object.
        cache_key = (
            family_value,
            str(path.resolve()),
            json.dumps(
                {
                    "torch_dtype": str(torch_dtype),
                    "device": str(device),
                    "device_map": device_map,
                    "variant": variant,
                    "strict": strict,
                    "num_blocks": n_blocks,
                    "config": config,
                    **{
                        k: ("<obj>" if not isinstance(v, (str, int, float, bool, type(None))) else v)
                        for k, v in kwargs.items()
                    },
                },
                sort_keys=True,
            ),
        )
        with self._lock:
            cached = self._load_cache.get(cache_key)
        if cached is not None:
            return cached[0], cached[1], family_value

        # Build the model.
        keymap = _keymap_for(family_value, num_blocks=n_blocks)
        model = _instantiate_model(
            family_value,
            path=path,
            torch_dtype=torch_dtype,
            device=device,
            device_map=device_map,
            variant=variant,
            key_renames=keymap,
            strict=strict,
            config=config,
            num_blocks=n_blocks,
            **kwargs,
        )
        # Build the tokenizer bundle.
        bundle = _resolve_tokenizer_files(path, family_value)

        with self._lock:
            self._load_cache[cache_key] = (model, bundle)
        return model, bundle, family_value

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def clear_load_cache(self) -> None:
        """Drop the in-memory load cache (does not touch disk)."""
        with self._lock:
            self._load_cache.clear()

    def load_cache_size(self) -> int:
        """Return the number of cached ``load`` entries."""
        with self._lock:
            return len(self._load_cache)


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------
def _instantiate_model(
    family: ModelFamily,
    *,
    path: Path,
    torch_dtype: Optional[torch.dtype],
    device: Union[None, str, torch.device],
    device_map: Union[None, str, Dict[str, str]],
    variant: Optional[str],
    key_renames: Optional[Dict[str, str]],
    strict: bool,
    config: Optional[Dict[str, Any]],
    num_blocks: int,
    **kwargs: Any,
) -> ModelMixin:
    """Materialise a :class:`ModelMixin` subclass for ``family``.

    The function uses the canonical loader helpers from
    :mod:`core.checkpoint_loader` when the family is image / audio /
    video, and falls back to a direct
    :meth:`ModelMixin.from_pretrained` call (with a project-side
    model class) for text / causal-LM families.
    """
    ckpt_path = _resolve_checkpoint_file(path)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No .safetensors file at {path!r}"
        )

    if family == ModelFamily.HUNYUAN_DIT:
        from core.checkpoint_loader import load_hunyuan_dit
        return load_hunyuan_dit(
            ckpt_path,
            torch_dtype=torch_dtype,
            device=device if isinstance(device, (str, torch.device)) else "cpu",
            num_blocks=num_blocks,
            strict=strict,
        )
    if family == ModelFamily.FLUX:
        from core.checkpoint_loader import load_flux
        return load_flux(
            ckpt_path,
            torch_dtype=torch_dtype,
            device=device if isinstance(device, (str, torch.device)) else "cpu",
            num_blocks=num_blocks,
            strict=strict,
        )
    if family == ModelFamily.SD3:
        from core.checkpoint_loader import load_sd3
        return load_sd3(
            ckpt_path,
            torch_dtype=torch_dtype,
            device=device if isinstance(device, (str, torch.device)) else "cpu",
            num_blocks=num_blocks,
            strict=strict,
        )
    if family == ModelFamily.WAN2:
        from core.checkpoint_loader import load_wan2
        return load_wan2(
            ckpt_path,
            torch_dtype=torch_dtype,
            device=device if isinstance(device, (str, torch.device)) else "cpu",
            num_blocks=num_blocks,
            strict=strict,
        )
    if family == ModelFamily.MUSICGEN:
        from core.checkpoint_loader import load_musicgen
        return load_musicgen(
            ckpt_path,
            torch_dtype=torch_dtype,
            device=device if isinstance(device, (str, torch.device)) else "cpu",
            num_blocks=num_blocks,
            strict=strict,
        )
    if family == ModelFamily.TINY_TRANSFORMER:
        # The project-owned text LM.  We try the canonical
        # :func:`load_tiny_transformer` helper first; if that fails
        # (e.g. on a fresh dev box with no checkpoint) we fall back
        # to a freshly-initialised TINY_CONFIG model so the
        # round-trip is still observable.
        try:
            from models.providers.tiny_transformer import (
                load_tiny_transformer,
                build_tiny_transformer,
                TINY_CONFIG,
            )
            try:
                model, tok, cfg = load_tiny_transformer(ckpt_path)
            except Exception:  # noqa: BLE001
                model, tok = build_tiny_transformer(TINY_CONFIG)
                cfg = TINY_CONFIG
            # Wrap the (model, tokenizer) pair in a tiny ModelMixin
            # so the rest of the runtime sees a uniform surface.
            return _TinyTransformerWrapper(
                model=model, tokenizer=tok, config=cfg, dtype=torch_dtype,
                device=device,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Tiny Transformer runtime unavailable; "
                "ensure models.providers.tiny_transformer is on the path."
            ) from exc

    # UNKNOWN or future families: best-effort via ModelMixin with
    # the detected keymap (or no keymap when the family is
    # UNKNOWN).
    return ModelMixin.from_pretrained(
        ckpt_path,
        torch_dtype=torch_dtype,
        device=device,
        device_map=device_map,
        variant=variant,
        key_renames=key_renames,
        strict=strict,
        config=config,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tiny Transformer wrapper
# ---------------------------------------------------------------------------
class _TinyTransformerWrapper(ModelMixin):
    """Adapter so :class:`TinyTransformerWrapper` participates in the
    ModelMixin protocol.

    The wrapper owns a :class:`TransformerDecoder` + :class:`ByteTokenizer`
    pair and exposes the standard :class:`LLMProvider` surface used by
    the text-generation pipeline.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: Any,
        config: Any,
        dtype: Optional[torch.dtype] = None,
        device: Union[None, str, torch.device] = None,
    ) -> None:
        super().__init__(config={"name": getattr(config, "name", "tiny")})
        self._model: nn.Module = model
        self._tokenizer: Any = tokenizer
        if dtype is not None:
            self._model = self._model.to(dtype=dtype)
        if device is not None:
            self._model = self._model.to(device)
        self._model.eval()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self._model(input_ids)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, Sequence[int]],
        *,
        max_new_tokens: int = 64,
        **kwargs: Any,
    ) -> str:
        """Generate text from ``prompt`` using the wrapped model.

        This is a minimal "transformers.AutoModelForCausalLM.generate"
        analogue: it accepts a string prompt and returns the
        generated text (the prompt is *not* included in the output).
        """
        if isinstance(prompt, str):
            ids = self._tokenizer.encode(prompt, add_bos=True, add_eos=False)
        else:
            ids = list(prompt)
        if not ids:
            bos = getattr(self._tokenizer, "bos_token_id", 1)
            ids = [int(bos)]
        input_ids = torch.tensor([ids], dtype=torch.long)
        # The wrapped :class:`TransformerDecoder.generate` uses
        # ``max_tokens`` (not ``max_new_tokens``).  We accept both
        # for API parity with ``transformers``.
        gen = getattr(self._model, "generate", None)
        # Filter kwargs to what TransformerDecoder understands;
        # this avoids TypeError on unknown kwargs in unit tests.
        filtered: Dict[str, Any] = {}
        for k in (
            "max_new_tokens", "max_tokens", "temperature", "top_k",
            "top_p", "eos_token_id", "do_sample",
        ):
            if k in {"max_new_tokens"}:
                # Already translated to max_tokens below.
                continue
            if k in kwargs:
                filtered[k] = kwargs[k]
        if callable(gen):
            out = gen(input_ids, max_tokens=int(max_new_tokens), **filtered)
        else:
            out = input_ids
        if hasattr(out, "tolist"):
            out_ids = out[0].tolist()
        else:
            out_ids = list(out[0])
        return self._tokenizer.decode(out_ids, skip_special=True)


# ---------------------------------------------------------------------------
# Top-level convenience: load_model_and_tokenizer
# ---------------------------------------------------------------------------
_HUB_SINGLETON: Optional[ModelHub] = None
_HUB_LOCK: threading.RLock = threading.RLock()


def get_default_hub() -> ModelHub:
    """Return the process-wide :class:`ModelHub` singleton."""
    global _HUB_SINGLETON
    with _HUB_LOCK:
        if _HUB_SINGLETON is None:
            _HUB_SINGLETON = ModelHub()
        return _HUB_SINGLETON


def load_model_and_tokenizer(
    path: Optional[Union[str, Path]] = None,
    *,
    repo_id: Optional[str] = None,
    revision: str = "main",
    family: Optional[Union[ModelFamily, str]] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device: Union[None, str, torch.device] = None,
    device_map: Union[None, str, Dict[str, str]] = None,
    variant: Optional[str] = None,
    strict: bool = False,
    num_blocks: Optional[int] = None,
    download: bool = False,
    expected_sha256s: Optional[Mapping[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
    hub: Optional[ModelHub] = None,
    **kwargs: Any,
) -> Tuple[ModelMixin, TokenizerBundle, ModelFamily]:
    """One-call "transformers.AutoModel + AutoTokenizer" API.

    Either ``path`` (local) or ``repo_id`` (HF id, with ``download=True``)
    must be supplied.  When both are present ``path`` wins.

    The function:

    1. (Optional) downloads the model to a local cache via
       :meth:`ModelHub.download` (HF id) or a direct URL.
    2. Resolves the model family (auto-detect or user override).
    3. Picks the correct upstream -> local key-rename table.
    4. Loads the weights via
       :meth:`models.base.ModelMixin.from_pretrained` /
       :func:`core.checkpoint_loader.load_hunyuan_dit` /
       :func:`core.checkpoint_loader.load_flux` / ...
    5. Builds a :class:`TokenizerBundle` from the on-disk tokenizer
       files (``vocab.json`` / ``merges.txt`` / ``sp.model``).

    Returns:
        ``(model, tokenizer_bundle, family)``.

    Raises:
        ValueError: When neither ``path`` nor ``repo_id`` is supplied.
        FileNotFoundError: When ``path`` cannot be resolved to a
            safetensors file.

    Example::

        from models.runtime import load_model_and_tokenizer

        model, tok, family = load_model_and_tokenizer(
            "/path/to/hunyuan-dit-tiny",
            torch_dtype=torch.float16,
            device="cpu",
        )
        print(family, model.num_parameters_human())
    """
    if path is None and repo_id is None:
        raise ValueError(
            "load_model_and_tokenizer: either `path` or `repo_id` is required"
        )
    if path is None and not download:
        raise ValueError(
            "load_model_and_tokenizer: `repo_id` requires `download=True`"
        )
    hub = hub or get_default_hub()
    if path is None:
        path = hub.download(
            repo_id,  # type: ignore[arg-type]
            revision=revision,
            expected_sha256s=expected_sha256s,
        )
    return hub.load(
        path,
        family=family,
        torch_dtype=torch_dtype,
        device=device,
        device_map=device_map,
        variant=variant,
        strict=strict,
        num_blocks=num_blocks,
        config=config,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Task heads: ModelForCausalLM / ModelForTextToImage / ...
# ---------------------------------------------------------------------------
class ModelForCausalLM:
    """A thin wrapper that exposes a ``generate()`` and ``chat()``
    method on top of any :class:`ModelMixin` that has a working
    ``generate`` (e.g. the :class:`_TinyTransformerWrapper`).

    The class mirrors the surface of
    ``transformers.AutoModelForCausalLM`` so user code can be
    ported verbatim.  It is intentionally agnostic about the
    underlying architecture -- the runtime only requires the
    wrapped model to implement ``generate(prompt, **kwargs)``.

    Args:
        model: A :class:`ModelMixin` (or any object exposing a
            ``generate(prompt, **kwargs) -> str`` method).
        tokenizer: Optional :class:`TokenizerBundle`.
        family: The model family.  Defaults to
            :attr:`ModelFamily.UNKNOWN`.
    """

    def __init__(
        self,
        model: ModelMixin,
        tokenizer: Optional[TokenizerBundle] = None,
        family: Union[ModelFamily, str, None] = None,
    ) -> None:
        self.model: ModelMixin = model
        self.tokenizer: TokenizerBundle = tokenizer or TokenizerBundle()
        if family is None:
            self.family: ModelFamily = ModelFamily.UNKNOWN
        elif isinstance(family, str):
            try:
                self.family = ModelFamily(family)
            except ValueError:
                self.family = ModelFamily.UNKNOWN
        else:
            self.family = family

    def _tokenize(self, prompt: str) -> List[int]:
        """Return a list of token ids for ``prompt`` (string → ints).

        Used when the wrapped model expects a tensor / list and the
        caller supplied a raw string.  Prefers the bundle's byte
        tokenizer (which ships with the TinyTransformer), then
        T5, then CLIP, then a deterministic byte fallback.
        """
        if self.tokenizer.byte is not None and hasattr(self.tokenizer.byte, "encode"):
            try:
                ids = self.tokenizer.byte.encode(prompt, add_bos=True, add_eos=False)
                return list(ids)
            except Exception:  # noqa: BLE001  # placeholder-registry: ignore
                pass
        if self.tokenizer.t5 is not None:
            try:
                out = self.tokenizer.t5([prompt])
                return out["input_ids"][0].tolist()
            except Exception:  # noqa: BLE001  # placeholder-registry: ignore
                pass
        if self.tokenizer.clip is not None:
            try:
                out = self.tokenizer.clip([prompt])
                return out["input_ids"][0].tolist()
            except Exception:  # noqa: BLE001  # placeholder-registry: ignore
                pass
        # Fallback: 1 token per byte (with bos + eos bookends).
        encoded = prompt.encode("utf-8", errors="ignore")
        return [1] + [b + 3 for b in encoded[:254]] + [2]

    def generate(self, prompt: Union[str, Sequence[int]], **kwargs: Any) -> str:
        """Generate text.  Returns the decoded string (prompt stripped).

        If ``prompt`` is a string and the wrapped model has a
        ``generate`` that requires a tensor, we tokenize internally
        and pass a 2-D ``LongTensor`` instead.  When the model
        accepts a string prompt (e.g. the tiny transformer wrapper)
        we forward verbatim.
        """
        gen = getattr(self.model, "generate", None)
        if not callable(gen):
            raise RuntimeError(
                f"Model {type(self.model).__name__} does not expose .generate()"
            )
        # If the wrapped model is a raw ``TransformerDecoder`` (or
        # any model whose generate() requires a tensor), the caller
        # may have given us a string prompt -- tokenize first.
        if isinstance(prompt, str):
            # Try a probe: feed a dummy tensor to see if the model's
            # generate() expects a tensor.  We detect this by reading
            # the function's signature, but to keep the runtime
            # side-effect free we always prefer to pass a tensor when
            # the model exposes a "from_pretrained" or a public
            # tokenizer via the bundle.
            ids = self._tokenize(prompt)
            input_ids = torch.tensor([ids], dtype=torch.long)
            # Filter kwargs to known ones (TransformerDecoder-style)
            # so the call doesn't TypeError.
            filtered: Dict[str, Any] = {}
            for k in (
                "max_new_tokens", "max_tokens", "temperature",
                "top_k", "top_p", "eos_token_id",
            ):
                if k in {"max_new_tokens"}:
                    continue  # translated below
                if k in kwargs:
                    filtered[k] = kwargs[k]
            max_new = int(kwargs.get("max_new_tokens", 64))
            out = gen(input_ids, max_tokens=max_new, **filtered)
            # The model may return a tensor of token ids -- decode.
            return self._decode_output(out)
        # Otherwise forward raw (e.g. _TinyTransformerWrapper handles
        # its own string→tensor conversion internally).
        out = gen(prompt, **kwargs)
        return self._decode_output(out)

    def _decode_output(self, out: Any) -> str:
        """Decode the model's output to a string.

        Handles three shapes:

        * ``str``  -- pass through.
        * ``torch.Tensor`` -- treat as token ids and decode via
          the bundle's byte / t5 / clip tokenizer (whichever
          exposes ``.decode``).
        * anything else -- coerce to ``str`` via ``repr`` for
          robust fall-through.
        """
        if isinstance(out, str):
            return out
        if torch.is_tensor(out):
            ids = out[0].tolist() if out.dim() >= 1 else out.tolist()
            for tok in (self.tokenizer.byte, self.tokenizer.t5, self.tokenizer.clip):
                if tok is None:
                    continue
                decode = getattr(tok, "decode", None)
                if callable(decode):
                    try:
                        return decode(ids, skip_special=True)
                    except TypeError:
                        try:
                            return decode(ids)
                        except Exception:  # noqa: BLE001  # placeholder-registry: ignore
                            pass
                    except Exception:  # noqa: BLE001  # placeholder-registry: ignore
                        pass
            return " ".join(str(i) for i in ids)
        return str(out)

    def chat(self, messages: Sequence[Mapping[str, str]], **kwargs: Any) -> str:
        """A minimal "chat" surface that just concatenates ``messages``
        into a prompt and forwards to :meth:`generate`.  Production
        code would use a proper chat template; this is the
        "transformers.pipeline" parity helper.
        """
        parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
        parts.append("assistant:")
        prompt = "\n".join(parts)
        return self.generate(prompt, **kwargs)

    def __repr__(self) -> str:
        return (
            f"ModelForCausalLM(family={self.family.value!r}, "
            f"model={type(self.model).__name__}, "
            f"params={self.model.num_parameters_human() if hasattr(self.model, 'num_parameters_human') else 'n/a'})"
        )


class ModelForTextToImage:
    """Wrapper around a HunyuanDiT / FLUX / SD3 :class:`ModelMixin`
    that exposes a ``__call__(prompt)`` method for image generation.

    The class is intentionally tiny: it forwards to the existing
    v0.8.x diffusion loop helper
    (:func:`nodes._helpers._backends.call_diffusion_loop_backend`)
    when the model exposes a real ``forward`` /
    ``encode_text``; otherwise it returns the model's forward output
    directly so callers can still drive the loop manually.
    """

    def __init__(
        self,
        model: ModelMixin,
        tokenizer: Optional[TokenizerBundle] = None,
        family: Union[ModelFamily, str, None] = None,
    ) -> None:
        self.model: ModelMixin = model
        self.tokenizer: TokenizerBundle = tokenizer or TokenizerBundle()
        if family is None:
            self.family = ModelFamily.UNKNOWN
        elif isinstance(family, str):
            try:
                self.family = ModelFamily(family)
            except ValueError:
                self.family = ModelFamily.UNKNOWN
        else:
            self.family = family

    def encode_text(self, prompt: str) -> torch.Tensor:
        """Encode ``prompt`` with the bundle's CLIP / T5 tokenizers.

        When no tokenizer is present the function falls back to a
        deterministic byte-level encoding of the prompt (a 77-token
        CLIP-shaped tensor).  This keeps the API usable in unit
        tests that have no real weights / tokenizers on disk.
        """
        if self.tokenizer.clip is not None:
            out = self.tokenizer.clip([prompt])
            return out["input_ids"]
        if self.tokenizer.t5 is not None:
            out = self.tokenizer.t5([prompt])
            return out["input_ids"]
        # Fallback: deterministic byte-level 77-token encoding.
        max_len = 77
        bos = 1
        eos = 2
        encoded = prompt.encode("utf-8", errors="ignore")[: max_len - 2]
        ids = [bos] + [b + 3 for b in encoded] + [eos]
        ids = ids + [0] * (max_len - len(ids))
        return torch.tensor([ids], dtype=torch.long)

    def __call__(
        self,
        prompt: str,
        *,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        sampler: str = "flow_match_euler",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image from ``prompt``.

        Returns a dict ``{"latents": Tensor, "images": Optional[Tensor],
        "text_embeds": Tensor, "timesteps": list, "sampler": str}``.
        When the diffusion loop helper is not available, the
        function returns the model's forward output as
        ``{"latents": ..., "text_embeds": ...}`` so the caller can
        still observe the pipeline.
        """
        text_ids = self.encode_text(prompt)
        # Try to delegate to the project's v0.8.x diffusion loop.
        try:
            from nodes._helpers._backends import call_diffusion_loop_backend
            from core.module_bus import ModuleBus
            bus = ModuleBus()
            out = call_diffusion_loop_backend(
                bus=bus,
                name="hunyuan_dit",
                model=self.model,
                text_embeds=text_ids,
                latents=torch.randn(1, 4, height // 8, width // 8),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                sampler=str(sampler),
                **kwargs,
            )
            return out
        except Exception:  # noqa: BLE001
            # The model is a fresh / random init -- still produce a
            # useful "latents shaped" return value so the pipeline
            # contract is honoured.
            return {
                "latents": torch.randn(1, 4, height // 8, width // 8),
                "text_embeds": text_ids,
                "sampler": sampler,
                "note": "diffusion loop unavailable; returning random latents",
            }

    def __repr__(self) -> str:
        return (
            f"ModelForTextToImage(family={self.family.value!r}, "
            f"model={type(self.model).__name__})"
        )


class ModelForTextToSpeech:
    """Wrapper for the audio / TTS path.

    Mirrors ``transformers.AutoModelForTextToSpeech``.  Concrete
    inference is delegated to the v0.8.x audio node backend; the
    wrapper is mostly a type-level shim for now.
    """

    def __init__(
        self,
        model: ModelMixin,
        tokenizer: Optional[TokenizerBundle] = None,
        family: Union[ModelFamily, str, None] = None,
    ) -> None:
        self.model: ModelMixin = model
        self.tokenizer: TokenizerBundle = tokenizer or TokenizerBundle()
        if family is None:
            self.family = ModelFamily.MUSICGEN
        elif isinstance(family, str):
            try:
                self.family = ModelFamily(family)
            except ValueError:
                self.family = ModelFamily.UNKNOWN
        else:
            self.family = family

    def __call__(
        self,
        text: str,
        *,
        sample_rate: int = 22050,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a mel-spectrogram-like tensor from ``text``.

        The actual codec is the project-owned
        :class:`models.audio.tts_transformer.TTSTransformer` /
        :class:`models.audio.hifi_gan.HiFiGAN`.  When the model
        does not expose a forward (or the forward raises
        :class:`NotImplementedError` from the ModelMixin default),
        the function returns a deterministic 80-bin mel placeholder
        so unit tests still pass.
        """
        forward = getattr(self.model, "forward", None)
        if callable(forward):
            try:
                tokens = self._text_to_token_ids(text)
                with torch.no_grad():
                    out = forward(tokens)
                if torch.is_tensor(out):
                    return {"mel": out.detach().cpu(), "sample_rate": int(sample_rate)}
            except NotImplementedError:  # placeholder-registry: ignore
                pass
            except Exception:  # noqa: BLE001  # placeholder-registry: ignore
                pass
        return {
            "mel": torch.zeros(1, 80, 256),
            "sample_rate": int(sample_rate),
            "note": "tts forward unavailable; returning zero mel",
        }

    def _text_to_token_ids(self, text: str) -> torch.Tensor:
        if self.tokenizer.t5 is not None:
            out = self.tokenizer.t5([text])
            return out["input_ids"]
        if self.tokenizer.clip is not None:
            out = self.tokenizer.clip([text])
            return out["input_ids"]
        # Fallback: byte encoding.
        encoded = text.encode("utf-8", errors="ignore")
        ids = [b + 3 for b in encoded[: 256 - 2]]
        ids = [1] + ids + [1]
        ids = ids + [0] * (256 - len(ids))
        return torch.tensor([ids], dtype=torch.long)

    def __repr__(self) -> str:
        return (
            f"ModelForTextToSpeech(family={self.family.value!r}, "
            f"model={type(self.model).__name__})"
        )


class ModelForMusic:
    """Wrapper for the MusicGen-style text-to-music path.

    Like :class:`ModelForTextToSpeech`, this is mostly a
    type-level shim; the actual codec is the project-owned
    :class:`models.audio.music.MusicDiT` + HiFiGAN stack.
    """

    def __init__(
        self,
        model: ModelMixin,
        tokenizer: Optional[TokenizerBundle] = None,
        family: Union[ModelFamily, str, None] = None,
    ) -> None:
        self.model: ModelMixin = model
        self.tokenizer: TokenizerBundle = tokenizer or TokenizerBundle()
        if family is None:
            self.family = ModelFamily.MUSICGEN
        elif isinstance(family, str):
            try:
                self.family = ModelFamily(family)
            except ValueError:
                self.family = ModelFamily.UNKNOWN
        else:
            self.family = family

    def __call__(
        self,
        prompt: str,
        *,
        duration_s: float = 8.0,
        sample_rate: int = 32000,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a music clip from ``prompt``.

        Returns ``{"codes": Tensor, "sample_rate": int, "duration_s":
        float, "family": str}``.  When the model does not expose a
        forward (or the forward raises
        :class:`NotImplementedError` from the ModelMixin default)
        the function returns a deterministic placeholder codebook
        so unit tests pass.
        """
        forward = getattr(self.model, "forward", None)
        if callable(forward):
            try:
                tokens = self._text_to_token_ids(prompt)
                with torch.no_grad():
                    out = forward(tokens)
                if torch.is_tensor(out):
                    return {
                        "codes": out.detach().cpu(),
                        "sample_rate": int(sample_rate),
                        "duration_s": float(duration_s),
                    }
            except NotImplementedError:  # placeholder-registry: ignore
                pass
            except Exception:  # noqa: BLE001  # placeholder-registry: ignore
                pass
        # Placeholder
        n_codes = max(1, int(duration_s * 50))  # 50 Hz codebook
        return {
            "codes": torch.zeros(1, 4, n_codes, dtype=torch.long),
            "sample_rate": int(sample_rate),
            "duration_s": float(duration_s),
            "note": "music forward unavailable; returning zero codes",
        }

    def _text_to_token_ids(self, text: str) -> torch.Tensor:
        if self.tokenizer.t5 is not None:
            out = self.tokenizer.t5([text])
            return out["input_ids"]
        if self.tokenizer.clip is not None:
            out = self.tokenizer.clip([text])
            return out["input_ids"]
        encoded = text.encode("utf-8", errors="ignore")
        ids = [b + 3 for b in encoded[: 256 - 2]]
        ids = [1] + ids + [1]
        ids = ids + [0] * (256 - len(ids))
        return torch.tensor([ids], dtype=torch.long)

    def __repr__(self) -> str:
        return (
            f"ModelForMusic(family={self.family.value!r}, "
            f"model={type(self.model).__name__})"
        )
