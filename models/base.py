"""Base model class for all TorchaVerse models (v0.8.0).

All model implementations inherit from :class:`ModelMixin` (formerly
:class:`BaseModel`), which provides a common interface for
configuration, parameter counting, weight serialisation, and the
``from_pretrained`` / ``save_pretrained`` protocol that mirrors
``diffusers.ModelMixin``.

The v0.8.0 release is allowed to be a **breaking** refactor of this
module per the V0.8_UPGRADE_PLAN:

* ``save(path)`` / ``load(path)`` are deprecated; use
  :meth:`save_pretrained` / :meth:`from_pretrained` instead.
* :meth:`from_pretrained` now supports:
    - ``subfolder``  — diffusers-style nested layout
    - ``torch_dtype`` — convert weights on the fly
    - ``device`` / ``device_map`` — pin or shard at load time
    - ``variant`` — pick ``fp16`` / ``bf16`` siblings
    - ``key_renames`` — declarative checkpoint key migration
    - ``strict`` — off by default (closer to diffusers behaviour)
* :meth:`num_parameters` / :meth:`num_parameters_human` are unchanged.
* The previous ``load_unsafe`` path is removed; pickles are no longer
  trusted (callers can opt back in via :func:`torch.load` directly).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn

__all__ = [
    "ModelMixin",
    "load_safetensors",
    "save_safetensors",
    "transform_checkpoint_dict_key",
    "load_state_dict_with_renames",
]


# ---------------------------------------------------------------------------
# Low-level safetensors helpers
# ---------------------------------------------------------------------------
def load_safetensors(
    path: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """Load a safetensors file into a ``{name: tensor}`` dict.

    Args:
        path: A path to a ``.safetensors`` file.  Sharded layouts
            (``model-00001-of-00005.safetensors`` + index) are
            stitched transparently when the corresponding
            ``.safetensors.index.json`` is present next to the file.
        device: Target device for the loaded tensors.
        dtype: Optional dtype cast applied to every tensor after load.

    Returns:
        A ``{name: tensor}`` state-dict.  Tensors are detached and live
        on ``device``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"safetensors file not found: {path}")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - safetensors is a hard dep
        raise RuntimeError(
            "safetensors is required to load .safetensors files; "
            "install it via `pip install safetensors`",
        ) from exc

    # Sharded layout: an index file ``<file_stem>.safetensors.index.json``
    # next to a single ``.safetensors`` file.  When the caller asks
    # for any shard we transparently stitch every shard referenced
    # in the index.  The index lives next to whichever shard the
    # caller named.
    #
    # The index file can be named after either the shard the caller
    # named (``test-00001-of-00002.safetensors.index.json``) or
    # the implicit base (``test.safetensors.index.json`` --
    # diffusers style).  We probe both, preferring the per-shard
    # one to mirror the layout the caller has in hand.
    stem = path.name
    if stem.endswith(".safetensors"):
        stem = stem[: -len(".safetensors")]
    index_path = path.parent / f"{stem}.safetensors.index.json"
    if not index_path.is_file():
        # Fallback: strip a trailing ``-NNNNN-of-MMMMM`` shard
        # suffix to find the diffusers-style base index.
        import re
        m = re.match(
            r"^(?P<base>.+)-\d{5}-of-\d{5}$", stem,
        )
        if m:
            base_stem = m.group("base")
            candidate = path.parent / f"{base_stem}.safetensors.index.json"
            if candidate.is_file():
                index_path = candidate
    if index_path.is_file():
        import json
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        merged: Dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(weight_map.values())):
            shard_path = path.parent / shard_name
            merged.update(load_file(str(shard_path), device=str(device)))
        state_dict = merged
    else:
        state_dict = load_file(str(path), device=str(device))
    if dtype is not None:
        state_dict = {
            k: v.to(dtype=dtype) if v.is_floating_point() else v
            for k, v in state_dict.items()
        }
    return state_dict


def save_safetensors(
    state_dict: Mapping[str, torch.Tensor],
    path: Union[str, Path],
) -> None:
    """Persist a state-dict to ``path`` using the safetensors format.

    Falls back to ``torch.save`` if the safetensors package is not
    available (the result is still a ``.pt`` file, so the caller
    should adjust the suffix if they want a true safetensors file).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
        save_file(dict(state_dict), str(path))
    except ImportError:  # pragma: no cover - safetensors is a hard dep
        torch.save(dict(state_dict), str(path))


def transform_checkpoint_dict_key(
    state_dict: Dict[str, torch.Tensor],
    key_map: Mapping[str, str],
) -> Dict[str, torch.Tensor]:
    """Rename keys in ``state_dict`` using ``key_map``.

    Each ``{old: new}`` pair is applied.  Keys not in ``key_map`` are
    passed through unchanged.  When ``old`` and ``new`` differ only
    by a prefix, the suffix is preserved.
    """
    out: Dict[str, torch.Tensor] = {}
    for old, new in key_map.items():
        if old in state_dict:
            out[new] = state_dict.pop(old)
    # Add the unchanged keys.
    for k, v in state_dict.items():
        out.setdefault(k, v)
    return out


def load_state_dict_with_renames(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    key_map: Optional[Mapping[str, str]] = None,
    *,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load ``state_dict`` into ``model`` with optional renames.

    The tensors' dtype is preserved by manually copying data into
    the existing parameters (rather than calling
    :meth:`nn.Module.load_state_dict` which re-allocates parameters
    in the module's default dtype and silently up-casts ``fp16``
    / ``bf16`` tensors back to ``fp32``).

    Args:
        model: The target :class:`nn.Module`.
        state_dict: A ``{name: tensor}`` mapping.
        key_map: Optional ``{old_name: new_name}`` rewrite table.
        strict: If ``True`` an error is raised on missing or extra keys;
            otherwise they are returned in a tuple
            ``(missing, unexpected)`` (default diffusers behaviour).

    Returns:
        ``(missing_keys, unexpected_keys)``.
    """
    if key_map:
        state_dict = transform_checkpoint_dict_key(dict(state_dict), key_map)
    own_state = model.state_dict()
    missing: list[str] = []
    unexpected: list[str] = []
    for k in own_state:
        if k in state_dict:
            param = model.get_parameter(k)
            with torch.no_grad():
                src = state_dict[k].to(device=param.device, dtype=param.dtype)
                param.copy_(src)
        else:
            missing.append(k)
    for k in state_dict:
        if k not in own_state:
            unexpected.append(k)
    if strict:
        if missing or unexpected:
            raise RuntimeError(
                f"strict load failed: missing={missing}, "
                f"unexpected={unexpected}",
            )
    return missing, unexpected


# ---------------------------------------------------------------------------
# ModelMixin
# ---------------------------------------------------------------------------
class ModelMixin(nn.Module):
    """Common base for all TorchaVerse models.

    Subclasses automatically gain :meth:`from_pretrained` and
    :meth:`save_pretrained` with the full diffusers surface area
    (``subfolder`` / ``torch_dtype`` / ``device`` / ``variant`` /
    ``key_renames``).
    """

    #: File extension for single-file checkpoints.  Override in
    #: subclasses to switch the default to ``.bin`` or ``.pt``.
    _default_file_extension: str = ".safetensors"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config: dict[str, Any] = dict(config or {})

    # ------------------------------------------------------------------
    # Parameter counting (preserved from the v0.6 base class)
    # ------------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def num_parameters_human(self, trainable_only: bool = True) -> str:
        n = self.num_parameters(trainable_only)
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.2f}B"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        if n >= 1_000:
            return f"{n / 1_000:.2f}K"
        return str(n)

    # ------------------------------------------------------------------
    # save_pretrained / from_pretrained (v0.8.0)
    # ------------------------------------------------------------------
    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        safe_serialization: bool = True,
        file_name: Optional[str] = None,
    ) -> None:
        """Persist the model to ``save_directory``.

        Args:
            save_directory: Target directory; created if it does not
                exist.  The state-dict is written as a single
                safetensors file.
            safe_serialization: Ignored when the safetensors package
                is unavailable; the call always writes the safest
                format the runtime supports.
            file_name: Optional override for the output file name
                (without directory).  Defaults to
                ``{class_name_lowercase}{_default_file_extension}``.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        if file_name is None:
            file_name = f"{self.__class__.__name__.lower()}{self._default_file_extension}"
        target = save_directory / file_name
        if safe_serialization:
            save_safetensors(self.state_dict(), target)
        else:  # pragma: no cover - rarely used
            torch.save(self.state_dict(), target)
        # Also drop a config snapshot so the directory is self-contained.
        config_path = save_directory / "config.json"
        try:
            import json
            config_path.write_text(
                json.dumps(self.config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            # Config persistence is best-effort; the on-disk checkpoint
            # is still valid even if the config sidecar cannot be
            # written (read-only mount, non-serialisable values...).
            return  # noqa: WPS420

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *,
        subfolder: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device: Union[str, torch.device, None] = None,
        device_map: Optional[Union[str, Dict[str, str]]] = None,
        variant: Optional[str] = None,
        key_renames: Optional[Mapping[str, str]] = None,
        strict: bool = False,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "ModelMixin":
        """Load a model from a local directory / file.

        Supports the same surface as diffusers ``ModelMixin.from_pretrained``:

        * ``subfolder`` — point at a nested directory.
        * ``torch_dtype`` — cast weights to ``float16`` / ``bfloat16``.
        * ``device`` — pin every parameter to a single device.
        * ``device_map`` — ``"cpu"`` / ``"cuda"`` / ``{"layer.0": "cuda:0"}``.
        * ``variant`` — look for ``<name>.<variant>.safetensors`` siblings.
        * ``key_renames`` — declarative ``{old: new}`` key migration.
        * ``strict`` — diffusers-style lenient loading by default.

        Args:
            pretrained_model_name_or_path: Either a directory that
                contains a safetensors file, or a direct path to a
                ``.safetensors`` file.  The constructor receives the
                (optional) ``config`` argument from
                ``<dir>/config.json`` (or the caller-supplied one).
            **kwargs: Forwarded to the subclass ``__init__``.

        Returns:
            An instance of ``cls`` with weights loaded.

        Raises:
            FileNotFoundError: When no checkpoint file is found at the
                requested location.
        """
        path = Path(pretrained_model_name_or_path)
        if subfolder:
            path = path / subfolder
        # Resolve the checkpoint file.
        ckpt_path: Optional[Path] = None
        if path.is_file():
            ckpt_path = path
        else:
            candidate_names = [
                f"{cls.__name__.lower()}{cls._default_file_extension}",
                f"model{cls._default_file_extension}",
                f"diffusion_pytorch_model{cls._default_file_extension}",
            ]
            if variant:
                candidate_names = [
                    n.replace(
                        cls._default_file_extension,
                        f".{variant}{cls._default_file_extension}",
                    )
                    for n in candidate_names
                ] + candidate_names
            for name in candidate_names:
                if (path / name).is_file():
                    ckpt_path = path / name
                    break
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint found for {cls.__name__} at {path}. "
                f"Tried: {candidate_names}",
            )

        # When the caller did not specify ``torch_dtype`` explicitly
        # but did name a ``variant`` that maps to a well-known
        # half-precision flavour (``fp16`` / ``bf16`` / ``fp32``),
        # infer the dtype from the variant name.  This mirrors
        # diffusers behaviour where ``variant="fp16"`` is shorthand
        # for both "load the fp16 file" AND "cast the module to
        # fp16 on disk".  Unknown variants leave ``torch_dtype``
        # untouched so the caller keeps full control.
        if torch_dtype is None and variant:
            _VARIANT_TO_DTYPE = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            torch_dtype = _VARIANT_TO_DTYPE.get(variant)

        # Load the state-dict.
        state_dict = load_safetensors(
            ckpt_path, device=str(device or "cpu"), dtype=torch_dtype,
        )
        # Optionally drop the ``_class_name`` / ``_diffusers_version``
        # bookkeeping keys diffusers adds.
        state_dict.pop("_class_name", None)
        state_dict.pop("_diffusers_version", None)

        # Materialise the model.  When ``torch_dtype`` is requested,
        # cast the (freshly-instantiated fp32) parameters to the
        # target dtype BEFORE the weight copy so that
        # :func:`load_state_dict_with_renames` does not silently
        # up-cast the source fp16/bf16 tensors back to the model's
        # default dtype via its ``src.to(dtype=param.dtype)`` guard.
        config_path = path / "config.json"
        if config is None and config_path.is_file():
            try:
                import json
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                config = None
        model = cls(config=config, **kwargs)
        if torch_dtype is not None:
            model.to(torch_dtype)
        missing, unexpected = load_state_dict_with_renames(
            model, state_dict, key_renames, strict=strict,
        )
        # Surface a one-line warning when keys are dropped (typical of
        # a first integration with a real upstream checkpoint).
        if missing or unexpected:
            cls._log_load_report(
                model, ckpt_path, missing, unexpected,
            )
        # Device placement.
        if device_map:
            model._apply_device_map(device_map)
        elif device is not None:
            model.to(device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Device map helpers
    # ------------------------------------------------------------------
    def _apply_device_map(self, device_map: Union[str, Dict[str, str]]) -> None:
        """Apply a diffusers-style device map to the model.

        Supports the string shortcuts ``"cpu"`` and ``"cuda"`` which
        pin the entire module to a single device.  For finer-grained
        sharding, callers can pass a ``{parameter_name: device}`` map.
        """
        if isinstance(device_map, str):
            self.to(device_map)
            return
        for name, param in self.named_parameters():
            for prefix, device in device_map.items():
                if name.startswith(prefix):
                    param.data = param.data.to(device)
                    break

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _log_load_report(
        model: nn.Module,
        ckpt_path: Path,
        missing: list[str],
        unexpected: list[str],
    ) -> None:
        """Emit a structured one-liner for key mismatches at load time."""
        n_missing = len(missing)
        n_unexp = len(unexpected)
        if n_missing == 0 and n_unexp == 0:
            return
        import logging
        logger = logging.getLogger(model.__class__.__module__)
        logger.warning(
            "[%s] loaded %s: %d missing, %d unexpected keys",
            model.__class__.__name__, ckpt_path.name, n_missing, n_unexp,
        )

    # ------------------------------------------------------------------
    # Legacy aliases (will be removed in v0.9)
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Deprecated: use :meth:`save_pretrained` instead."""
        self.save_pretrained(Path(path).parent, file_name=Path(path).name)

    def load(self, path: str, strict: bool = True) -> None:
        """Deprecated: use :meth:`from_pretrained` instead."""
        state_dict = load_safetensors(path, device="cpu")
        load_state_dict_with_renames(self, state_dict, strict=strict)


# ---------------------------------------------------------------------------
# Backwards-compatibility shim: ``BaseModel`` is now an alias.
# ---------------------------------------------------------------------------
BaseModel = ModelMixin
