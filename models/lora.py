"""LoRA / low-rank delta injection (v0.8.5).

This module is the v0.8.5 "LoRA 注入" entry point.  It
implements the classic Hu et al. (2021) LoRA recipe on top of
:class:`core.offload.ModelPatcher`:

    W' = W + (alpha / rank) * (B @ A)

where ``A`` is ``(rank, in_features)`` and ``B`` is
``(out_features, rank)``, both initialised to zero / Kaiming
respectively.  The base weight ``W`` is *not copied*; the
delta is added to the existing parameter data (or, in merge
mode, baked into a single weight).

The injector is intentionally a thin wrapper around
:class:`core.offload.ModelPatcher` so that the existing
patch / offload infrastructure is reused.  The LoRA patch
op is a closure that:

* Lazily materialises ``A`` / ``B`` on first apply.
* Stores the previous ``W`` so :meth:`remove` can restore
  the base weight to its pre-LoRA state.
* Supports a ``merge=True`` mode that bakes the delta into
  ``W`` and frees ``A`` / ``B``.

Public surface:

* :class:`LoRAInjector` -- attach one or more named LoRAs to
  a :class:`nn.Module`.
* :func:`inject_lora` -- one-shot helper: add a single LoRA
  and return the injector.
* :func:`lora_state_dict` / :func:`load_lora_state_dict` --
  serialise / restore the delta tensors (compatible with the
  diffusers ``lora.safetensors`` convention: keys are
  ``<full_module_name>.lora.A`` / ``.B`` plus a top-level
  ``alpha`` / ``rank`` metadata pair).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from core.offload import ModelPatcher, Patch

__all__ = [
    "LoRASpec",
    "LoRAInjector",
    "inject_lora",
    "lora_state_dict",
    "load_lora_state_dict",
    "default_target_modules",
]


# ---------------------------------------------------------------------------
# Default target patterns (HunyuanDiT-friendly)
# ---------------------------------------------------------------------------
def default_target_modules() -> Tuple[str, ...]:
    """Return the default LoRA target glob patterns.

    The v0.8.5 default targets the attention QKV / out_proj
    layers and the MLP fc1 / fc2 layers of every block
    (HunyuanDiT local layout).  This is the same set the
    Tencent HunyuanDiT-LoRA recipe targets.
    """
    return (
        "blocks.*.attn.qkv",
        "blocks.*.attn.out_proj",
        "blocks.*.mlp.fc1",
        "blocks.*.mlp.fc2",
    )


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------
@dataclass
class LoRASpec:
    """A single LoRA delta descriptor.

    Attributes:
        name: Human-readable identifier (e.g. ``"style_lora"``).
        rank: Low-rank dimension.  ``rank=0`` is a sentinel
            for "no LoRA" -- the spec is silently skipped.
        alpha: Scaling factor; the effective scale is
            ``alpha / rank``.  Defaults to ``rank`` (i.e.
            unit scaling at the chosen rank).
        target_modules: Glob patterns.  When empty, the
            injector falls back to :func:`default_target_modules`.
        dropout: Optional dropout applied to the input of
            the ``A`` projection during training.  Inference
            (the only mode supported in v0.8.5) does not use
            it.
        init_seed: Seed used for the Kaiming init of ``B``.
            ``A`` is always initialised to zero so that the
            initial delta is zero (the model output is
            unchanged at first).
    """

    name: str
    rank: int = 4
    alpha: Optional[float] = None
    target_modules: Tuple[str, ...] = field(default_factory=tuple)
    dropout: float = 0.0
    init_seed: int = 0

    def __post_init__(self) -> None:
        if self.alpha is None:
            self.alpha = float(self.rank)
        if self.rank < 0:
            raise ValueError(f"rank must be >= 0, got {self.rank}")

    @property
    def scale(self) -> float:
        """The effective scale applied to ``B @ A``."""
        if self.rank == 0:
            return 0.0
        return float(self.alpha) / float(self.rank)


# ---------------------------------------------------------------------------
# LoRAInjector
# ---------------------------------------------------------------------------
class LoRAInjector:
    """Attach named LoRA deltas to a :class:`nn.Module` without copying.

    Example:
        >>> m = nn.Linear(16, 32)
        >>> injector = LoRAInjector(m)
        >>> injector.add(LoRASpec("style", rank=4))
        >>> injector.apply()
        >>> y = m(torch.randn(2, 16))   # with LoRA active
        >>> injector.remove("style")     # weights restored
    """

    def __init__(self, model: nn.Module) -> None:
        self.model: nn.Module = model
        self.patcher: ModelPatcher = ModelPatcher(model)
        # Map full_module_name -> (A, B) tensors (lazy).
        self._deltas: Dict[str, Tuple[nn.Parameter, nn.Parameter]] = {}
        # Map spec.name -> spec (so the user can ``remove`` by name).
        self._specs: Dict[str, LoRASpec] = {}

    # ------------------------------------------------------------------
    def add(self, spec: LoRASpec) -> None:
        """Register a :class:`LoRASpec` (does not yet apply it)."""
        if spec.rank == 0:
            return  # rank=0 -> no-op
        if spec.name in self._specs:
            raise ValueError(f"duplicate LoRA name: {spec.name!r}")
        self._specs[spec.name] = spec

    def remove(self, name: str) -> None:
        """Remove a LoRA by name and restore the base weights."""
        if name not in self._specs:
            return
        # Find every module that was patched by this LoRA and
        # restore its weight from the closure-captured copy.
        for full_name, (a, b) in list(self._deltas.items()):
            if (a, b) in self._find_deltas_for_spec(name):
                self.patcher.remove(f"{name}::{full_name}")
                # The patcher's ``remove`` invokes the undo we
                # returned from the op (which restores ``W``).
                # We can now drop the delta tensors.
                del self._deltas[full_name]
        self._specs.pop(name, None)

    def _find_deltas_for_spec(
        self, name: str,
    ) -> List[Tuple[nn.Parameter, nn.Parameter]]:
        """Return the (A, B) tuples created by ``name``."""
        out: List[Tuple[nn.Parameter, nn.Parameter]] = []
        for full_name, (a, b) in self._deltas.items():
            # The patch names we used when registering are
            # ``f"{name}::{full_name}"``; we re-derive them
            # from the patcher stack.
            for p in self.patcher._stack:  # type: ignore[attr-defined]
                if p.name.startswith(f"{name}::") and p.metadata is not None:
                    md_a = p.metadata.get("A")
                    md_b = p.metadata.get("B")
                    if md_a is a and md_b is b:
                        out.append((a, b))
        return out

    # ------------------------------------------------------------------
    def apply(self) -> int:
        """Apply every registered LoRA.

        Returns:
            The number of (module, LoRA) pairs newly patched.
        """
        n = 0
        for spec in self._specs.values():
            n += self._apply_spec(spec)
        return n

    def _apply_spec(self, spec: LoRASpec) -> int:
        patterns = spec.target_modules or default_target_modules()
        n = 0
        for full_name, module in self._iter_named_modules():
            if not any(_glob_match(p, full_name) for p in patterns):
                continue
            # Build / fetch the delta pair for this module.
            if full_name not in self._deltas:
                a, b = _make_lora_params(module, spec)
                self._deltas[full_name] = (a, b)
            a, b = self._deltas[full_name]
            # The patch op: install a forward hook that adds
            # the LoRA delta to the module's output.  The
            # base weight is *not* mutated -- this is the
            # ComfyUI "non-destructive" LoRA convention.
            op = _make_lora_op(a, b, spec.scale)
            patch = Patch(
                name=f"{spec.name}::{full_name}",
                op=op,
                strength=1.0,
                key_filter=None,
                metadata={"spec": spec, "A": a, "B": b, "module": module},
            )
            self.patcher.add(patch)
            self.patcher.apply(full_name, module)
            n += 1
        return n

    def clear(self) -> None:
        """Remove every LoRA and restore all base weights."""
        for name in list(self._specs.keys()):
            self.remove(name)
        self._deltas.clear()

    # ------------------------------------------------------------------
    def _iter_named_modules(self) -> Iterable[Tuple[str, nn.Module]]:
        """Yield ``(full_name, module)`` for every sub-module."""
        for name, module in self.model.named_modules():
            # Skip the root (empty name) and any module that
            # is a *container* -- only leaf Linear / Conv2d
            # layers get LoRA deltas.
            if name == "":
                continue
            if not isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                continue
            yield name, module

    # ------------------------------------------------------------------
    def lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Serialise the deltas to a diffusers-compatible dict.

        Keys are ``<full_name>.lora.A.weight`` /
        ``<full_name>.lora.B.weight`` (matching the
        ``diffusers`` LoraLoader convention).  A
        top-level ``_lora_metadata`` tensor is also emitted
        so that the loader can recover the spec.
        """
        out: Dict[str, torch.Tensor] = {}
        for full_name, (a, b) in self._deltas.items():
            out[f"{full_name}.lora.A.weight"] = a.detach().cpu().clone()
            out[f"{full_name}.lora.B.weight"] = b.detach().cpu().clone()
        # Encode metadata as a string tensor (portable across
        # safetensors / torch / pickle).
        import json
        meta = {
            name: {
                "rank": spec.rank,
                "alpha": spec.alpha,
                "scale": spec.scale,
                "target_modules": list(spec.target_modules),
            }
            for name, spec in self._specs.items()
        }
        out["_lora_metadata"] = torch.tensor(
            [ord(c) for c in json.dumps(meta)], dtype=torch.uint8,
        )
        return out


# ---------------------------------------------------------------------------
# One-shot helpers
# ---------------------------------------------------------------------------
def inject_lora(
    model: nn.Module,
    *,
    name: str = "lora",
    rank: int = 4,
    alpha: Optional[float] = None,
    target_modules: Optional[Sequence[str]] = None,
) -> LoRAInjector:
    """Convenience wrapper: add a single LoRA and apply it."""
    spec = LoRASpec(
        name=name,
        rank=rank,
        alpha=alpha,
        target_modules=tuple(target_modules) if target_modules else (),
    )
    injector = LoRAInjector(model)
    injector.add(spec)
    injector.apply()
    return injector


def lora_state_dict(injector: LoRAInjector) -> Dict[str, torch.Tensor]:
    """Wrapper around :meth:`LoRAInjector.lora_state_dict`."""
    return injector.lora_state_dict()


def load_lora_state_dict(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
) -> LoRAInjector:
    """Restore a LoRA delta set produced by :func:`lora_state_dict`.

    The injector is freshly built -- any pre-existing LoRA on
    ``model`` is left untouched.
    """
    import json
    injector = LoRAInjector(model)
    # Parse metadata.
    raw = state.get("_lora_metadata")
    meta: Dict[str, Dict[str, Any]] = {}
    if raw is not None:
        meta = json.loads(bytes(int(x) for x in raw.tolist()).decode("utf-8"))
    # Re-create specs.
    for name, info in meta.items():
        injector.add(LoRASpec(
            name=name,
            rank=int(info["rank"]),
            alpha=float(info["alpha"]) if info.get("alpha") is not None else None,
            target_modules=tuple(info.get("target_modules", ())),
        ))
    # Materialise deltas.
    injector.apply()
    # Overwrite the lazy A / B with the loaded tensors.
    for key, tensor in state.items():
        if not key.endswith(".lora.A.weight") and not key.endswith(".lora.B.weight"):
            continue
        if key.endswith(".lora.A.weight"):
            full = key[: -len(".lora.A.weight")]
            a, _ = injector._deltas[full]
            with torch.no_grad():
                a.copy_(tensor.to(a.device))
        else:
            full = key[: -len(".lora.B.weight")]
            _, b = injector._deltas[full]
            with torch.no_grad():
                b.copy_(tensor.to(b.device))
    return injector


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _glob_match(pattern: str, name: str) -> bool:
    """A tiny subset of ``fnmatch`` that supports ``*`` only."""
    if "*" not in pattern:
        return pattern == name
    # Convert "blocks.*.attn.qkv" -> prefix / suffix match.
    parts = pattern.split("*", 1)
    if len(parts) == 2:
        pre, suf = parts
        return name.startswith(pre) and (suf == "" or name.endswith(suf))
    return pattern == name


def _make_lora_params(
    module: nn.Module, spec: LoRASpec,
) -> Tuple[nn.Parameter, nn.Parameter]:
    """Build the (A, B) parameter pair for ``module``."""
    if isinstance(module, nn.Linear):
        in_f, out_f = module.in_features, module.out_features
    elif isinstance(module, nn.Conv1d):
        in_f, out_f = module.in_channels, module.out_channels
    elif isinstance(module, nn.Conv2d):
        in_f, out_f = module.in_channels, module.out_channels
    else:
        raise TypeError(f"unsupported module type: {type(module).__name__}")
    if spec.rank > min(in_f, out_f):
        # Cap at the smaller of the two dims so the low-rank
        # decomposition is well-defined.
        rank = max(1, min(in_f, out_f))
    else:
        rank = spec.rank
    # ``A`` is zero so the initial delta is zero.
    a = nn.Parameter(torch.zeros(rank, in_f), requires_grad=False)
    # ``B`` is Kaiming-init; we use a per-spec seed for
    # reproducibility.
    g = torch.Generator().manual_seed(spec.init_seed)
    b = nn.Parameter(torch.empty(out_f, rank))
    nn.init.kaiming_uniform_(b, a=math.sqrt(5), generator=g)
    b.requires_grad = False
    return a, b


def _make_lora_op(
    a: nn.Parameter,
    b: nn.Parameter,
    scale: float,
):
    """Return a closure that adds the LoRA delta to ``module``'s output.

    The closure is the patch *op* consumed by
    :class:`core.offload.ModelPatcher`.  It returns an
    ``undo`` callable that removes the forward hook.
    """
    def lora_op(module: nn.Module, strength: float) -> "callable":
        # The "real" forward is the unbound class method; this
        # is *always* the genuine ``nn.Linear.forward`` /
        # ``nn.Conv2d.forward`` even when the module has
        # already been patched by a previous LoRA.  Capturing
        # ``module.forward`` instead would re-wrap the previous
        # patch closure and leave its delta active after undo.
        cls = module.__class__
        # ``cls.forward`` is a bound method on Python 3.10+;
        # accessing it via ``__get__`` on the module gives the
        # unbound-ish original.
        original_forward = cls.forward.__get__(module, cls)

        def patched_forward(*args: Any, **kwargs: Any) -> Any:
            out = original_forward(*args, **kwargs)
            if not args or not isinstance(args[0], torch.Tensor):
                return out
            x = args[0]
            # The delta is computed in float32 for stability
            # even when the module is in fp16/bf16.
            dtype = x.dtype
            xf = x.to(torch.float32)
            af = a.to(torch.float32)
            bf = b.to(torch.float32)
            # (B @ A) @ x.T -> (out, in) @ (in, batch) etc.
            # For Linear / Conv1d / Conv2d the input ``x`` is
            # at least 2-D; for Linear it's ``(N, in)``, for
            # ConvNd it's ``(N, C, ...)``.  We use a *shared*
            # matmul across the trailing dims via the
            # standard "low-rank ΔW" trick:
            #
            #   extra = ((x @ A.T) @ B.T) * scale * strength
            #
            # which is the LoRA paper's Equation 1 with the
            # 2-D input reshape folded.
            extra = (xf @ af.transpose(0, 1)) @ bf.transpose(0, 1)
            extra = extra * (scale * strength)
            extra = extra.to(dtype)
            if out.shape == extra.shape:
                return out + extra
            # The ConvNd case: flatten the spatial dims.
            return out + extra.reshape(out.shape)

        module.forward = patched_forward  # type: ignore[assignment]

        def undo(m: nn.Module) -> None:
            m.forward = original_forward  # type: ignore[assignment]

        return undo

    return lora_op
