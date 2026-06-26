"""Runtime offload + patch helpers (v0.8.5).

The v0.8.5 deliverable for :mod:`core.offload` is the *smallest
useful subset* of the ComfyUI ``ModelPatcher`` / diffusers
``enable_model_cpu_offload`` / ``enable_sequential_cpu_offload``
recipe.  The whole module is pure Python + PyTorch, with no
external dependencies, and is designed to be **safe to call on
stock CPU** -- the offload simply degenerates to a no-op when
the model is already on a single device.

Three pieces of public surface:

* :class:`ModelPatcher` -- a thin registry of *named patches*
  attached to a single :class:`nn.Module`.  Each patch is a
  ``(Callable[[nn.Module, float], None], key_filter, strength)``
  triple that is applied on-demand by the patcher.  Base-model
  weights are never copied; patches modify the module in place
  with proper state-restore semantics.

* :func:`enable_model_cpu_offload` -- per-submodule offload.  Each
  leaf :class:`nn.Module` gets a forward pre-hook that moves
  the module to the *compute device* (default ``"cpu"``), and a
  forward post-hook that moves it back to the *offload device*
  (default ``"cpu"``).  When the two devices are the same
  (the common CPU-only case) the hooks are no-ops.

* :func:`enable_sequential_cpu_offload` -- one submodule at a
  time.  All leaf modules start on the offload device; the
  forward pre-hook for each moves the module to compute,
  and a *global* post-hook moves the previous module back
  to offload.  This is the diffusers "stream" mode.

The patcher / offload helpers are also the foundation for
:class:`models.lora.LoRAInjector` (see :mod:`models.lora`),
which uses :class:`ModelPatcher` to apply a low-rank
``\Delta W = (B @ A) * scale`` delta to a chosen set of
``Linear`` / ``Conv2d`` layers without copying the base
weight.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn

__all__ = [
    "Patch",
    "ModelPatcher",
    "enable_model_cpu_offload",
    "enable_sequential_cpu_offload",
    "disable_offload",
]


# ---------------------------------------------------------------------------
# Patch descriptor
# ---------------------------------------------------------------------------
@dataclass
class Patch:
    """A single named patch entry.

    Attributes:
        name: Human-readable identifier (also used as the
            key-filter prefix when ``key_filter`` is ``None``).
        op: Callable that mutates ``module`` in place with
            ``strength`` (a float in ``[0, 1+]``).  LoRA
            patches use ``op=inject_lora_delta`` etc.  The
            op **must** return an ``undo`` callable (or
            ``None`` to signal an irreversible patch).
        strength: Patch scaling factor (``1.0`` = full
            effect).  The op may ignore it for
            non-strength-aware patches.
        key_filter: Optional glob pattern.  When set, the
            patch only applies to modules whose *fully
            qualified* name matches the pattern.  When
            ``None`` the patch applies to every module.
        metadata: Free-form dict for caller bookkeeping
            (e.g. ``{"rank": 4, "alpha": 1.0}``).
    """

    name: str
    op: Callable[[nn.Module, float], Optional[Callable[[nn.Module], None]]]
    strength: float = 1.0
    key_filter: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def matches(self, full_name: str) -> bool:
        if self.key_filter is None:
            return True
        return fnmatch.fnmatchcase(full_name, self.key_filter)


# ---------------------------------------------------------------------------
# ModelPatcher
# ---------------------------------------------------------------------------
class ModelPatcher:
    """A ComfyUI-style hook registry for a single :class:`nn.Module`.

    The patcher is a thin state container.  It does **not**
    copy the base model; instead it stores a list of
    :class:`Patch` entries that are *applied* to the
    module on demand via :meth:`apply`.  Removing a patch
    (:meth:`remove`) reverses the in-place mutation.  The
    patcher is safe to use as a context manager -- the
    ``__exit__`` removes every patch in ``inverse`` order.

    Example:
        >>> m = nn.Linear(8, 8)
        >>> patcher = ModelPatcher(m)
        >>> def zero_out(mod, s): mod.weight.data.zero_()
        >>> patcher.add(Patch("zero", zero_out))
        >>> with patcher:
        ...     _ = m(torch.randn(1, 8))  # all-zero output
        >>> # After the ``with`` block the weights are restored.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model: nn.Module = model
        # Stack of applied patches (most recent at the end).
        self._stack: List[Patch] = []
        # The set of *active* patches.  A patch is active if it
        # is in ``_stack`` and its key_filter matches the
        # queried module name.
        self._names: Dict[str, Patch] = {}

    # ------------------------------------------------------------------
    def add(self, patch: Patch) -> None:
        """Register a patch.  Idempotent on ``patch.name``."""
        if patch.name in self._names:
            raise ValueError(f"duplicate patch name: {patch.name!r}")
        self._names[patch.name] = patch
        # ``_undo`` is filled in by ``apply`` with the callable
        # returned by the patch op.  When the patch is removed
        # *before* being applied, ``_undo`` stays ``None`` and
        # ``remove`` is a no-op for that patch.
        patch._undo = None  # type: ignore[attr-defined]

    def remove(self, name: str) -> None:
        """Remove a patch by name and apply the inverse.

        The inverse is the callable returned by the patch op
        at apply time (see :class:`Patch`).  The op returns a
        closure of ``(Callable[[nn.Module], None])`` that is
        invoked with the **same module** that the op received
        -- not the patcher's root model.  This matters for
        :class:`LoRAInjector` and other tools that patch
        leaves in a deep module tree.
        """
        patch = self._names.pop(name, None)
        if patch is None:
            return
        # Pop the most-recent occurrence from the stack (if any)
        # so that restore order matches apply order.  The
        # module that was patched is stored in the patch
        # metadata so we can pass it to the undo callable.
        target_module = None
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i] is patch:
                meta = self._stack[i].metadata or {}
                target_module = meta.get("module", self.model)
                del self._stack[i]
                break
        undo = getattr(patch, "_undo", None)
        if callable(undo):
            undo(target_module)
        patch._undo = None  # type: ignore[attr-defined]

    def clear(self) -> None:
        """Remove every patch (in reverse apply order)."""
        for patch in list(reversed(self._stack)):
            self.remove(patch.name)

    # ------------------------------------------------------------------
    def apply(self, full_name: str = "", module: Optional[nn.Module] = None) -> int:
        """Apply every patch whose key_filter matches ``full_name``.

        Args:
            full_name: Fully qualified module name (e.g.
                ``"blocks.0.attn.qkv"``).  May be empty to
                mean "the model itself".
            module: The module to apply patches to.  Defaults
                to :attr:`model`.

        Returns:
            The number of patches newly applied.
        """
        if module is None:
            module = self.model
        n = 0
        for name, patch in self._names.items():
            if not patch.matches(full_name):
                continue
            if any(p is patch for p in self._stack):
                # Already applied; skip.
                continue
            undo = patch.op(module, patch.strength)
            patch._undo = undo  # type: ignore[attr-defined]
            self._stack.append(patch)
            n += 1
        return n

    def __enter__(self) -> "ModelPatcher":
        self.apply("", self.model)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.clear()


# ---------------------------------------------------------------------------
# Offload helpers
# ---------------------------------------------------------------------------
_HOOK_KEY = "_torcha_verse_offload"


def _hooked(m: nn.Module) -> bool:
    return bool(getattr(m, _HOOK_KEY, False))


def _mark_hooked(m: nn.Module) -> None:
    setattr(m, _HOOK_KEY, True)


def _move(m: nn.Module, device: Union[str, torch.device]) -> nn.Module:
    """Move ``m`` to ``device`` *in place* and return it.

    This is a thin wrapper around ``m.to(device)`` that returns
    the same module for fluent chaining.  When the module is
    already on ``device`` this is a no-op.
    """
    return m.to(device)


def _make_offload_hooks(
    leaf: nn.Module,
    compute_device: Union[str, torch.device],
    offload_device: Union[str, torch.device],
    *,
    sequential: bool,
    stream: List[nn.Module],
) -> None:
    """Attach the forward pre/post hooks that implement offload.

    The leaf module is moved to the **offload_device** at
    registration time, and re-materialised to the
    **compute_device** on every forward call.
    """
    if _hooked(leaf):
        return
    _mark_hooked(leaf)

    # Move to the offload device at registration time so that the
    # base memory footprint matches the "no offload" cost when
    # compute_device == offload_device.
    if str(leaf.device) != str(offload_device):
        try:
            leaf.to(offload_device)
        except Exception:  # noqa: BLE001
            # Some modules (e.g. plain Python objects) may not
            # support ``.to``; fall back to no-op.
            pass

    pre_hook_handles: List[torch.utils.hooks.RemovableHook] = []
    post_hook_handles: List[torch.utils.hooks.RemovableHook] = []

    def pre_hook(_mod: nn.Module, _inputs: Any) -> None:
        # Move the leaf to the compute device.
        try:
            leaf.to(compute_device)
        except Exception:  # noqa: BLE001
            return
        if sequential and stream:
            # Move the *previous* leaf in the stream back to
            # the offload device (so the offload stays tight).
            for prev in reversed(stream[:-1]):
                if prev is leaf:
                    continue
                try:
                    prev.to(offload_device)
                except Exception:  # noqa: BLE001
                    pass
                break

    pre_hook_handles.append(leaf.register_forward_pre_hook(pre_hook))

    def post_hook(_mod: nn.Module, _inputs: Any, _outputs: Any) -> None:
        if not sequential:
            # Per-submodule offload: move the leaf back to
            # the offload device as soon as its forward
            # pass is done.
            try:
                leaf.to(offload_device)
            except Exception:  # noqa: BLE001
                pass

    post_hook_handles.append(leaf.register_forward_hook(post_hook))

    # Stash the handles so ``disable_offload`` can find them.
    setattr(leaf, _HOOK_KEY + "_handles", pre_hook_handles + post_hook_handles)


def _iter_leaves(model: nn.Module) -> Iterable[nn.Module]:
    """Yield every *leaf* :class:`nn.Module` (no children)."""
    for m in model.modules():
        # ``m.modules()`` yields the root first; skip it when
        # it has children.
        if len(list(m.children())) == 0:
            yield m


def enable_model_cpu_offload(
    model: nn.Module,
    *,
    compute_device: Union[str, torch.device] = "cpu",
    offload_device: Union[str, torch.device] = "cpu",
) -> int:
    """Attach per-submodule CPU offload hooks to ``model``.

    When ``compute_device == offload_device`` (the common
    CPU-only path) this is a no-op that returns ``0`` -- the
    leaf modules are not moved and no hooks are attached.

    Args:
        model: The :class:`nn.Module` to offload.
        compute_device: Device the leaf module is moved to
            immediately before its forward pass.  Defaults
            to ``"cpu"``.
        offload_device: Device the leaf module lives on at
            rest.  Defaults to ``"cpu"``.

    Returns:
        The number of leaf modules that were hooked.
    """
    if str(compute_device) == str(offload_device):
        return 0
    n = 0
    for leaf in _iter_leaves(model):
        _make_offload_hooks(
            leaf, compute_device, offload_device,
            sequential=False, stream=[],
        )
        n += 1
    return n


def enable_sequential_cpu_offload(
    model: nn.Module,
    *,
    compute_device: Union[str, torch.device] = "cpu",
    offload_device: Union[str, torch.device] = "cpu",
) -> int:
    """Attach the strict stream-mode sequential offload hooks.

    The "stream" is the iteration order of ``model.modules()``;
    each leaf is moved to the compute device as it is entered
    and the previous leaf is moved back to the offload device
    before the new one is materialised.  This produces the
    smallest possible peak memory (one leaf at a time) at the
    cost of CPU<->GPU traffic.

    When ``compute_device == offload_device`` this is a no-op
    that returns ``0``.
    """
    if str(compute_device) == str(offload_device):
        return 0
    stream: List[nn.Module] = []
    n = 0
    for leaf in _iter_leaves(model):
        stream.append(leaf)
        _make_offload_hooks(
            leaf, compute_device, offload_device,
            sequential=True, stream=stream,
        )
        n += 1
    return n


def disable_offload(model: nn.Module) -> int:
    """Remove offload hooks attached by the two helpers above.

    Returns the number of modules whose hooks were removed.
    """
    n = 0
    for leaf in _iter_leaves(model):
        handles = getattr(leaf, _HOOK_KEY + "_handles", None)
        if not handles:
            continue
        for h in handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        try:
            delattr(leaf, _HOOK_KEY + "_handles")
        except AttributeError:
            pass
        try:
            delattr(leaf, _HOOK_KEY)
        except AttributeError:
            pass
        n += 1
    return n
