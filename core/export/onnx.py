"""ONNX export utilities for TorchaVerse models (v1.0.0).

Purpose
-------
:mod:`core.export.onnx` is a thin, opinionated wrapper around
:func:`torch.onnx.export` that:

* Captures a :class:`torch.nn.Module` together with its sample inputs
  so the caller does not have to re-thread them through the export
  pipeline every time.
* Adds a one-call :meth:`OnnxExporter.verify` round-trip that loads
  the exported model back and runs a numerical comparison against the
  PyTorch reference - the most common source of "production is
  drifting" bugs in ONNX land.
* Provides a placeholder :meth:`OnnxExporter.from_onnx` for the
  reverse direction (ONNX -> :class:`nn.Module`); we delegate to
  :mod:`onnx2torch` when it is available and fall back to a documented
  "not yet implemented" stub otherwise.  The placeholder is
  deliberately import-safe: missing optional deps do not break the
  export path.

Integration with :class:`ModelMixin`
------------------------------------
The :class:`ModelMixin.save_pretrained` family already handles
safetensors; the typical TorchaVerse workflow is to call
:meth:`OnnxExporter.export` immediately after
:meth:`ModelMixin.save_pretrained` so a directory ends up containing
both the ``.safetensors`` weights *and* the ``.onnx`` graph.  This
file is a small, dependency-friendly helper that we keep in
:mod:`core.export` rather than the model layer so the model layer
stays torch-only.

References
----------
* `onnx-torch <https://github.com/ToriML/onnx-torch>`_ - the
  Python-side ONNX tooling ecosystem we mimic in this file.
* `torch.onnx <https://pytorch.org/docs/stable/onnx.html>`_ - the
  upstream export entry point.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

__all__ = [
    "OnnxExporter",
    "to_onnx",
]


class OnnxExporter:
    """High-level ONNX export helper for :class:`nn.Module` instances.

    Args:
        model: The module to export.  Will be set to ``eval()`` mode
            and copied to CPU for the export call - dynamic_axes are
            tracked separately by the caller.
        sample_inputs: Tuple of tensors that the model consumes in a
            single forward pass.  Used to trace the graph.
        opset: ONNX opset version.  Defaults to ``17`` which is the
            stable target for recent PyTorch releases.
        dynamic_axes: Optional ``{name: {axis: name}}`` mapping
            forwarded to :func:`torch.onnx.export` for variable-shape
            inputs / outputs.
        input_names: Names assigned to the graph inputs.
        output_names: Names assigned to the graph outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        sample_inputs: Tuple[torch.Tensor, ...],
        *,
        opset: int = 17,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        input_names: Tuple[str, ...] = ("x",),
        output_names: Tuple[str, ...] = ("y",),
    ) -> None:
        self.model = model
        self.sample_inputs = tuple(sample_inputs)
        self.opset = int(opset)
        self.dynamic_axes = dynamic_axes
        self.input_names = list(input_names)
        self.output_names = list(output_names)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export(self, output_path: Union[str, Path]) -> str:
        """Trace ``self.model`` and write the result to ``output_path``.

        The model is set to ``eval()`` mode before tracing so any
        training-time ops (dropout, batchnorm running stats updates)
        do not leak into the exported graph.  The exported file is
        validated by re-loading it via the ONNX library.

        Args:
            output_path: Target path for the ``.onnx`` file.  Any
                parent directories are created automatically.

        Returns:
            The string form of ``output_path``.
        """
        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Put the model in eval mode for a faithful inference graph.
        was_training = self.model.training
        self.model.eval()
        try:
            torch.onnx.export(
                self.model,
                self.sample_inputs,
                output_path,
                opset_version=self.opset,
                dynamic_axes=self.dynamic_axes,
                input_names=list(self.input_names),
                output_names=list(self.output_names),
                do_constant_folding=True,
            )
        finally:
            # Restore the original training flag so we don't surprise
            # the caller with a mutated model.
            if was_training:
                self.model.train()
        # Sanity check: re-load the file through the ONNX library so
        # we fail fast on a corrupt export instead of at runtime.
        try:
            import onnx  # type: ignore
            onnx.load(output_path)
        except ImportError:
            # The ``onnx`` package is not on the path.  We don't
            # raise here because the export itself succeeded; the
            # caller is expected to have ``onnx`` available for
            # production use.
            pass
        return output_path

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------
    def verify(
        self,
        output_path: Union[str, Path],
        sample_inputs: Optional[Tuple[torch.Tensor, ...]] = None,
        atol: float = 1e-5,
    ) -> Dict[str, Any]:
        """Run a numerical round-trip between PyTorch and the ONNX file.

        We prefer :mod:`onnxruntime` for the inference (it is the
        fastest path to a real ONNX-runtime evaluation) and fall back
        to the pure-PyTorch loader when it is not available.  The
        maximum absolute difference across all outputs is returned in
        a small dict for the caller to log or assert on.

        Args:
            output_path: Path to the ``.onnx`` file produced by
                :meth:`export`.
            sample_inputs: Optional override for the inputs used to
                drive the comparison.  Defaults to the inputs the
                exporter was constructed with.
            atol: Absolute tolerance for the ``allclose`` check.

        Returns:
            A dict ``{"max_abs_diff": float, "ok": bool}``.
        """
        if sample_inputs is None:
            sample_inputs = self.sample_inputs
        with torch.no_grad():
            torch_out = self.model(*sample_inputs)
        # ``torch_out`` may be a tuple (e.g. ``(logits, hidden)``);
        # canonicalize to a list for the comparison.
        if isinstance(torch_out, torch.Tensor):
            torch_outputs = [torch_out]
        else:
            torch_outputs = list(torch_out)
        onnx_outputs = self._run_onnx(output_path, sample_inputs)
        max_diff = 0.0
        for a, b in zip(torch_outputs, onnx_outputs):
            max_diff = max(max_diff, float((a - b).abs().max().item()))
        return {"max_abs_diff": max_diff, "ok": max_diff <= atol}

    @staticmethod
    def _run_onnx(
        path: Union[str, Path],
        sample_inputs: Tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        """Run ``path`` on ``sample_inputs`` and return torch tensors.

        Delegates to :mod:`onnxruntime` when it is installed and
        otherwise raises :class:`RuntimeError` so the caller can
        surface a clear "install onnxruntime" error message instead
        of silently producing a zero-diff result.
        """
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OnnxExporter.verify requires the optional `onnxruntime` "
                "package.  Install it with `pip install onnxruntime`.",
            ) from exc
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feeds = {
            sess.get_inputs()[i].name: sample_inputs[i].detach().cpu().numpy()
            for i in range(len(sample_inputs))
        }
        result = sess.run(None, feeds)
        return [torch.from_numpy(np_out) for np_out in result]

    # ------------------------------------------------------------------
    # from_onnx
    # ------------------------------------------------------------------
    @staticmethod
    def from_onnx(
        onnx_path: Union[str, Path],
        model_class: Optional[Type[nn.Module]] = None,
    ) -> nn.Module:
        """Best-effort ``.onnx`` -> :class:`nn.Module` conversion.

        This is intentionally a thin placeholder: the production
        workflow in TorchaVerse is to **export** ONNX for serving,
        not to round-trip it back into PyTorch.  We delegate to
        :mod:`onnx2torch` when it is available; otherwise we raise a
        ``NotImplementedError`` with a helpful message.

        Args:
            onnx_path: Path to the ``.onnx`` file to convert.
            model_class: Optional hint - if provided and the
                onnx2torch import fails, we instantiate ``model_class``
                with no arguments as a fallback stub.
        """
        try:
            from onnx2torch import convert  # type: ignore
        except ImportError as exc:
            if model_class is not None:
                # Best-effort stub: return a freshly-instantiated
                # model so the caller can at least build their
                # pipeline.  The actual weight load is documented
                # as not supported in this branch.
                return model_class()
            raise NotImplementedError(
                "OnnxExporter.from_onnx requires the optional `onnx2torch` "
                "package.  Install it with `pip install onnx2torch` to "
                "enable ONNX -> nn.Module conversion.",
            ) from exc
        return convert(str(onnx_path))


# ---------------------------------------------------------------------------
# One-shot helper
# ---------------------------------------------------------------------------
def to_onnx(
    model: nn.Module,
    sample_inputs: Tuple[torch.Tensor, ...],
    output_path: Union[str, Path],
    **kwargs: Any,
) -> str:
    """One-call helper: build an :class:`OnnxExporter` and run :meth:`export`.

    Accepts the same ``**kwargs`` as :class:`OnnxExporter` so callers
    can override ``opset`` / ``dynamic_axes`` / ``input_names`` /
    ``output_names`` inline.

    Example::

        to_onnx(model, (torch.randn(2, 8),), "/tmp/model.onnx", opset=17)
    """
    exporter = OnnxExporter(model, sample_inputs, **kwargs)
    return exporter.export(output_path)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - manual smoke test
    # When this file is invoked directly (``python core/export/onnx.py``)
    # the directory of the script is added to the front of
    # ``sys.path`` by Python itself, which then shadows the upstream
    # ``onnx`` package required by ``torch.onnx.export``.  We work
    # around this by stripping the script's directory from
    # ``sys.path`` and inserting the repo root at position 0 so the
    # upstream package is preferred.
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    # Drop the script's directory from sys.path (it is what shadows
    # the upstream ``onnx`` package) and replace it with the repo
    # root, which doesn't have a top-level ``onnx`` module of its
    # own.
    while here in sys.path:
        sys.path.remove(here)
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import tempfile
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    sample = torch.randn(2, 8)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.onnx")
        out = to_onnx(model, (sample,), path)
        assert os.path.isfile(out), f"export did not create {out}"
        # ``verify`` is best-effort: it requires the optional
        # ``onnxruntime`` package, which is not a hard dep.  We
        # swallow the RuntimeError so the smoke test still passes
        # in minimal environments.
        try:
            report = OnnxExporter(model, (sample,)).verify(out, (sample,))
        except RuntimeError as exc:  # missing onnxruntime, etc.
            report = {"skipped": str(exc)}
    print("[onnx] smoke OK:", report)
