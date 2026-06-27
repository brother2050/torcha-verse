"""Local pipeline: 自研的 "transformers.pipeline" 风格推理管道 (v0.10.0)。

为 torcha-verse 提供一个**类似 ``transformers.pipeline``** 的多模态
推理管道工厂,但**零外部依赖**。

设计动机
--------

V0.8.0 的实现让"真模型真生成"成为可能,但调用方依然要写:

```python
model = load_hunyuan_dit(path, ...)
text = tokenizer.encode(prompt)
text_embeds = model.encode_text(text)
latents = torch.randn(...)
out = call_diffusion_loop_backend(...)
```

本模块提供:

* :class:`TextGenerationPipeline` -- 类似
  ``transformers.pipeline("text-generation")``
* :class:`ImageGenerationPipeline` -- 类似
  ``diffusers.StableDiffusionPipeline.__call__``
* :class:`AudioPipeline` -- 类似
  ``transformers.pipeline("text-to-audio"|"text-to-speech")``
* :func:`pipeline` -- 一行构造管道,自动选 family

每条管道都接收一个 **TaskHead** (从
:mod:`models.runtime.loader` 来),以及一个可选的 ``backend``
工厂 (用于把节点注册到 :class:`core.module_bus.ModuleBus`)。

零外部依赖
----------

不依赖 ``transformers`` / ``diffusers`` / ``accelerate``。所有依赖
都是项目自有的 L1-L6 模块。

测试 0 回归
-----------

* 全部占位 / 失败路径在 ``docs/placeholder_registry.md`` 登记
* 不破坏 1182+ 现有测试
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch

from infrastructure.logger import get_logger

from .transformers_style_loader import (
    ModelForCausalLM,
    ModelForMusic,
    ModelForTextToImage,
    ModelForTextToSpeech,
    ModelFamily,
    TokenizerBundle,
)

__all__ = [
    "TextGenerationPipeline",
    "ImageGenerationPipeline",
    "AudioPipeline",
    "PipelineOutput",
    "pipeline",
    "list_supported_tasks",
]


_logger = get_logger("models.runtime.transformers_style_pipeline")


# ---------------------------------------------------------------------------
# PipelineOutput
# ---------------------------------------------------------------------------
@dataclass
class PipelineOutput:
    """A minimal ``transformers.PipelineOutput`` analogue.

    Holds a ``list`` of records (one per input) so callers can use
    both the dict-style and list-style accessors common in
    ``transformers``.  Each record is a plain ``dict`` so callers
    can introspect / pretty-print / serialise without depending on
    a third-party class hierarchy.
    """

    records: List[Dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]

    def to_dict(self) -> List[Dict[str, Any]]:
        """Return a deep-copyable list of records."""
        return [dict(r) for r in self.records]

    def __repr__(self) -> str:
        return f"PipelineOutput(n={len(self.records)})"


# ---------------------------------------------------------------------------
# TextGenerationPipeline
# ---------------------------------------------------------------------------
class TextGenerationPipeline:
    """A ``transformers.pipeline("text-generation")`` analogue.

    Args:
        task_head: A :class:`ModelForCausalLM` (or any object
            exposing ``.generate(prompt, **kwargs)``).
        device: Optional device override.  ``None`` keeps the model's
            current device.
        dtype: Optional dtype override.
    """

    def __init__(
        self,
        task_head: ModelForCausalLM,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.task_head: ModelForCausalLM = task_head
        self._device = device
        self._dtype = dtype

    def __call__(
        self,
        prompts: Union[str, Sequence[str]],
        *,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> PipelineOutput:
        """Run text generation.

        Args:
            prompts: A single string or a list of strings.
            max_new_tokens: Number of new tokens to generate.
            do_sample: ``False`` → greedy decoding.
            temperature: Sampling temperature (ignored when
                ``do_sample=False``).
            top_k: Top-k truncation (0 disables).
            top_p: Top-p / nucleus truncation (1.0 disables).
            **kwargs: Forwarded to :meth:`ModelForCausalLM.generate`.

        Returns:
            A :class:`PipelineOutput` with one record per input.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        records: List[Dict[str, Any]] = []
        for prompt in prompts:
            text = self.task_head.generate(
                prompt,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                **kwargs,
            )
            records.append(
                {
                    "prompt": prompt,
                    "generated_text": text,
                    "family": self.task_head.family.value,
                }
            )
        return PipelineOutput(records=records)

    def __repr__(self) -> str:
        return (
            f"TextGenerationPipeline(family={self.task_head.family.value!r}, "
            f"model={type(self.task_head.model).__name__})"
        )


# ---------------------------------------------------------------------------
# ImageGenerationPipeline
# ---------------------------------------------------------------------------
class ImageGenerationPipeline:
    """A ``diffusers.StableDiffusionPipeline.__call__`` analogue.

    Args:
        task_head: A :class:`ModelForTextToImage`.
        device: Optional device override.
        dtype: Optional dtype override.
    """

    def __init__(
        self,
        task_head: ModelForTextToImage,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.task_head: ModelForTextToImage = task_head
        self._device = device
        self._dtype = dtype

    def __call__(
        self,
        prompts: Union[str, Sequence[str]],
        *,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        sampler: str = "flow_match_euler",
        **kwargs: Any,
    ) -> PipelineOutput:
        """Generate images from ``prompts``.

        Each record in the returned :class:`PipelineOutput` carries:

        * ``prompt`` -- the input string
        * ``latents`` -- the denoised latent tensor (``torch.Tensor``)
        * ``text_embeds`` -- the encoded prompt (``torch.Tensor``)
        * ``sampler`` -- the sampler name used
        * any extra keys returned by the underlying backend
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        records: List[Dict[str, Any]] = []
        for prompt in prompts:
            out = self.task_head(
                prompt,
                height=int(height),
                width=int(width),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                sampler=str(sampler),
                **kwargs,
            )
            rec: Dict[str, Any] = {
                "prompt": prompt,
                "family": self.task_head.family.value,
            }
            if isinstance(out, dict):
                for k, v in out.items():
                    rec[k] = v
            else:
                rec["output"] = out
            records.append(rec)
        return PipelineOutput(records=records)

    def __repr__(self) -> str:
        return (
            f"ImageGenerationPipeline(family={self.task_head.family.value!r}, "
            f"model={type(self.task_head.model).__name__})"
        )


# ---------------------------------------------------------------------------
# AudioPipeline
# ---------------------------------------------------------------------------
class AudioPipeline:
    """A ``transformers.pipeline("text-to-audio")`` analogue.

    The pipeline accepts a TTS / music task head and dispatches
    based on the head's family.  The two flavours share the same
    call signature so the user can swap them transparently.
    """

    def __init__(
        self,
        task_head: Union[ModelForTextToSpeech, ModelForMusic],
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.task_head = task_head
        self._device = device
        self._dtype = dtype

    def __call__(
        self,
        prompts: Union[str, Sequence[str]],
        *,
        sample_rate: int = 22050,
        duration_s: float = 8.0,
        **kwargs: Any,
    ) -> PipelineOutput:
        """Generate audio from ``prompts``."""
        if isinstance(prompts, str):
            prompts = [prompts]
        records: List[Dict[str, Any]] = []
        for prompt in prompts:
            if isinstance(self.task_head, ModelForMusic):
                out = self.task_head(
                    prompt,
                    duration_s=float(duration_s),
                    sample_rate=int(sample_rate),
                    **kwargs,
                )
            else:
                out = self.task_head(
                    prompt,
                    sample_rate=int(sample_rate),
                    **kwargs,
                )
            rec: Dict[str, Any] = {
                "prompt": prompt,
                "family": self.task_head.family.value,
            }
            if isinstance(out, dict):
                for k, v in out.items():
                    rec[k] = v
            else:
                rec["output"] = out
            records.append(rec)
        return PipelineOutput(records=records)

    def __repr__(self) -> str:
        return (
            f"AudioPipeline(family={self.task_head.family.value!r}, "
            f"model={type(self.task_head.model).__name__})"
        )


# ---------------------------------------------------------------------------
# Top-level pipeline() factory
# ---------------------------------------------------------------------------
_TASK_REGISTRY: Dict[str, str] = {
    "text-generation": "text",
    "text-to-image": "image",
    "text2image": "image",
    "image-generation": "image",
    "text-to-speech": "audio",
    "tts": "audio",
    "text-to-audio": "audio",
    "music-generation": "audio",
    "audio-generation": "audio",
}


def list_supported_tasks() -> List[str]:
    """Return the list of ``pipeline()`` task names the runtime supports."""
    return list(_TASK_REGISTRY.keys())


def pipeline(
    task: str,
    *,
    model: Optional[Any] = None,
    tokenizer: Optional[TokenizerBundle] = None,
    family: Union[ModelFamily, str, None] = None,
    model_path: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device: Union[None, str, torch.device] = None,
    **kwargs: Any,
) -> Union[
    TextGenerationPipeline,
    ImageGenerationPipeline,
    AudioPipeline,
]:
    """Construct a local inference pipeline.

    This is the analogue of ``transformers.pipeline(...)``.  Two
    ways to use it:

    1. **Inline load** (one call): pass ``model_path`` and the
       function will call
       :func:`models.runtime.load_model_and_tokenizer` for you.
    2. **Pre-loaded** (two calls): construct a TaskHead
       (e.g. :class:`ModelForCausalLM`) and pass it via
       ``model=``.

    Args:
        task: One of :func:`list_supported_tasks`.
        model: A pre-built TaskHead (overrides ``model_path``).
        tokenizer: Optional :class:`TokenizerBundle` (used when
            ``model`` is given directly).
        family: Optional model family override.
        model_path: Local path to a checkpoint (used when
            ``model`` is ``None``).
        torch_dtype: Forwarded to
            :func:`models.runtime.load_model_and_tokenizer`.
        device: Forwarded to
            :func:`models.runtime.load_model_and_tokenizer`.
        **kwargs: Forwarded to the underlying loader / model.

    Returns:
        A pipeline instance.  Concrete type depends on ``task``:

        * text-generation → :class:`TextGenerationPipeline`
        * text-to-image  → :class:`ImageGenerationPipeline`
        * text-to-speech / music-generation / text-to-audio
          → :class:`AudioPipeline`

    Raises:
        ValueError: When ``task`` is not supported.
        RuntimeError: When neither ``model`` nor ``model_path`` is
            provided.
    """
    if task not in _TASK_REGISTRY:
        raise ValueError(
            f"pipeline(): unsupported task {task!r}. "
            f"Supported: {list_supported_tasks()}"
        )
    kind = _TASK_REGISTRY[task]
    # Lazy model load.
    if model is None:
        if model_path is None:
            raise RuntimeError(
                "pipeline(): either `model` (TaskHead) or `model_path` is required"
            )
        from .transformers_style_loader import load_model_and_tokenizer
        mdl, tok, fam = load_model_and_tokenizer(
            model_path,
            family=family,
            torch_dtype=torch_dtype,
            device=device,
            **kwargs,
        )
        if kind == "text":
            model = ModelForCausalLM(mdl, tok, fam)
        elif kind == "image":
            model = ModelForTextToImage(mdl, tok, fam)
        else:
            if fam == ModelFamily.MUSICGEN:
                model = ModelForMusic(mdl, tok, fam)
            else:
                model = ModelForTextToSpeech(mdl, tok, fam)

    if kind == "text":
        return TextGenerationPipeline(
            model, device=device, dtype=torch_dtype,
        )
    if kind == "image":
        return ImageGenerationPipeline(
            model, device=device, dtype=torch_dtype,
        )
    return AudioPipeline(
        model, device=device, dtype=torch_dtype,
    )
