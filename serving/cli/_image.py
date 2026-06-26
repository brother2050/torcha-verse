"""Image sub-group: ``torcha image txt2img`` and ``torcha image img2img``.

Both commands wrap the
:func:`serving.service.PipelineService.image_txt2img` /
:func:`serving.service.PipelineService.image_img2img`
operations and write the result to ``--output``.  The output
format is inferred from the file extension when possible.

The backend may return the image as a ``PIL.Image.Image`` (a
real diffusion model), a ``torch.Tensor`` of shape ``(3, H, W)``
or ``(1, 3, H, W)`` in ``[0, 1]`` (the local Torch image
provider), or a ``placeholder_image`` dict (the no-model echo
factory).  All three shapes are normalised to a ``PIL.Image``
before the saver is invoked.
"""

from __future__ import annotations

from typing import Any

import click
import numpy as np
from PIL import Image as PILImage

from ._runtime import (
    _get_service,
    _print_engine_info,
    _save_image,
    console,
)


def _to_pil(image_obj: Any) -> PILImage.Image:
    """Normalise an image backend response to a :class:`PIL.Image.Image`.

    Accepted shapes:

    * ``PIL.Image.Image`` -- returned as-is.
    * ``torch.Tensor`` ``(3, H, W)`` or ``(1, 3, H, W)`` of floats
      in ``[0, 1]`` (the local Torch diffusion provider) -- converted
      via ``tensor.detach().cpu().clamp(0, 1) * 255``.
    * ``numpy.ndarray`` of ``uint8`` shape ``(H, W, 3)`` -- used
      directly.
    * A placeholder dict (the echo backend) -- rendered as a small
      annotated PNG so the caller still gets a file on disk.
    """
    if isinstance(image_obj, PILImage.Image):
        return image_obj
    # torch.Tensor path: local Torch image provider.
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None  # type: ignore
    if torch is not None and isinstance(image_obj, torch.Tensor):
        tensor = image_obj.detach().cpu().float()
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
            # (C, H, W) -> (H, W, C)
            tensor = tensor.permute(1, 2, 0)
        array = (tensor.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
        return PILImage.fromarray(array)
    # numpy.ndarray path.
    if isinstance(image_obj, np.ndarray):
        array = image_obj
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype("uint8")
        return PILImage.fromarray(array)
    # Placeholder dict (echo backend): paint a small labelled PNG so
    # callers still receive an artefact on disk.
    if isinstance(image_obj, dict):
        width = int(image_obj.get("width", 256))
        height = int(image_obj.get("height", 256))
        prompt = str(image_obj.get("prompt", "placeholder"))[: 24]
        canvas = PILImage.new("RGB", (width, height), (24, 24, 32))
        # Draw the prompt with PIL.ImageDraw for a useful artefact
        # (this path runs only when no real model is registered).
        try:
            from PIL import ImageDraw
            drawer = ImageDraw.Draw(canvas)
            drawer.text((8, 8), f"[placeholder]\n{prompt}", fill=(220, 220, 220))
        except Exception:  # noqa: BLE001
            pass  # placeholder #95 (serving/cli/_image.py:87) -- PIL ImageDraw 不可用时静默回退到纯色背景,不影响产文件
        return canvas
    raise TypeError(
        f"Backend returned an unsupported image type: {type(image_obj).__name__}"
    )


@click.group()
def image() -> None:
    """Image generation utilities."""


@image.command(name="txt2img")
@click.argument("prompt")
@click.option(
    "--negative-prompt",
    default="",
    help="Negative prompt (things to avoid).",
)
@click.option("--width", type=int, default=512)
@click.option("--height", type=int, default=512)
@click.option("--steps", type=int, default=30)
@click.option("--guidance-scale", type=float, default=7.5)
@click.option("--seed", type=int, default=None)
@click.option(
    "--model",
    default="default",
    help="Image model identifier.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="output.png",
    help="Output file path.",
)
def txt2img(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    model: str,
    output: str,
) -> None:
    """Generate an image from a text prompt."""
    _print_engine_info("image_txt2img", model)
    result = _get_service().image_txt2img(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        model=model,
    )
    if "error" in result:
        console.print(f"[red]Engine error:[/red] {result['error']}")
        raise SystemExit(1)
    image_obj = result.get("image")
    if image_obj is None:
        console.print("[red]Engine returned no image.[/red]")
        raise SystemExit(1)
    # The image may be a torch.Tensor / numpy.ndarray / placeholder
    # dict; normalise to a PIL.Image before saving.
    try:
        image_obj = _to_pil(image_obj)
    except Exception as exc:
        console.print(f"[red]Could not convert image:[/red] {exc}")
        raise SystemExit(1)
    _save_image(image_obj, output)


@image.command(name="img2img")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("prompt")
@click.option(
    "--negative-prompt",
    default="",
    help="Negative prompt (things to avoid).",
)
@click.option("--strength", type=float, default=0.75)
@click.option("--steps", type=int, default=30)
@click.option("--guidance-scale", type=float, default=7.5)
@click.option("--seed", type=int, default=None)
@click.option(
    "--model",
    default="default",
    help="Image model identifier.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="output.png",
    help="Output file path.",
)
def img2img(
    input_path: str,
    prompt: str,
    negative_prompt: str,
    strength: float,
    steps: int,
    guidance_scale: float,
    seed: int,
    model: str,
    output: str,
) -> None:
    """Transform an image guided by a text prompt."""
    _print_engine_info("image_img2img", model)
    input_image = PILImage.open(input_path).convert("RGB")
    result = _get_service().image_img2img(
        image=input_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        model=model,
    )
    if "error" in result:
        console.print(f"[red]Engine error:[/red] {result['error']}")
        raise SystemExit(1)
    image_obj = result.get("image")
    if image_obj is None:
        console.print("[red]Engine returned no image.[/red]")
        raise SystemExit(1)
    try:
        image_obj = _to_pil(image_obj)
    except Exception as exc:
        console.print(f"[red]Could not convert image:[/red] {exc}")
        raise SystemExit(1)
    _save_image(image_obj, output)


__all__ = ["image", "txt2img", "img2img", "_to_pil"]
