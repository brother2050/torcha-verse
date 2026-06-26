"""Image sub-group: ``torcha image txt2img`` and ``torcha image img2img``.

Both commands wrap the
:func:`serving.service.PipelineService.image_txt2img` /
:func:`serving.service.PipelineService.image_img2img`
operations and write the result to ``--output``.  The output
format is inferred from the file extension when possible.
"""

from __future__ import annotations

import click
from PIL import Image as PILImage

from ._runtime import (
    _get_service,
    _print_engine_info,
    _save_image,
    console,
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
    # The image may be returned as a numpy array -- convert to PIL.
    if not isinstance(image_obj, PILImage.Image):
        try:
            import numpy as np
            image_obj = PILImage.fromarray(np.asarray(image_obj).astype("uint8"))
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
    if not isinstance(image_obj, PILImage.Image):
        try:
            import numpy as np
            image_obj = PILImage.fromarray(np.asarray(image_obj).astype("uint8"))
        except Exception as exc:
            console.print(f"[red]Could not convert image:[/red] {exc}")
            raise SystemExit(1)
    _save_image(image_obj, output)


__all__ = ["image", "txt2img", "img2img"]
