"""Video sub-group: ``torcha video txt2vid``.

Wraps :func:`serving.service.PipelineService.video_txt2vid`
and writes the GIF to ``--output``.
"""

from __future__ import annotations

import click

from ._runtime import (
    _get_service,
    _print_engine_info,
    _save_video,
    console,
)


@click.group()
def video() -> None:
    """Video generation utilities."""


@video.command(name="txt2vid")
@click.argument("prompt")
@click.option(
    "--negative-prompt",
    default="",
    help="Negative prompt (things to avoid).",
)
@click.option("--width", type=int, default=256)
@click.option("--height", type=int, default=256)
@click.option("--num-frames", type=int, default=16)
@click.option("--fps", type=int, default=8)
@click.option("--steps", type=int, default=30)
@click.option("--guidance-scale", type=float, default=7.5)
@click.option("--seed", type=int, default=None)
@click.option(
    "--model",
    default="default",
    help="Video model identifier.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="output.gif",
    help="Output GIF file path.",
)
def txt2vid(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    model: str,
    output: str,
) -> None:
    """Generate a video from a text prompt."""
    _print_engine_info("video_txt2vid", model)
    result = _get_service().video_txt2vid(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        model=model,
    )
    if "error" in result:
        console.print(f"[red]Engine error:[/red] {result['error']}")
        raise SystemExit(1)
    video_obj = result.get("video")
    if video_obj is None:
        console.print("[red]Engine returned no video.[/red]")
        raise SystemExit(1)
    _save_video(video_obj, output)


__all__ = ["video", "txt2vid"]
