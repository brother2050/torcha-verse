"""Audio sub-group: ``torcha audio tts``.

Wraps :func:`serving.service.PipelineService.audio_tts` and
writes the WAV file to ``--output``.
"""

from __future__ import annotations

import click

from ._runtime import (
    _get_service,
    _print_engine_info,
    _save_audio,
    console,
)


@click.group()
def audio() -> None:
    """Audio synthesis utilities."""


@audio.command(name="tts")
@click.argument("text")
@click.option(
    "--voice",
    default="default",
    help="Voice / speaker identifier.",
)
@click.option("--speed", type=float, default=1.0)
@click.option("--emotion", default="neutral")
@click.option(
    "--model",
    default="default",
    help="Audio model identifier.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="output.wav",
    help="Output WAV file path.",
)
def tts(text: str, voice: str, speed: float, emotion: str, model: str, output: str) -> None:
    """Synthesise speech from text."""
    _print_engine_info("audio_tts", model)
    result = _get_service().audio_tts(
        text=text,
        voice=voice,
        speed=speed,
        emotion=emotion,
        model=model,
    )
    if "error" in result:
        console.print(f"[red]Engine error:[/red] {result['error']}")
        raise SystemExit(1)
    audio_obj = result.get("audio")
    if audio_obj is None:
        console.print("[red]Engine returned no audio.[/red]")
        raise SystemExit(1)
    _save_audio(audio_obj, output)


__all__ = ["audio", "tts"]
