"""Shared runtime helpers for the Click CLI (v0.6.x).

These helpers are used by every sub-command in
:mod:`serving.cli._text`, :mod:`serving.cli._image`,
:mod:`serving.cli._audio`, :mod:`serving.cli._video`,
:mod:`serving.cli._rag`, :mod:`serving.cli._agent`,
:mod:`serving.cli._info`.

* :data:`console` -- shared :class:`rich.console.Console` for
  pretty-printed output.
* :data:`logger` -- shared :class:`infrastructure.logger.Logger`.
* :func:`_get_service` -- lazily-instantiated
  :class:`serving.service.PipelineService` singleton.
* :func:`_print_engine_info` -- print a small panel with the
  engine / model / device info at the start of a run.
* :func:`_save_image` / :func:`_save_audio` / :func:`_save_video` --
  save the generated artefact to disk, gracefully handling the
  "placeholder" dicts returned by unimplemented nodes.
* :func:`_print_step` -- pretty-print one step of an agent's
  ReAct loop (used by ``agent run --stream``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger
from rich.console import Console
from rich.panel import Panel

from serving.service import PipelineService

__all__ = [
    "console",
    "logger",
    "_get_service",
    "_print_engine_info",
    "_save_image",
    "_save_audio",
    "_save_video",
    "_print_step",
    "_cli_overrides",
]


# ---------------------------------------------------------------------------
# R-17 — global flag overrides set by the root ``cli`` group.
# Populated when ``torcha --config <DIR>`` runs, consumed by
# anything that needs the user-selected config dir (e.g.
# ``ConfigCenter``).  Kept at module level so any sub-command
# can read it without touching private state on the group.
# ---------------------------------------------------------------------------
_cli_overrides: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Shared singletons
# ---------------------------------------------------------------------------
console: Console = Console()
logger = get_logger("cli")

# ---------------------------------------------------------------------------
# PipelineService (lazy)
# ---------------------------------------------------------------------------
_service: Optional[PipelineService] = None


def _get_service() -> PipelineService:
    """Return a lazily-created :class:`PipelineService` singleton.

    R-17: when the user passed ``--config DIR`` on the root
    group, the override is stashed in :data:`_cli_overrides` and
    forwarded to the :class:`ConfigCenter` on first construction.
    The service is cached for the lifetime of the CLI process so
    the override applies consistently across sub-commands.
    """
    global _service
    if _service is None:
        # Resolve the config-dir override, if any.  We do not
        # re-read it on every call because the user can only
        # pass it once on the command line.
        config_dir = _cli_overrides.get("config_dir")
        # ``PipelineService`` does not expose a ``config_dir``
        # constructor argument directly; it constructs
        # ``ConfigCenter`` itself.  To honour the override we
        # *pre-create* a ConfigCenter here with the override
        # applied -- the singleton short-circuits any later
        # ``ConfigCenter()`` calls.  This is the cleanest way to
        # keep the existing ``PipelineService`` constructor
        # signature stable.
        if config_dir is not None:
            try:
                from infrastructure.config_center import ConfigCenter
                # Touch the singleton with the override.  Subsequent
                # accesses (including the one inside
                # ``PipelineService.__init__``) get the same instance.
                ConfigCenter(config_dir=config_dir)
            except Exception:  # pragma: no cover - defensive
                logger.warning(
                    "Could not honour --config %s; falling back to defaults.",
                    config_dir,
                )
        _service = PipelineService()
    return _service


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def _print_engine_info(engine_name: str, model: str) -> None:
    """Print a small panel with engine/model/device info."""
    device = DeviceManager().get_device()
    console.print(
        Panel(
            f"[bold cyan]Pipeline:[/bold cyan] {engine_name}\n"
            f"[bold cyan]Model:[/bold cyan]    {model}\n"
            f"[bold cyan]Device:[/bold cyan]   {device}",
            title="TorchaVerse",
            border_style="cyan",
            expand=False,
        )
    )


# ---------------------------------------------------------------------------
# Artefact savers
# ---------------------------------------------------------------------------
def _save_image(image: Any, output: str) -> None:
    """Save a PIL image to ``output``."""
    image.save(output)
    console.print(f"[green]Image saved to[/green] [bold]{output}[/bold]")


def _save_audio(audio: Any, output: str) -> None:
    """Save an audio object to ``output`` as a WAV file.

    Accepts either a real audio object exposing ``numpy`` / ``waveform``
    / ``sample_rate`` attributes or a placeholder dict returned by the
    node system (in which case nothing is written).
    """
    import numpy as np
    import wave

    waveform = getattr(audio, "numpy", None)
    if waveform is None:
        waveform = getattr(audio, "waveform", None)
    if waveform is None:
        console.print(
            "[yellow]Audio node returned placeholder data; "
            "no file written.[/yellow]"
        )
        return
    waveform = np.asarray(waveform)
    if waveform.ndim == 2:
        waveform = waveform[0]
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm = (waveform * 32767).astype(np.int16)

    sample_rate = getattr(audio, "sample_rate", 22050)
    with wave.open(output, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    console.print(f"[green]Audio saved to[/green] [bold]{output}[/bold]")


def _save_video(video: Any, output: str) -> None:
    """Save a video object to ``output`` as a GIF.

    Accepts either a real video object exposing ``frames`` / ``fps``
    attributes or a placeholder dict returned by the node system (in
    which case nothing is written).
    """
    from PIL import Image as PILImage
    import numpy as np

    frames = getattr(video, "frames", None)
    if frames is None:
        console.print(
            "[yellow]Video node returned placeholder data; "
            "no file written.[/yellow]"
        )
        return
    frames_np = np.asarray(frames)
    if frames_np.ndim == 5:
        frames_np = frames_np[0]
    if frames_np.ndim == 4 and frames_np.shape[-1] not in (1, 3, 4):
        frames_np = np.transpose(frames_np, (0, 2, 3, 1))
    frames_np = (np.clip(frames_np, 0, 1) * 255).astype("uint8") \
        if frames_np.dtype.kind == "f" else frames_np.astype("uint8")
    pil_frames = [PILImage.fromarray(f) for f in frames_np]
    if not pil_frames:
        console.print(
            "[yellow]Video node returned no frames; no file written.[/yellow]"
        )
        return
    fps = getattr(video, "fps", 8)
    pil_frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / max(1, fps)),
        loop=0,
    )
    console.print(f"[green]Video saved to[/green] [bold]{output}[/bold]")


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------
def _print_step(idx: int, step: dict) -> None:
    """Pretty-print one step of an agent's ReAct loop.

    The ``step`` dict has the keys produced by the ``agent_run``
    L4 node: ``thought`` (string, the LLM's reasoning), ``action``
    (string, the action it picked), ``observation`` (string, the
    tool result).  Missing keys are rendered as ``-``.
    """
    thought = step.get("thought", "-")
    action = step.get("action", "-")
    observation = step.get("observation", "-")
    console.print(
        Panel(
            f"[bold]Thought:[/bold]\n{thought}\n\n"
            f"[bold]Action:[/bold]\n{action}\n\n"
            f"[bold]Observation:[/bold]\n{observation}",
            title=f"Step {idx}",
            border_style="magenta",
            expand=False,
        )
    )
