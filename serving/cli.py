"""Command-line interface for TorchaVerse.

This module provides a ``click``-based CLI with ``rich``-powered output
for all framework capabilities: text generation, chat, image synthesis,
audio TTS, video generation, RAG ingestion/query, and agent execution.

Usage examples::

    torcha text generate --model llama-8b --prompt "Hello" --stream
    torcha text chat --model llama-8b
    torcha image txt2img --model sd15 --prompt "a cat" --output out.png
    torcha audio tts --model cosyvoice --text "Hello world" --output out.wav
    torcha video txt2vid --model wan2.2 --prompt "sunset" --output out.mp4
    torcha rag ingest --docs ./data/
    torcha rag query --question "What is RAG?"
    torcha agent run --task "Summarise the news" --flow hierarchical
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional

import click

from infrastructure.config_manager import ConfigManager
from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

# Reuse the PipelineService from the API server so the CLI shares the
# same Pipeline/Node back-end as the REST API and Web UI.
from serving.api_server import PipelineService

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.markdown import Markdown
except ImportError:  # pragma: no cover - rich is in requirements.txt
    raise ImportError(
        "rich is required for the CLI. Install it with: pip install rich"
    )

__all__ = ["main", "cli"]

console: Console = Console()
logger = get_logger("cli")


# ===========================================================================
# Pipeline service holder (lazy singleton)
# ===========================================================================
_service: Optional[PipelineService] = None


def _get_service() -> PipelineService:
    """Return a lazily-created :class:`PipelineService` singleton."""
    global _service
    if _service is None:
        _service = PipelineService()
    return _service


def _print_engine_info(engine_name: str, model: str) -> None:
    """Print a small panel with engine/model info."""
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


# ===========================================================================
# CLI group: torcha
# ===========================================================================
@click.group()
@click.version_option(version="0.3.1", prog_name="torcha")
def cli() -> None:
    """TorchaVerse -- a pure PyTorch all-modal generative AI framework.

    Use --help on any subcommand for detailed options.
    """


# ===========================================================================
# Text commands
# ===========================================================================
@cli.group()
def text() -> None:
    """Text generation and chat commands."""


@text.command()
@click.option("--model", default="default", help="Model name to use.")
@click.option("--prompt", required=True, help="Input prompt text.")
@click.option(
    "--max-tokens", default=256, type=int, help="Maximum tokens to generate."
)
@click.option("--temperature", default=0.7, type=float, help="Sampling temperature.")
@click.option("--top-k", default=50, type=int, help="Top-k filtering.")
@click.option("--top-p", default=0.9, type=float, help="Nucleus sampling threshold.")
@click.option("--stream", is_flag=True, help="Stream output token by token.")
def generate(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    stream: bool,
) -> None:
    """Generate text from a prompt."""
    _print_engine_info("text_completion", model)
    service = _get_service()

    start = time.time()

    if stream:
        console.print("[dim]Streaming output...[/dim]\n")
        result = service.text_completion(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            return
        full_text = result.get("text", "")
        console.print(full_text, style="white")
        console.print()
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating text...", total=1)
            result = service.text_completion(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            progress.update(task, completed=1)

        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            return
        full_text = result.get("text", "")
        console.print()
        console.print(Panel(full_text, title="Generated Text", border_style="green"))

    elapsed = time.time() - start
    console.print(
        f"\n[dim]Completed in {elapsed:.2f}s "
        f"({len(full_text)} chars)[/dim]"
    )


@text.command()
@click.option("--model", default="default", help="Model name to use.")
@click.option("--system", default="", help="Optional system prompt.")
@click.option("--max-tokens", default=512, type=int, help="Max tokens per reply.")
def chat(model: str, system: str, max_tokens: int) -> None:
    """Interactive multi-turn chat."""
    _print_engine_info("text_chat", model)
    service = _get_service()

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})

    console.print(
        Panel(
            "[bold]Interactive Chat[/bold]\n"
            "Type 'exit' or 'quit' to end the conversation.\n"
            "Type 'clear' to reset history.",
            border_style="cyan",
            expand=False,
        )
    )

    while True:
        try:
            user_input = console.input("[bold green]You>[/bold green] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Chat ended.[/dim]")
            break

        if user_input.strip().lower() in ("exit", "quit"):
            console.print("[dim]Chat ended.[/dim]")
            break
        if user_input.strip().lower() == "clear":
            messages.clear()
            if system:
                messages.append({"role": "system", "content": system})
            console.print("[dim]History cleared.[/dim]")
            continue

        messages.append({"role": "user", "content": user_input})

        # Flatten the conversation into a single prompt for the node.
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        with console.status("[bold cyan]Assistant is thinking...[/bold cyan]"):
            result = service.text_chat(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
            )

        if "error" in result:
            reply_text = f"[Error] {result['error']}"
        else:
            reply_text = result.get("text", "")
        messages.append({"role": "assistant", "content": reply_text})

        console.print()
        console.print(
            Panel(
                Markdown(reply_text),
                title="[bold blue]Assistant[/bold blue]",
                border_style="blue",
            )
        )
        console.print()


# ===========================================================================
# Image commands
# ===========================================================================
@cli.group()
def image() -> None:
    """Image generation commands."""


@image.command()
@click.option("--model", default="default", help="Image model name.")
@click.option("--prompt", required=True, help="Text prompt for image generation.")
@click.option("--negative-prompt", default="", help="Negative prompt.")
@click.option("--output", default="output.png", help="Output file path.")
@click.option("--width", default=512, type=int, help="Image width.")
@click.option("--height", default=512, type=int, help="Image height.")
@click.option("--steps", default=30, type=int, help="Denoising steps.")
@click.option("--guidance-scale", default=7.5, type=float, help="CFG guidance scale.")
@click.option("--seed", default=None, type=int, help="Random seed.")
def txt2img(
    model: str,
    prompt: str,
    negative_prompt: str,
    output: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
) -> None:
    """Generate an image from a text prompt."""
    _print_engine_info("image_txt2img", model)
    service = _get_service()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Generating {width}x{height} image ({steps} steps)...", total=1
        )
        result = service.image_txt2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            model=model,
        )
        progress.update(task, completed=1)

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return

    image = result.get("image", result)
    from PIL import Image as PILImage

    if not isinstance(image, PILImage.Image):
        console.print(
            "[yellow]Image node returned placeholder data; "
            "no file written.[/yellow]"
        )
        return
    _save_image(image, output)


@image.command()
@click.option("--model", default="default", help="Image model name.")
@click.option("--input", "input_path", required=True, help="Input image path.")
@click.option("--prompt", required=True, help="Transformation prompt.")
@click.option("--output", default="output.png", help="Output file path.")
@click.option("--strength", default=0.75, type=float, help="Transformation strength.")
@click.option("--steps", default=30, type=int, help="Denoising steps.")
@click.option("--guidance-scale", default=7.5, type=float, help="CFG guidance scale.")
@click.option("--seed", default=None, type=int, help="Random seed.")
def img2img(
    model: str,
    input_path: str,
    prompt: str,
    output: str,
    strength: float,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
) -> None:
    """Transform an existing image using a text prompt."""
    from PIL import Image as PILImage

    _print_engine_info("image_img2img", model)
    service = _get_service()

    input_image = PILImage.open(input_path).convert("RGB")
    console.print(f"[dim]Loaded input image: {input_path}[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Transforming image...", total=1)
        result = service.image_img2img(
            image=input_image,
            prompt=prompt,
            strength=strength,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            model=model,
        )
        progress.update(task, completed=1)

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return

    out_image = result.get("image", result)
    if not isinstance(out_image, PILImage.Image):
        console.print(
            "[yellow]Image node returned placeholder data; "
            "no file written.[/yellow]"
        )
        return
    _save_image(out_image, output)


# ===========================================================================
# Audio commands
# ===========================================================================
@cli.group()
def audio() -> None:
    """Audio generation commands."""


@audio.command()
@click.option("--model", default="default", help="TTS model name.")
@click.option("--text", "text_input", required=True, help="Text to synthesise.")
@click.option("--output", default="output.wav", help="Output WAV file path.")
@click.option("--speaker-id", default=0, type=int, help="Speaker identity.")
@click.option("--emotion", default="neutral", help="Emotion label.")
@click.option("--speed", default=1.0, type=float, help="Speech speed multiplier.")
def tts(
    model: str,
    text_input: str,
    output: str,
    speaker_id: int,
    emotion: str,
    speed: float,
) -> None:
    """Synthesize speech from text."""
    _print_engine_info("audio_tts", model)
    service = _get_service()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Synthesising speech...", total=1)
        result = service.audio_tts(
            text=text_input,
            voice=str(speaker_id),
            speed=speed,
            emotion=emotion,
            model=model,
        )
        progress.update(task, completed=1)

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return

    audio = result.get("audio", result)
    _save_audio(audio, output)


# ===========================================================================
# Video commands
# ===========================================================================
@cli.group()
def video() -> None:
    """Video generation commands."""


@video.command()
@click.option("--model", default="default", help="Video model name.")
@click.option("--prompt", required=True, help="Text prompt for video generation.")
@click.option("--output", default="output.gif", help="Output file path.")
@click.option("--width", default=512, type=int, help="Video width.")
@click.option("--height", default=512, type=int, help="Video height.")
@click.option("--num-frames", default=16, type=int, help="Number of frames.")
@click.option("--fps", default=8, type=int, help="Output frame rate.")
@click.option("--steps", default=30, type=int, help="Denoising steps.")
@click.option("--guidance-scale", default=7.5, type=float, help="CFG guidance scale.")
@click.option("--seed", default=None, type=int, help="Random seed.")
def txt2vid(
    model: str,
    prompt: str,
    output: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
) -> None:
    """Generate a video from a text prompt."""
    _print_engine_info("video_txt2vid", model)
    service = _get_service()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Generating {num_frames} frames ({steps} steps)...", total=1
        )
        result = service.video_txt2vid(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            model=model,
        )
        progress.update(task, completed=1)

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return

    video = result.get("video", result)
    _save_video(video, output)


# ===========================================================================
# RAG commands
# ===========================================================================
@cli.group()
def rag() -> None:
    """Retrieval-Augmented Generation commands."""


@rag.command()
@click.option(
    "--docs",
    "docs_path",
    required=True,
    help="Path to a document file or directory.",
)
@click.option("--chunk-size", default=512, type=int, help="Chunk size in characters.")
@click.option(
    "--chunk-overlap", default=64, type=int, help="Chunk overlap in characters."
)
def ingest(docs_path: str, chunk_size: int, chunk_overlap: int) -> None:
    """Ingest documents into the RAG vector store.

    RAG ingestion is not yet backed by a node; a descriptive message is
    printed while the command structure is preserved.
    """
    service = _get_service()
    result = service.rag_query()
    msg = result.get("error", "RAG ingestion is not yet available via the Pipeline/Node system.")
    console.print(f"[yellow]{msg}[/yellow]")
    console.print(
        f"[dim]Requested source: {docs_path} "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})[/dim]"
    )


@rag.command()
@click.option("--question", required=True, help="Question to ask.")
@click.option("--top-k", default=5, type=int, help="Number of chunks to retrieve.")
@click.option("--rerank", is_flag=True, help="Apply keyword reranking.")
def query(question: str, top_k: int, rerank: bool) -> None:
    """Query the RAG engine with a question.

    RAG query is not yet backed by a node; a ``not_implemented`` message
    is printed while the command structure is preserved.
    """
    service = _get_service()
    result = service.rag_query()
    msg = result.get("error", "RAG query is not yet available via the Pipeline/Node system.")
    console.print(f"[yellow]{msg}[/yellow]")
    console.print(
        f"[dim]Question: {question} (top_k={top_k}, rerank={rerank})[/dim]"
    )


# ===========================================================================
# Agent commands
# ===========================================================================
@cli.group()
def agent() -> None:
    """Agent execution commands."""


@agent.command()
@click.option("--task", required=True, help="Task description for the agent.")
@click.option(
    "--flow",
    type=click.Choice(["sequential", "parallel", "hierarchical", "debate"]),
    default=None,
    help="Multi-agent flow topology (single agent if omitted).",
)
@click.option("--max-steps", default=10, type=int, help="Maximum reasoning steps.")
@click.option("--stream", is_flag=True, help="Stream execution steps.")
def run(task: str, flow: Optional[str], max_steps: int, stream: bool) -> None:
    """Run an agent on a task.

    Agent execution is not yet backed by a node; a ``not_implemented``
    message is printed while the command structure is preserved.
    """
    service = _get_service()
    if flow:
        console.print(
            f"[cyan]Requested multi-agent flow:[/cyan] [bold]{flow}[/bold]"
        )
    else:
        console.print("[cyan]Requested single agent (ReAct)...[/cyan]")

    result = service.agent_run()
    msg = result.get("error", "Agent execution is not yet available via the Pipeline/Node system.")
    console.print(f"\n[yellow]{msg}[/yellow]")
    console.print(
        f"[dim]Task: {task} (max_steps={max_steps}, stream={stream})[/dim]"
    )


def _print_step(step: Any) -> None:
    """Print a single agent step with rich formatting."""
    console.print(
        Panel(
            f"[bold]Thought:[/bold] {step.thought}\n"
            f"[bold cyan]Action:[/bold cyan] {step.action or '-'}\n"
            f"[bold cyan]Input:[/bold cyan] {step.action_input or '-'}",
            title=f"Step {step.step_number}",
            border_style="yellow",
            expand=False,
        )
    )
    if step.observation:
        console.print(
            Panel(
                step.observation,
                title="Observation",
                border_style="dim",
                expand=False,
            )
        )
    console.print()


# ===========================================================================
# Utility commands
# ===========================================================================
@cli.command()
def info() -> None:
    """Display framework and device information."""
    cfg = ConfigManager()
    device_mgr = DeviceManager()
    device_info = device_mgr.get_device_info()

    table = Table(title="TorchaVerse Info", border_style="cyan")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Version", "0.3.1")
    table.add_row("Environment", cfg.environment)
    table.add_row("Config dir", str(cfg.config_dir))
    table.add_row("Device", device_info.get("device", "cpu"))
    table.add_row("Device type", device_info.get("type", "cpu"))
    if "name" in device_info:
        table.add_row("Device name", device_info["name"])
    if "total_memory_gb" in device_info:
        table.add_row("Total memory", f"{device_info['total_memory_gb']} GB")
    table.add_row("CUDA available", str(device_info.get("cuda_available", False)))
    table.add_row("Distributed", str(device_info.get("distributed", False)))

    console.print(table)

    # Loaded config files.
    if cfg.loaded_files:
        files_table = Table(title="Loaded Config Files", border_style="dim")
        files_table.add_column("File", style="white")
        for f in cfg.loaded_files:
            files_table.add_row(str(f))
        console.print(files_table)


@cli.command()
def models() -> None:
    """List all registered node types (models)."""
    service = _get_service()
    available = service.list_models()

    if not available:
        console.print("[yellow]No node types registered.[/yellow]")
        return

    table = Table(title="Registered Node Types", border_style="cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Node Type", style="white")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="dim")
    for i, m in enumerate(available, 1):
        table.add_row(
            str(i),
            str(m.get("id", "")),
            str(m.get("name", "")),
            str(m.get("description", ""))[:60],
        )
    console.print(table)


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    """Entry point for the TorchaVerse CLI."""
    cli()


if __name__ == "__main__":
    main()
