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
from typing import Any, Dict, Iterator, List, Optional

import click

from engines.agent_engine import AgentEngine
from engines.audio_engine import AudioEngine
from engines.image_engine import ImageEngine
from engines.rag_engine import RAGEngine
from engines.text_engine import Message, TextEngine
from engines.video_engine import VideoEngine
from infrastructure.config_manager import ConfigManager
from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

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
# Shared helpers
# ===========================================================================
def _get_text_engine(model: str) -> TextEngine:
    """Create or retrieve a :class:`TextEngine` for ``model``."""
    return TextEngine(model)


def _get_image_engine(model: str) -> ImageEngine:
    """Create or retrieve an :class:`ImageEngine` for ``model``."""
    return ImageEngine(model)


def _get_audio_engine() -> AudioEngine:
    """Create or retrieve an :class:`AudioEngine`."""
    return AudioEngine()


def _get_video_engine(model: str) -> VideoEngine:
    """Create or retrieve a :class:`VideoEngine` for ``model``."""
    return VideoEngine(model)


def _get_rag_engine() -> RAGEngine:
    """Create or retrieve a :class:`RAGEngine`."""
    return RAGEngine()


def _get_agent_engine() -> AgentEngine:
    """Create or retrieve an :class:`AgentEngine`."""
    return AgentEngine()


def _print_engine_info(engine_name: str, model: str) -> None:
    """Print a small panel with engine/model info."""
    device = DeviceManager().get_device()
    console.print(
        Panel(
            f"[bold cyan]Engine:[/bold cyan] {engine_name}\n"
            f"[bold cyan]Model:[/bold cyan]  {model}\n"
            f"[bold cyan]Device:[/bold cyan] {device}",
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
    """Save an :class:`AudioTensor` to ``output`` as a WAV file."""
    import numpy as np
    import wave

    waveform = audio.numpy()
    if waveform.ndim == 2:
        waveform = waveform[0]
    waveform = np.clip(waveform, -1.0, 1.0)
    pcm = (waveform * 32767).astype(np.int16)

    with wave.open(output, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(audio.sample_rate)
        wf.writeframes(pcm.tobytes())
    console.print(f"[green]Audio saved to[/green] [bold]{output}[/bold]")


def _save_video(video: Any, output: str) -> None:
    """Save a :class:`VideoTensor` to ``output`` as a GIF."""
    from PIL import Image as PILImage
    import numpy as np

    frames = video.frames
    if frames.dim() == 5:
        frames = frames[0]
    frames_np = (frames.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(
        "uint8"
    )
    pil_frames = [PILImage.fromarray(f) for f in frames_np]
    pil_frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / video.fps),
        loop=0,
    )
    console.print(f"[green]Video saved to[/green] [bold]{output}[/bold]")


# ===========================================================================
# CLI group: torcha
# ===========================================================================
@click.group()
@click.version_option(version="0.1.0", prog_name="torcha")
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
    _print_engine_info("TextEngine", model)
    engine = _get_text_engine(model)

    start = time.time()

    if stream:
        console.print("[dim]Streaming output...[/dim]\n")
        result = engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stream=True,
        )
        assert isinstance(result, Iterator)
        full_text = ""
        for chunk in result:
            console.print(chunk, end="", style="white")
            full_text += chunk
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
            result = engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            progress.update(task, completed=1)

        console.print()
        console.print(Panel(result, title="Generated Text", border_style="green"))
        full_text = result

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
    _print_engine_info("TextEngine", model)
    engine = _get_text_engine(model)

    messages: List[Message] = []
    if system:
        messages.append(Message(role="system", content=system))

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
            engine.reset_history()
            if system:
                messages.append(Message(role="system", content=system))
            console.print("[dim]History cleared.[/dim]")
            continue

        messages.append(Message(role="user", content=user_input))

        with console.status("[bold cyan]Assistant is thinking...[/bold cyan]"):
            reply = engine.chat(messages, max_tokens=max_tokens)

        reply_text = reply.content if isinstance(reply, Message) else str(reply)
        messages.append(Message(role="assistant", content=reply_text))

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
    _print_engine_info("ImageEngine", model)
    engine = _get_image_engine(model)

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
        image = engine.txt2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        progress.update(task, completed=1)

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

    _print_engine_info("ImageEngine", model)
    engine = _get_image_engine(model)

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
        result = engine.img2img(
            image=input_image,
            prompt=prompt,
            strength=strength,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        progress.update(task, completed=1)

    _save_image(result, output)


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
    _print_engine_info("AudioEngine", model)
    engine = _get_audio_engine()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Synthesising speech...", total=1)
        audio = engine.synthesize(
            text=text_input,
            speaker_id=speaker_id,
            emotion=emotion,
            speed=speed,
        )
        progress.update(task, completed=1)

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
    _print_engine_info("VideoEngine", model)
    engine = _get_video_engine(model)

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
        video = engine.txt2video(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        progress.update(task, completed=1)

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
    """Ingest documents into the RAG vector store."""
    engine = RAGEngine(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    console.print(f"[cyan]Ingesting documents from[/cyan] [bold]{docs_path}[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading, chunking, and embedding...", total=1)
        engine.ingest(docs_path)
        progress.update(task, completed=1)

    table = Table(title="Ingestion Summary", border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Source", docs_path)
    table.add_row("Chunk size", str(chunk_size))
    table.add_row("Chunk overlap", str(chunk_overlap))
    table.add_row("Total chunks indexed", str(engine.index_size))
    console.print(table)


@rag.command()
@click.option("--question", required=True, help="Question to ask.")
@click.option("--top-k", default=5, type=int, help="Number of chunks to retrieve.")
@click.option("--rerank", is_flag=True, help="Apply keyword reranking.")
def query(question: str, top_k: int, rerank: bool) -> None:
    """Query the RAG engine with a question."""
    engine = _get_rag_engine()

    if engine.index_size == 0:
        console.print(
            "[yellow]Warning: The index is empty. "
            "Run 'torcha rag ingest' first.[/yellow]"
        )

    with console.status("[bold cyan]Retrieving and generating answer...[/bold cyan]"):
        answer, sources = engine.query(question, top_k=top_k, rerank=rerank)

    console.print()
    console.print(
        Panel(
            Markdown(answer.text),
            title="[bold green]Answer[/bold green]",
            border_style="green",
        )
    )

    if sources.chunks:
        table = Table(title="Retrieved Sources", border_style="blue")
        table.add_column("#", style="dim", width=4)
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Source", style="white")
        table.add_column("Excerpt", style="dim")

        for i, chunk in enumerate(sources.chunks, 1):
            excerpt = chunk.text[:80].replace("\n", " ") + "..."
            table.add_row(
                str(i),
                f"{chunk.score:.3f}",
                chunk.metadata.get("source", "unknown"),
                excerpt,
            )
        console.print(table)

    console.print(f"\n[dim]Confidence: {answer.confidence:.3f}[/dim]")


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
    """Run an agent on a task."""
    engine = _get_agent_engine()

    if flow:
        console.print(
            f"[cyan]Running multi-agent flow:[/cyan] [bold]{flow}[/bold]"
        )
        engine.create_agent(role="manager", max_steps=max_steps)
        engine.create_agent(role="worker", max_steps=max_steps)
        orchestrator = engine.create_flow(
            agents=["manager", "worker"],
            topology=flow,
        )

        with console.status("[bold cyan]Executing flow...[/bold cyan]"):
            result = engine.execute(orchestrator, task)
    else:
        console.print("[cyan]Running single agent (ReAct)...[/cyan]")
        if stream:
            console.print("\n[bold]Execution Trace:[/bold]\n")
            for step in engine.stream(task, max_steps=max_steps):
                _print_step(step)
            # Get the final result for the summary.
            result = engine.run(task, max_steps=max_steps)
        else:
            with console.status("[bold cyan]Agent is reasoning...[/bold cyan]"):
                result = engine.run(task, max_steps=max_steps)

    console.print()
    console.print(
        Panel(
            Markdown(result.output),
            title="[bold green]Final Output[/bold green]",
            border_style="green",
        )
    )

    # Print execution trace table.
    if result.steps:
        table = Table(title="Execution Trace", border_style="blue")
        table.add_column("Step", style="dim", width=5)
        table.add_column("Thought", style="white")
        table.add_column("Action", style="cyan")
        table.add_column("Observation", style="dim")

        for step in result.steps:
            table.add_row(
                str(step.step_number),
                step.thought[:60] + ("..." if len(step.thought) > 60 else ""),
                step.action or "-",
                step.observation[:60] + ("..." if len(step.observation) > 60 else ""),
            )
        console.print(table)

    meta = result.metadata
    console.print(
        f"\n[dim]Steps: {meta.get('steps_taken', len(result.steps))} | "
        f"Truncated: {meta.get('truncated', False)}[/dim]"
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

    table.add_row("Version", "0.1.0")
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
    """List all registered models."""
    from core.model_registry import ModelRegistry

    registry = ModelRegistry()
    available = registry.list_available()

    if not available:
        console.print("[yellow]No models registered.[/yellow]")
        return

    table = Table(title="Registered Models", border_style="cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Model Name", style="white")
    for i, name in enumerate(available, 1):
        table.add_row(str(i), name)
    console.print(table)


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    """Entry point for the TorchaVerse CLI."""
    cli()


if __name__ == "__main__":
    main()
