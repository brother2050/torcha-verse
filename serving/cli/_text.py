"""Text sub-group: ``torcha text generate`` and ``torcha text chat``.

These commands are thin wrappers around
:func:`serving.service.PipelineService.text_completion` and
:func:`serving.service.PipelineService.text_chat`.  Both write
the generated text to ``--output`` (default ``-`` for stdout)
and print a small panel with engine / model / device info.
"""

from __future__ import annotations

import click
from rich.panel import Panel

from ._runtime import (
    _get_service,
    _print_engine_info,
    console,
)


@click.group()
def text() -> None:
    """Text generation utilities."""


@text.command()
@click.argument("prompt")
@click.option(
    "--model",
    default="default",
    help="Text model identifier.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=128,
    help="Maximum tokens to generate.",
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="-",
    help="Output file (``-`` for stdout).",
)
def generate(prompt: str, model: str, max_tokens: int, temperature: float, output: str) -> None:
    """Run a single text completion."""
    _print_engine_info("text_completion", model)
    # v0.10.3: forward --model to the local runtime so the
    # factory tries the user-downloaded checkpoint first.
    # ``"default"`` is the legacy alias for "use whatever the
    # runtime decides" and is mapped to ``None`` here.
    rt_model_id = None if model in ("default", "", None) else model
    result = _get_service(model_id=rt_model_id).text_completion(
        prompt=prompt, model=model, max_tokens=max_tokens, temperature=temperature,
    )
    if "error" in result:
        console.print(f"[red]Engine error:[/red] {result['error']}")
        raise SystemExit(1)
    text_out = result.get("text", "")
    if output == "-":
        console.print(f"[bold green]>>>[/bold green] {text_out}")
        # v0.10.5: surface the v0.10.4 quality diagnostics so
        # the user knows when the project micro-transformer
        # emitted mostly noise (random-weight artefact).  The
        # warning is never raised for a properly-loaded
        # user checkpoint.
        if result.get("quality_warning"):
            console.print(
                f"[yellow][garble: {result.get('garble_level', '?')}]"
                f" {result['quality_warning']}[/yellow]"
            )
    else:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(text_out)
        console.print(f"[green]Text saved to[/green] [bold]{output}[/bold]")
        if result.get("quality_warning"):
            console.print(
                f"[yellow][garble: {result.get('garble_level', '?')}]"
                f" {result['quality_warning']}[/yellow]"
            )


@text.command()
@click.argument("user_message")
@click.option(
    "--system",
    "system_prompt",
    default="You are a helpful assistant.",
    help="System message for the conversation.",
)
@click.option(
    "--model",
    default="default",
    help="Text model identifier.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=256,
    help="Maximum tokens to generate.",
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="-",
    help="Output file (``-`` for stdout).",
)
def chat(
    user_message: str,
    system_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    output: str,
) -> None:
    """Run a one-shot chat completion."""
    _print_engine_info("text_chat", model)
    from serving.models import ChatMessage
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_message),
    ]
    result = _get_service(model_id=(
        None if model in ("default", "", None) else model
    )).text_chat(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if "error" in result:
        console.print(f"[red]Engine error:[/red] {result['error']}")
        raise SystemExit(1)
    text_out = result.get("text", "")
    if output == "-":
        console.print(f"[bold green]>>>[/bold green] {text_out}")
        if result.get("quality_warning"):
            console.print(
                f"[yellow][garble: {result.get('garble_level', '?')}]"
                f" {result['quality_warning']}[/yellow]"
            )
    else:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(text_out)
        console.print(f"[green]Chat saved to[/green] [bold]{output}[/bold]")
        if result.get("quality_warning"):
            console.print(
                f"[yellow][garble: {result.get('garble_level', '?')}]"
                f" {result['quality_warning']}[/yellow]"
            )


__all__ = ["text", "generate", "chat"]
