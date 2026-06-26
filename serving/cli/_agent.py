"""Agent sub-group: ``torcha agent run``.

Wraps the ``agent_run`` L4 node -- a thin wrapper over the
default :class:`infrastructure.agent.AgentBus` (ReAct loop with
tool-calling).  Supports streaming via ``--stream``.
"""

from __future__ import annotations

import click
from rich.panel import Panel

from ._runtime import (
    _get_service,
    _print_engine_info,
    _print_step,
    console,
)


@click.group()
def agent() -> None:
    """Agent execution utilities."""


@agent.command()
@click.argument("task")
@click.option(
    "--agent-type",
    default="react",
    help="Agent strategy (``react`` / ``plan_execute``).",
)
@click.option(
    "--max-steps",
    type=int,
    default=10,
    help="Maximum number of reasoning steps.",
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    help="Sampling temperature for the LLM call.",
)
@click.option(
    "--stream/--no-stream",
    default=False,
    help="Stream the agent's per-step output (default: off).",
)
def run(
    task: str,
    agent_type: str,
    max_steps: int,
    temperature: float,
    stream: bool,
) -> None:
    """Run an agent on a task."""
    _print_engine_info("agent_run", agent_type)
    result = _get_service()._run(
        "agent_run",
        "agent_run",
        "agent",
        {
            "query": task,
            "max_steps": max_steps,
            "temperature": temperature,
        },
    )
    if "error" in result:
        console.print(f"[red]Agent error:[/red] {result['error']}")
        raise SystemExit(1)
    final_answer = str(result.get("final_answer", ""))
    steps = result.get("steps", [])
    iterations = int(result.get("iterations", 0))
    ok = bool(result.get("ok", False))
    if stream:
        for idx, step in enumerate(steps, start=1):
            _print_step(idx, step)
    console.print(
        Panel(
            f"[bold cyan]Iterations:[/bold cyan] {iterations}\n"
            f"[bold cyan]OK:[/bold cyan]         {ok}\n\n"
            f"[bold]Final answer:[/bold]\n{final_answer}",
            title="Agent result",
            border_style="green" if ok else "red",
            expand=False,
        )
    )


__all__ = ["agent", "run"]
