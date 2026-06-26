"""RAG sub-group: ``torcha rag ingest`` and ``torcha rag query``.

* ``ingest`` wraps the ``rag_ingest`` L4 node -- chunk + embed
  a source file (or directory) into the named index.
* ``query`` wraps the ``rag_query`` L4 node -- retrieve top-k
  chunks (and optionally synthesise an answer via the LLM) for a
  given question.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, List

import click
from rich.panel import Panel

from ._runtime import (
    _get_service,
    _print_engine_info,
    console,
)


@click.group()
def rag() -> None:
    """Retrieval-augmented generation utilities."""


@rag.command()
@click.argument("source")
@click.option(
    "--index",
    "index_name",
    required=True,
    help="Name of the target index.",
)
@click.option(
    "--chunk-size",
    type=int,
    default=512,
    help="Chunk size in characters (or tokens, depending on the embedder).",
)
@click.option(
    "--chunk-overlap",
    type=int,
    default=64,
    help="Number of overlapping tokens / characters between consecutive chunks.",
)
@click.option(
    "--recursive/--no-recursive",
    default=False,
    help="Walk directories recursively (default: off).",
)
@click.option(
    "--file-type",
    "file_types",
    multiple=True,
    help="Filter by file extension (e.g. ``.md``).  Can be passed multiple times.",
)
@click.option(
    "--metadata",
    "metadata_json",
    default=None,
    help="Optional JSON-encoded metadata dict attached to every chunk.",
)
def ingest(
    source: str,
    index_name: str,
    chunk_size: int,
    chunk_overlap: int,
    recursive: bool,
    file_types: List[str],
    metadata_json: str,
) -> None:
    """Ingest a file / directory into a RAG index."""
    _print_engine_info("rag_ingest", index_name)
    metadata: Any = None
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            console.print(f"[red]Invalid --metadata JSON:[/red] {exc}")
            raise SystemExit(1)
    result = _get_service()._run(
        "rag_ingest",
        "rag_ingest",
        "ingest",
        {
            "source": source,
            "index_name": index_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "recursive": recursive,
            "file_types": list(file_types),
            "metadata": metadata,
        },
    )
    if "error" in result:
        console.print(f"[red]Ingest error:[/red] {result['error']}")
        raise SystemExit(1)
    chunk_count = result.get("chunk_count", result.get("chunks", 0))
    console.print(
        Panel(
            f"[bold cyan]Index:[/bold cyan]    {index_name}\n"
            f"[bold cyan]Source:[/bold cyan]   {source}\n"
            f"[bold cyan]Chunks:[/bold cyan]   {chunk_count}",
            title="Ingest complete",
            border_style="green",
            expand=False,
        )
    )


@rag.command()
@click.argument("question")
@click.option(
    "--index",
    "index_name",
    required=True,
    help="Name of the index to query.",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Number of chunks to retrieve.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=256,
    help="Maximum tokens to generate (synthesis only).",
)
@click.option(
    "--raw/--synthesise",
    "raw",
    default=False,
    help="When ``--raw`` is set, return the retrieved context without LLM synthesis.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="-",
    help="Output file (``-`` for stdout).",
)
def query(
    question: str,
    index_name: str,
    top_k: int,
    max_tokens: int,
    raw: bool,
    output: str,
) -> None:
    """Ask a question over a RAG index."""
    _print_engine_info("rag_query", index_name)
    result = _get_service()._run(
        "rag_query",
        "rag_query",
        "retrieval",
        {
            "index_name": index_name,
            "query": question,
            "top_k": top_k,
        },
    )
    if "error" in result:
        console.print(f"[red]RAG error:[/red] {result['error']}")
        raise SystemExit(1)
    hits = result.get("hits", [])
    context = result.get("context", "")
    if raw:
        # Print the retrieved hits (id, score, snippet) in a panel.
        from rich.table import Table
        table = Table(title=f"Top {len(hits)} hits", show_header=True, header_style="bold")
        table.add_column("score", style="cyan", no_wrap=True)
        table.add_column("id", style="magenta")
        table.add_column("snippet", style="white")
        for h in hits:
            table.add_row(
                f"{h.get('score', 0):.3f}",
                str(h.get("id", "")),
                str(h.get("snippet", ""))[:200],
            )
        console.print(table)
        return
    if not context:
        console.print("[yellow]No context retrieved; nothing to synthesise.[/yellow]")
        return
    # Synthesise the final answer via the LLM node.
    user_prompt = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    answer_result = _get_service()._run(
        "rag_query_synthesise",
        "text_chat",
        "answer",
        {
            "prompt": user_prompt,
            "model": "default",
            "max_tokens": max_tokens,
        },
    )
    if "error" in answer_result:
        console.print(f"[red]LLM error:[/red] {answer_result['error']}")
        raise SystemExit(1)
    text_out = str(answer_result.get("text", ""))
    if output == "-":
        console.print(f"[bold green]>>>[/bold green] {text_out}")
    else:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(text_out)
        console.print(f"[green]Answer saved to[/green] [bold]{output}[/bold]")


__all__ = ["rag", "ingest", "query"]
