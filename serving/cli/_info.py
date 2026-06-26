"""Root-level informational commands: ``torcha info`` and ``torcha models``.

These are attached directly to the root :func:`cli` group (not to
a sub-group) so the public invocation is just ``torcha info`` and
``torcha models`` -- matching the v0.4.x / v0.5.x contract.
"""

from __future__ import annotations

import platform
from typing import Any, Dict

import click
from rich.table import Table

from infrastructure.device_manager import DeviceManager

from ._runtime import console


@click.command()
def info() -> None:
    """Show runtime / device / version information."""
    device = DeviceManager().get_device()
    info_dict: Dict[str, Any] = {
        "Python": platform.python_version(),
        "Platform": platform.platform(),
        "Device": device,
    }
    table = Table(title="TorchaVerse environment", show_header=True, header_style="bold")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for key, value in info_dict.items():
        table.add_row(key, str(value))
    console.print(table)


@click.command()
def models() -> None:
    """List all registered node types (model identifiers)."""
    from ._runtime import _get_service
    rows = _get_service().list_models()
    # ``rows`` is a list of dicts (one per registered node
    # specification): ``{"id", "object", "name", "description",
    # "tags"}``.  We render three columns -- #, the model id, and
    # a derived domain tag ("text" / "image" / "audio" / "video"
    # / "rag" / "agent" / "other") taken from the first matching
    # tag so the user can scan the list at a glance.
    def _domain_for(row: Dict[str, Any]) -> str:
        for tag in row.get("tags", []) or []:
            t = str(tag).lower()
            if t in {"text", "image", "audio", "video", "rag", "agent"}:
                return t
        return "other"

    table = Table(
        title=f"Registered models ({len(rows)})",
        show_header=True,
        header_style="bold",
    )
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Model ID", style="magenta")
    table.add_column("Name", style="white")
    table.add_column("Domain", style="green")
    for i, row in enumerate(rows, start=1):
        model_id = str(row.get("id", "?"))
        name = str(row.get("name", ""))
        table.add_row(str(i), model_id, name, _domain_for(row))
    console.print(table)


__all__ = ["info", "models"]
