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
    table = Table(title=f"Registered models ({len(rows)})", show_header=True, header_style="bold")
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Name", style="magenta")
    table.add_column("Type", style="green")
    for i, name in enumerate(rows, start=1):
        kind = "text" if "text" in name else "image" if "image" in name else "?"
        table.add_row(str(i), name, kind)
    console.print(table)


__all__ = ["info", "models"]
