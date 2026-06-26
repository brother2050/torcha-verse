"""Click-based command-line interface for TorchaVerse (v0.6.x).

The CLI was extracted from a single ``cli.py`` (962 lines) into a
sub-package.  Each sub-group now lives in its own sub-module:

* :mod:`._text` -- ``torcha text generate`` / ``torcha text chat``
* :mod:`._image` -- ``torcha image txt2img`` / ``torcha image img2img``
* :mod:`._audio` -- ``torcha audio tts``
* :mod:`._video` -- ``torcha video txt2vid``
* :mod:`._rag` -- ``torcha rag ingest`` / ``torcha rag query``
* :mod:`._agent` -- ``torcha agent run`` (with ``--stream``)
* :mod:`._info` -- ``torcha info`` / ``torcha models``

Shared state (the rich console, the lazy ``PipelineService``
singleton, the artefact-savers) lives in
:mod:`._runtime`.

R-17 — global flags on the root group:

* ``--config PATH``  — project-layer config dir override, applied
  to :class:`ConfigCenter` before any sub-command runs.
* ``--log-format {text,json}``  — flip the console log format
  (rich / plain) to JSON.  Affects all loggers created *after*
  the flag is processed.
* ``--log-level LEVEL``  — shortcut for ``configure(console_level=...)``.

Public surface (preserved from v0.4.x / v0.5.x):

* :func:`main` -- CLI entry point (``torcha`` from the shell).
* :data:`cli` -- the root :class:`click.Group` (re-exported for
  tests that build runner objects).
"""

from __future__ import annotations

import click

from infrastructure.logger import configure as _configure_logging

from ._agent import agent
from ._audio import audio
from ._image import image
from ._info import info, models
from ._rag import rag
from ._runtime import _cli_overrides, _get_service, console, logger
from ._text import text
from ._video import video

__all__ = ["main", "cli", "console", "logger", "_get_service", "_cli_overrides"]


# ---------------------------------------------------------------------------
# Root CLI group
# ---------------------------------------------------------------------------
@click.group()
@click.version_option(package_name="torcha-verse")
@click.option(
    "--config",
    "config_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    envvar="TORCHA_CONFIG_DIR",
    help=(
        "Override the project-layer config directory.  Equivalent to "
        "setting the TORCHA_CONFIG_DIR environment variable.  The path "
        "is validated by ConfigCenter and falls back to the bundled "
        "defaults when the directory does not exist."
    ),
)
@click.option(
    "--log-format",
    "log_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    envvar="TORCHA_LOG_FORMAT",
    show_default=True,
    help=(
        "Console log format.  'json' emits one JSON object per line "
        "for ELK / Loki / CloudWatch ingestion.  'text' uses the "
        "default rich / plain format."
    ),
)
@click.option(
    "--log-level",
    "log_level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"],
        case_sensitive=False,
    ),
    default=None,
    envvar="TORCHA_LOG_LEVEL",
    help=(
        "Console log level.  Defaults to INFO.  Equivalent to "
        "configure(console_level=...) on the logger module."
    ),
)
def cli(
    config_dir: str | None,
    log_format: str,
    log_level: str | None,
) -> None:
    """TorchaVerse -- unified multimodal inference toolkit.

    R-17: process the global flags here so every sub-command
    inherits the same config dir and log format.  Sub-commands
    that need the config dir can call
    :func:`infrastructure.config_center.ConfigCenter(config_dir=...)`
    directly, but the central place to read it is this group.
    """
    # 1. Logger setup.  Done before any other handler can grab a
    #    logger so the JSON / text choice sticks.
    _configure_logging(
        console_level=log_level or "INFO",
        json=(log_format.lower() == "json"),
    )

    # 2. ConfigCenter override.  We do not eagerly instantiate the
    #    singleton here because some sub-commands (e.g. ``info``)
    #    never touch configuration; the override is recorded in
    #    the :data:`_cli_overrides` module dict so ConfigCenter
    #    picks it up on first access.
    if config_dir is not None:
        from pathlib import Path
        from ._runtime import _cli_overrides
        _cli_overrides["config_dir"] = Path(config_dir).expanduser().resolve()


# Register the sub-groups.
cli.add_command(text)
cli.add_command(image)
cli.add_command(audio)
cli.add_command(video)
cli.add_command(rag)
cli.add_command(agent)
# Root-level informational commands.
cli.add_command(info)
cli.add_command(models)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the TorchaVerse CLI.

    Used by the ``torcha`` console script declared in
    :file:`setup.py` (``"torcha=serving.cli:main"``).  Catches
    :class:`click.ClickException` / :class:`click.exceptions.Abort`
    so the traceback is hidden on user errors; any other
    exception is logged at ``ERROR`` level and re-raised as a
    :class:`click.ClickException` so the exit code is non-zero.
    """
    import click as _click
    try:
        cli()
    except _click.exceptions.Exit:
        # ``--help`` / explicit ``--version`` cause a clean exit
        # -- let it propagate.
        raise
    except _click.ClickException:
        # User-facing Click errors (missing arg, bad choice, ...)
        # -- already formatted by Click, just re-raise.
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Uncaught exception in CLI: %s", exc, exc_info=True)
        raise _click.ClickException("internal error: {}".format(exc))


if __name__ == "__main__":
    main()
