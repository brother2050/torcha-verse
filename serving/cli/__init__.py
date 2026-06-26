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

Public surface (preserved from v0.4.x / v0.5.x):

* :func:`main` -- CLI entry point (``torcha`` from the shell).
* :data:`cli` -- the root :class:`click.Group` (re-exported for
  tests that build runner objects).
"""

from __future__ import annotations

import click

from ._agent import agent
from ._audio import audio
from ._image import image
from ._info import info, models
from ._rag import rag
from ._runtime import _get_service, console, logger
from ._text import text
from ._video import video

__all__ = ["main", "cli", "console", "logger", "_get_service"]


# ---------------------------------------------------------------------------
# Root CLI group
# ---------------------------------------------------------------------------
@click.group()
@click.version_option(package_name="torcha-verse")
def cli() -> None:
    """TorchaVerse -- unified multimodal inference toolkit."""


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
