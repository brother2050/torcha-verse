"""One-line model loader for the v0.4.0 P0 milestone (pure-torch).

This module ties :mod:`models.source` (the cache + license policy)
together with :mod:`models.providers.tiny_transformer` and
:mod:`models.providers.local_text` so the v0.4.0 P0 milestone
ships a single ergonomic entry point::

    from models.providers import fetch_and_load_text
    provider = fetch_and_load_text(
        "torcha-verse/tiny-transformer-small",
        config_name="small",
    )
    reply = provider.chat(
        [{"role": "user", "content": "Hello, who are you?"}],
    )

The loader uses a *virtual* source called ``"local"`` for the
project-owned tiny Transformer.  The :class:`ModelFetcher` does not
know how to reach the network for ``"local"``; instead the
:meth:`fetch_and_load_text` function takes a local ``.pt`` path
directly, or -- if no path is supplied -- constructs a fresh
random-initialised model via :class:`LocalTorchTextProvider` (the
P0 "no checkpoint available" fallback).  Both paths are
dependency-free.

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``models.providers`` (this module) -- one-line loader.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from infrastructure.logger import get_logger

from .local_text import LocalTorchTextProvider
from .tiny_transformer import (
    SMALL_CONFIG,
    TINY_CONFIG,
    TinyTransformerConfig,
    build_tiny_transformer,
    load_tiny_transformer,
    save_tiny_transformer,
)

__all__ = [
    "fetch_and_load_text",
    "publish_tiny_transformer",
    "resolve_config_by_name",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Mapping of public preset names to :class:`TinyTransformerConfig`.
_PRESETS: Dict[str, TinyTransformerConfig] = {
    "tiny": TINY_CONFIG,
    "small": SMALL_CONFIG,
}

#: Module-level logger.
_logger = get_logger("models.providers.factory")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_config_by_name(name: str) -> TinyTransformerConfig:
    """Return the :class:`TinyTransformerConfig` for a preset name.

    Supported presets: ``"tiny"`` (~0.3M params, for CI) and
    ``"small"`` (~10M params, for the P0 demo).  An unknown name
    raises :class:`ValueError`.
    """
    if not name or not name.strip():
        raise ValueError("preset name must be non-empty")
    key = name.strip().lower()
    if key not in _PRESETS:
        raise ValueError(
            "unknown preset {!r}; expected one of {}".format(
                name, sorted(_PRESETS.keys()),
            )
        )
    return _PRESETS[key]


# ---------------------------------------------------------------------------
# Publish (local-only "fake source") and fetch
# ---------------------------------------------------------------------------
def publish_tiny_transformer(
    out_path: Union[str, Path],
    *,
    config_name: str = "small",
) -> Path:
    """Initialise a fresh tiny Transformer and save it to ``out_path``.

    This is the *publish* side of the v0.4.0 P0 self-hosted model
    distribution: a maintainer runs this once to produce a
    ``.pt`` file, which is then committed to the repository
    under ``assets/checkpoints/`` (or downloaded via the local
    source).  End-users call :func:`fetch_and_load_text` to load
    the resulting file.
    """
    cfg = resolve_config_by_name(config_name)
    model, tok = build_tiny_transformer(cfg)
    return save_tiny_transformer(model, tok, out_path, config=cfg)


def fetch_and_load_text(
    repo_id: str = "torcha-verse/tiny-transformer-small",
    *,
    revision: str = "main",
    config_name: Optional[str] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cpu",
) -> LocalTorchTextProvider:
    """Load a project-owned tiny Transformer provider.

    The loader supports two modes:

    1. **Random-init fallback** -- when ``checkpoint_path`` is
       ``None`` and the cache does not contain a published
       ``.pt`` file, a fresh :class:`LocalTorchTextProvider` is
       constructed with a randomly-initialised model.  This is
       the v0.4.0 P0 default and is what the demo / tests rely
       on to keep the milestone dependency-free.
    2. **Local checkpoint** -- when ``checkpoint_path`` is given
       (or found in :mod:`models.source` cache), the
       ``.pt`` is loaded and the model is reconstructed from
       its state-dict.

    Args:
        repo_id: A logical id used as the cache key when
            ``checkpoint_path`` is ``None`` and the file is found
            in :mod:`models.source` cache.  The default is
            ``"torcha-verse/tiny-transformer-small"``.
        revision: Cache revision (default ``"main"``).
        config_name: Optional preset name (``"tiny"`` / ``"small"``)
            used when a random-init fallback model is built.  When
            ``None`` the function tries to read the config from
            the loaded ``.pt`` payload.
        checkpoint_path: Optional explicit path to a ``.pt`` file.
        device: Device to map the loaded tensors onto.

    Returns:
        A fully constructed :class:`LocalTorchTextProvider`.
    """
    # 1. Explicit path wins.
    if checkpoint_path is not None:
        p = Path(checkpoint_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(
                "checkpoint file not found: {}".format(p)
            )
        return LocalTorchTextProvider.from_file(p, device=device)

    # 2. Try the local source cache via models.source.fetch.
    cached: Optional[Path] = None
    try:
        from models.source import (
            fetch as _fetch,
            FetchResult as _FetchResult,
            DEFAULT_ALLOW_LICENSE,
        )
        result = _fetch(
            repo_id=repo_id,
            source="local",
            revision=revision,
            allow_license=list(DEFAULT_ALLOW_LICENSE),
            verify_cache=True,
        )
        if isinstance(result, _FetchResult) and result.accepted:
            candidate = result.location.path() / "model.pt"
            if candidate.is_file():
                cached = candidate
    except Exception as exc:  # noqa: BLE001
        # The "local" source may not be registered -- that is
        # fine, we just fall through to the random-init path.
        _logger.debug(
            "models.source.fetch did not resolve a checkpoint "
            "for %s: %s",
            repo_id, exc,
        )

    if cached is not None and cached.is_file():
        return LocalTorchTextProvider.from_file(cached, device=device)

    # 3. Fallback: build a random-init provider.
    cfg_name = config_name or "tiny"
    cfg = resolve_config_by_name(cfg_name)
    _logger.info(
        "No checkpoint available for %s@%s; building a random-init "
        "%s provider (params~%.1fM).",
        repo_id, revision, cfg.name, cfg.approx_params_m(),
    )
    return LocalTorchTextProvider.from_random(cfg, device=device)


# ---------------------------------------------------------------------------
# Public singleton (for ``register_default_text_backend`` use)
# ---------------------------------------------------------------------------
_default_provider_lock: threading.Lock = threading.Lock()
_default_provider: Optional[LocalTorchTextProvider] = None


def get_default_provider() -> LocalTorchTextProvider:
    """Return the process-level singleton :class:`LocalTorchTextProvider`.

    The singleton uses the **TINY preset** (small enough to be
    free in CI) and is rebuilt only on the first call.  Callers
    that need a different config / device should construct
    their own :class:`LocalTorchTextProvider` via
    :func:`fetch_and_load_text`.
    """
    global _default_provider
    if _default_provider is None:
        with _default_provider_lock:
            if _default_provider is None:
                _default_provider = (
                    LocalTorchTextProvider.from_random(TINY_CONFIG)
                )
    return _default_provider
