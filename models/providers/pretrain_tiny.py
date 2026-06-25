"""Self-contained tiny Transformer trainer for the v0.4.0 P0 milestone
(pure-torch, no external dependencies).

The trainer is intentionally minimal:

* it builds a :class:`TransformerDecoder` from a
  :class:`TinyTransformerConfig`;
* it trains on a **byte-level language-modelling objective** over
  a tiny in-memory corpus (a few paragraphs of English text that
  ship with the project) using AdamW + a simple cosine schedule;
* it saves the trained model to a single ``.pt`` file via
  :func:`models.providers.tiny_transformer.save_tiny_transformer`.

The training script is **not** meant to be a production trainer --
it exists so the v0.4.0 P0 milestone can demonstrate an end-to-end
real-model coverage of the 30-node L4 capability layer without
introducing a single external dependency (no ``transformers``, no
``datasets``, no ``tokenizers``, no ``safetensors``, no
``accelerate``, no ``numpy``-only ops -- everything is built on
top of the project's existing
:mod:`models.text.transformer.TransformerDecoder`).

Layering (L1 -> L6):

* L1 ``infrastructure`` -- logging.
* L6 ``models.providers`` (this module) -- training script.

CLI usage
---------

::

    python -m models.providers.pretrain_tiny \
        --out assets/checkpoints/tiny-transformer-small.pt \
        --preset small --steps 600 --batch-size 8 --block-size 128
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from infrastructure.logger import get_logger

from .tiny_transformer import (
    ByteTokenizer,
    SMALL_CONFIG,
    TINY_CONFIG,
    TinyTransformerConfig,
    build_tiny_transformer,
    save_tiny_transformer,
)

__all__ = [
    "TinyCorpus",
    "DEFAULT_CORPUS",
    "TrainConfig",
    "train_tiny_transformer",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: A small, self-contained English corpus.  Roughly 1.2kB of
#: training text, repeated until the requested step count is
#: reached.  The text is deliberately *plain* -- no code, no
#: markup, no tabs -- so the byte-level tokenizer can be
#: exercised without surprises.
DEFAULT_CORPUS: str = (
    "Once upon a time, in a small village by the river, there lived a "
    "curious child who loved to ask questions. The child asked the "
    "baker about the smell of fresh bread. The child asked the "
    "fisherman about the shape of the clouds. Every morning the "
    "child greeted the sun with a smile and a question. The villagers "
    "called the child the Little Inquirer, and they were always happy "
    "to share what they knew.\n\n"
    "The baker said: bread is made of flour, water, salt and time. "
    "The fisherman said: clouds are made of water, wind and patience. "
    "The teacher said: questions are made of curiosity, courage and "
    "kindness. The child listened carefully, and every evening the "
    "child wrote a small note in a notebook by candlelight.\n\n"
    "Years went by, and the Little Inquirer grew up. The notebooks "
    "filled an entire shelf, and the village children came from far "
    "away to read them. The grown Inquirer opened a small school by "
    "the river, where every lesson began with a question and ended "
    "with a smile.\n"
) * 4

#: Module-level logger.
_logger = get_logger("models.providers.pretrain_tiny")


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
class TinyCorpus:
    """A tiny in-memory byte-level corpus.

    Wraps a string in a :class:`ByteTokenizer` and exposes a
    ``get_batch(batch_size, block_size)`` method that yields
    random fixed-length windows for next-token prediction.
    """

    def __init__(
        self,
        text: str = DEFAULT_CORPUS,
        tokenizer: Optional[ByteTokenizer] = None,
        seed: int = 20260625,
    ) -> None:
        self._text: str = text
        self._tokenizer: ByteTokenizer = tokenizer or ByteTokenizer()
        self._ids: torch.Tensor = torch.tensor(
            self._tokenizer.encode(text, add_bos=False, add_eos=False),
            dtype=torch.long,
        )
        if self._ids.numel() == 0:
            raise ValueError("corpus produced zero tokens")
        self._generator: torch.Generator = torch.Generator()
        self._generator.manual_seed(int(seed))

    @property
    def size(self) -> int:
        """Number of tokens in the corpus."""
        return int(self._ids.numel())

    def get_batch(
        self,
        batch_size: int,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a random (input, target) pair.

        The target is the input shifted one position to the right
        -- the standard next-token prediction objective.  When the
        corpus is shorter than ``block_size + 1`` the entire
        corpus is returned and the target is the same length.
        """
        if self.size < 2:
            raise ValueError("corpus is too small for LM training")
        max_start = max(1, self.size - block_size - 1)
        starts = torch.randint(
            low=0, high=max_start, size=(batch_size,),
            generator=self._generator,
        )
        rows: List[torch.Tensor] = []
        for s in starts.tolist():
            end = min(s + block_size + 1, self.size)
            rows.append(self._ids[s:end])
        # Pad to the same length with PAD id (0).
        max_len = max(r.numel() for r in rows)
        padded = torch.zeros(len(rows), max_len, dtype=torch.long)
        for i, r in enumerate(rows):
            padded[i, : r.numel()] = r
        x = padded[:, :-1]
        y = padded[:, 1:].clone()
        # Mask padded positions so the loss ignores them.
        if x.shape[1] < block_size:
            # All rows had the same short length; nothing to mask.
            return x, y
        # Rows that ended at the corpus boundary carry a zero pad
        # in the last column -- replace the corresponding ``y``
        # with -100 so ``F.cross_entropy`` ignores it.
        valid_lengths = torch.tensor(
            [min(block_size, r.numel() - 1) for r in rows],
            dtype=torch.long,
        )
        for i, n in enumerate(valid_lengths.tolist()):
            if n < y.shape[1]:
                y[i, n:] = -100
        return x, y


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Hyper-parameters for :func:`train_tiny_transformer`.

    Defaults are tuned for the ``"small"`` preset (~10M params) on
    CPU: 600 AdamW steps, batch size 8, block size 128, peak
    learning rate 3e-4, weight decay 0.1, cosine decay to 10% of
    peak.  On a single CPU thread this completes in about 2-3
    minutes; on a single GPU it is well under 30 s.
    """

    preset: str = "small"
    steps: int = 600
    batch_size: int = 8
    block_size: int = 128
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    warmup_steps: int = 30
    grad_clip: float = 1.0
    log_every: int = 50
    seed: int = 20260625
    out_path: Optional[Union[str, Path]] = None


# ---------------------------------------------------------------------------
# LR schedule + training loop
# ---------------------------------------------------------------------------
def _cosine_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup + cosine decay to ``cfg.min_lr_ratio * cfg.lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(
        1, cfg.steps - cfg.warmup_steps,
    )
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cos)


def train_tiny_transformer(
    config: Optional[TinyTransformerConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    corpus: Optional[TinyCorpus] = None,
    *,
    device: Union[str, torch.device] = "cpu",
    save: bool = True,
) -> Tuple[nn.Module, ByteTokenizer, TinyTransformerConfig]:
    """Train a tiny Transformer LM on the in-memory corpus.

    Args:
        config: Optional :class:`TinyTransformerConfig`.  Defaults
            to :data:`SMALL_CONFIG`.
        train_cfg: Optional :class:`TrainConfig`.  Defaults to
            :data:`TrainConfig()`.
        corpus: Optional pre-built :class:`TinyCorpus`.  When
            ``None`` a default :class:`TinyCorpus` over
            :data:`DEFAULT_CORPUS` is used.
        device: Device to train on.  CPU is the default so the
            v0.4.0 P0 demo works in any environment.
        save: When ``True`` (default) the trained model is
            written to ``train_cfg.out_path`` (which defaults to
            ``assets/checkpoints/tiny-transformer-<preset>.pt``).

    Returns:
        ``(model, tokenizer, config)``.
    """
    cfg = config or SMALL_CONFIG
    tcfg = train_cfg or TrainConfig(preset=cfg.name)
    tcfg.preset = cfg.name  # sync

    torch.manual_seed(int(tcfg.seed))
    device_t = torch.device(device) if not isinstance(device, torch.device) else device

    model, tokenizer = build_tiny_transformer(cfg)
    model = model.to(device_t)
    if corpus is None:
        corpus = TinyCorpus(tokenizer=tokenizer, seed=tcfg.seed)

    # AdamW: do not apply weight decay to bias / norm parameters.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    optimiser = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": tcfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=tcfg.lr,
        betas=(0.9, 0.95),
    )
    model.train()

    _logger.info(
        "Starting training: preset=%s, params=%d, steps=%d, "
        "batch=%d, block=%d, lr=%.2e, device=%s",
        cfg.name,
        sum(p.numel() for p in model.parameters()),
        tcfg.steps, tcfg.batch_size, tcfg.block_size, tcfg.lr, device_t,
    )

    losses: List[float] = []
    t0 = time.time()
    for step in range(int(tcfg.steps)):
        lr = _cosine_lr(step, tcfg)
        for g in optimiser.param_groups:
            g["lr"] = lr
        x, y = corpus.get_batch(tcfg.batch_size, tcfg.block_size)
        x = x.to(device_t, non_blocking=True)
        y = y.to(device_t, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            y.view(-1),
            ignore_index=-100,
        )
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        if tcfg.grad_clip and tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimiser.step()
        losses.append(float(loss.item()))
        if (step + 1) % int(tcfg.log_every) == 0 or step == 0:
            recent = sum(losses[-tcfg.log_every:]) / max(
                1, min(tcfg.log_every, len(losses))
            )
            elapsed = time.time() - t0
            _logger.info(
                "step %4d/%d  loss=%.4f  recent=%.4f  lr=%.2e  "
                "elapsed=%.1fs",
                step + 1, tcfg.steps, float(loss.item()), recent,
                lr, elapsed,
            )

    elapsed = time.time() - t0
    final_loss = sum(losses[-max(1, tcfg.log_every):]) / max(
        1, min(tcfg.log_every, len(losses))
    )
    _logger.info(
        "Training done: steps=%d, final_loss=%.4f, elapsed=%.1fs",
        tcfg.steps, final_loss, elapsed,
    )
    model.eval()

    if save:
        out = Path(
            tcfg.out_path
            or ("assets/checkpoints/tiny-transformer-{}.pt".format(cfg.name))
        ).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        save_tiny_transformer(model, tokenizer, out, config=cfg)
        _logger.info("Checkpoint written to %s (%.1fMB)",
                     out, out.stat().st_size / 1e6)
    return model, tokenizer, cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pretrain_tiny",
        description="Train a project-owned tiny Transformer LM (pure-torch).",
    )
    p.add_argument(
        "--preset", choices=("tiny", "small"), default="small",
        help="Model preset (default: small).",
    )
    p.add_argument(
        "--out", type=str, default=None,
        help="Output .pt path (default: assets/checkpoints/...).",
    )
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Training device (default: cpu).",
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Skip writing the checkpoint to disk.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = TINY_CONFIG if args.preset == "tiny" else SMALL_CONFIG
    tcfg = TrainConfig(preset=cfg.name)
    if args.steps is not None:
        tcfg.steps = int(args.steps)
    if args.batch_size is not None:
        tcfg.batch_size = int(args.batch_size)
    if args.block_size is not None:
        tcfg.block_size = int(args.block_size)
    if args.lr is not None:
        tcfg.lr = float(args.lr)
    if args.seed is not None:
        tcfg.seed = int(args.seed)
    if args.out is not None:
        tcfg.out_path = args.out
    train_tiny_transformer(
        config=cfg, train_cfg=tcfg, device=args.device, save=not args.no_save,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
