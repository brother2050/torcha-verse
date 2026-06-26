"""v1.0.0 CheckpointManager subsystem tests.

Exercises the seven-module
:mod:`infrastructure.checkpoint_manager` sub-package that is
otherwise 100% untested in the current suite:

* :func:`infrastructure.checkpoint_manager._state.capture_rng_states`
  / :func:`..._state.restore_rng_states` -- reproducible resume.
* :class:`infrastructure.checkpoint_manager._protocols.LocalCheckpointBackend`
  -- the default storage backend.
* :class:`infrastructure.checkpoint_manager.CheckpointManager`
  -- full + weights-only save / load round-trip + pruning policy.

5 tests; all CPU-only.
"""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from infrastructure.checkpoint_manager import (
    CheckpointManager,
    LocalCheckpointBackend,
    META_FILE,
)
from infrastructure.checkpoint_manager._manager import CheckpointManager as _CM
from infrastructure.checkpoint_manager._prune import list_checkpoints
from infrastructure.checkpoint_manager._state import (
    capture_rng_states,
    restore_rng_states,
)


# ---------------------------------------------------------------------------
# Section 1 -- CheckpointManager (3 tests)
# ---------------------------------------------------------------------------
class TestCheckpointManagerRoundtrip:
    """Full / weights-only save+load + pruning policy."""

    def test_save_and_load_full_checkpoint_roundtrip(self, tmp_path):
        """``save_checkpoint`` + ``load_checkpoint`` round-trip on a tiny
        model; metadata is written and survives the round-trip.

        The :class:`CheckpointManager` writes a ``training_state.pt``
        file alongside the weights.  Loading it back requires
        ``allow_unsafe_pickle=True`` (a safety net against pickle
        RCE), so the test opts in explicitly.
        """
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        cm = CheckpointManager(tmp_path)
        ckpt_dir = cm.save_checkpoint(
            model, optimizer=None, step=10,
            metadata={"foo": "bar"},
        )
        # metadata.json is present + contains our key.
        meta_path = ckpt_dir / META_FILE
        assert meta_path.is_file()
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        # The user-supplied metadata is merged into the top-level dict.
        assert meta.get("foo") == "bar"
        assert int(meta.get("step", -1)) == 10
        # Now build a fresh model and reload the weights + training state.
        model2 = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        # The manager guards the training-state pickle behind an
        # explicit opt-in flag.  The checkpoint we are loading was
        # produced by this same test, so opt-in is safe here.
        cm.load_checkpoint(ckpt_dir, model2, allow_unsafe_pickle=True)
        # The weights match exactly.
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters(),
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"weight {n1!r} did not round-trip"

    def test_save_weights_only_roundtrip(self, tmp_path):
        """``save_weights_only`` + ``load_weights`` round-trip on a tiny
        model; the destination file extension is picked automatically
        (``.safetensors`` when the library is available, ``.pt`` else)."""
        torch.manual_seed(1)
        model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
        # Snapshot the original weights for byte-equal comparison.
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}

        cm = CheckpointManager(tmp_path)
        target = tmp_path / "weights"
        written_path = cm.save_weights_only(model, target)
        assert written_path.is_file()

        # Build a fresh model with different weights.
        torch.manual_seed(2)
        model2 = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
        # Sanity check: the two models start out with different weights.
        assert not torch.equal(
            list(model2.parameters())[0],
            list(model.parameters())[0],
        )

        cm.load_weights(written_path, model2)
        # Every named parameter now matches the original.
        for k, v in original.items():
            assert k in dict(model2.named_parameters())
            actual = dict(model2.named_parameters())[k]
            assert torch.equal(v, actual), f"weight {k!r} did not round-trip"

    def test_prune_checkpoints_keeps_latest(self, tmp_path):
        """Saving 5 checkpoints with ``save_total_limit=3`` leaves
        only ``checkpoint-2``, ``checkpoint-3``, ``checkpoint-4``."""
        cm = CheckpointManager(tmp_path, save_total_limit=3)
        model = nn.Linear(4, 4)
        for step in range(5):
            cm.save_checkpoint(model, optimizer=None, step=step)
        # Use the public list_checkpoints helper to enumerate the survivors.
        survivors = list_checkpoints(tmp_path)
        # Only 3 directories should remain.
        assert len(survivors) == 3
        # In ascending step order: 2, 3, 4.
        steps = [int(p.name.split("-")[1]) for p in survivors]
        assert steps == [2, 3, 4]


# ---------------------------------------------------------------------------
# Section 2 -- RNG state capture / restore (1 test)
# ---------------------------------------------------------------------------
class TestRngStateCaptureRestore:
    """Reproducible resume: ``capture_rng_states`` / ``restore_rng_states``."""

    def test_capture_restore_rng_states(self):
        """``capture_rng_states`` -> ``restore_rng_states`` round-trip.

        The capture / restore pair is designed to make a
        subsequent :func:`torch.randn` call reproduce the
        random numbers that *would have been drawn* had the
        restore not happened.  Concretely:

        1. Seed + capture BEFORE the first draw.
        2. Draw ``x1``.
        3. Restore, draw again -> ``x2`` must equal ``x1``.
        4. Draw without restoring -> ``x3`` must differ.
        """
        torch.manual_seed(42)
        # 1) Capture the post-seed / pre-draw state.
        state = capture_rng_states()
        # 2) Draw the reference tensor.
        x1 = torch.randn(3)
        # 3) Restore + draw again -> must match x1.
        restore_rng_states(state)
        x2 = torch.randn(3)
        assert torch.equal(x1, x2), (
            f"expected x2 to match x1 after restore, "
            f"got x1={x1}, x2={x2}"
        )
        # 4) Draw once more without restoring -> the RNG has moved
        # on, so x3 must differ from x1.
        x3 = torch.randn(3)
        assert not torch.equal(x1, x3)


# ---------------------------------------------------------------------------
# Section 3 -- LocalCheckpointBackend (1 test)
# ---------------------------------------------------------------------------
class TestLocalCheckpointBackend:
    """The default :class:`LocalCheckpointBackend` storage backend."""

    def test_local_backend_write_read_exists(self, tmp_path):
        """``write`` / ``read`` / ``exists`` round-trip and
        :meth:`read` on a missing key raises ``FileNotFoundError``."""
        backend = LocalCheckpointBackend(root=tmp_path / "blobs")
        # write() creates the file.
        uri = backend.write("key1", b"data1")
        assert isinstance(uri, str)
        # exists() returns True for the new key.
        assert backend.exists("key1") is True
        # read() returns the original bytes.
        assert backend.read("key1") == b"data1"
        # exists() returns False for an unseen key.
        assert backend.exists("nonexistent") is False
        # read() on a missing key raises FileNotFoundError.
        with pytest.raises(FileNotFoundError):
            backend.read("nonexistent")
