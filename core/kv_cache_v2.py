"""Paged KV-Cache v2 for TorchaVerse v0.3.0.

This module replaces :mod:`core.kv_cache_manager` (v0.1.0) with a
corrected, thread-safe, paged KV-cache implementation.

Key improvements over v0.1.0:

* **Bug fix**: :meth:`PagedKVCache.evict_cold_blocks` now returns an
  :class:`EvictionResult` enum (``EVICTION_OK``,
  ``EVICTION_OFFLOADED``, ``EVICTION_FAILED``) instead of ``None``.
  In v0.1.0, ``_evict_cold_page`` returned ``None`` when a page was
  offloaded to CPU, causing the caller to believe eviction had failed
  even though GPU memory *was* freed.  The new design distinguishes
  "freed" from "offloaded" from "failed".

* **Block-level mutex**: each physical block has its own
  :class:`threading.RLock`, so concurrent ``append`` / ``get``
  operations on *different* blocks do not contend on a global lock.

* **Prefix caching**: when enabled, block-sized token chunks are
  indexed so that sequences sharing a common prefix can reuse the same
  physical blocks without recomputation.

* **CPU offload / reload**: blocks can be moved to CPU pinned memory
  and brought back on demand, freeing GPU memory without losing data.
"""

from __future__ import annotations

import enum
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

__all__ = [
    "EvictionResult",
    "KVCacheConfig",
    "KVCacheBlock",
    "PagedKVCache",
]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
#: Default block size (number of tokens per physical block).
_DEFAULT_BLOCK_SIZE: int = 16

#: Default maximum number of concurrent sequences the cache is sized for.
_DEFAULT_MAX_CONCURRENT_SEQS: int = 64

#: Default dtype for K/V tensors.
_DEFAULT_DTYPE: torch.dtype = torch.float32

#: Numerical-stability epsilon.
_EPSILON: float = 1e-8

#: LRU timestamp threshold multiplier (blocks older than this are "cold").
_DEFAULT_COLD_THRESHOLD_SECONDS: float = 30.0


# ---------------------------------------------------------------------------
# EvictionResult
# ---------------------------------------------------------------------------
class EvictionResult(enum.Enum):
    """Outcome of a cold-block eviction attempt.

    Attributes:
        EVICTION_OK: One or more cold blocks were freed (deleted from
            GPU memory).  Their block IDs are now available for reuse.
        EVICTION_FAILED: No cold blocks were available for eviction.
            The caller must handle the out-of-memory condition.
        EVICTION_OFFLOADED: Cold blocks were moved to CPU pinned
            memory rather than deleted.  GPU memory was freed, but the
            data is retained and can be reloaded with
            :meth:`PagedKVCache.reload_from_cpu`.
    """

    EVICTION_OK = "eviction_ok"
    EVICTION_FAILED = "eviction_failed"
    EVICTION_OFFLOADED = "eviction_offloaded"


# ---------------------------------------------------------------------------
# KVCacheConfig
# ---------------------------------------------------------------------------
@dataclass
class KVCacheConfig:
    """Configuration for :class:`PagedKVCache`.

    Attributes:
        num_layers: Number of transformer layers.
        num_heads: Number of attention query heads.
        num_kv_heads: Number of key/value heads (``< num_heads`` for
            grouped-query attention).
        head_dim: Dimension of each attention head.
        max_seq_len: Maximum sequence length the cache should support.
        block_size: Number of tokens per physical block.
        enable_prefix_cache: Whether to index block-sized token chunks
            for prefix reuse.
    """

    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    block_size: int = _DEFAULT_BLOCK_SIZE
    enable_prefix_cache: bool = True

    def __post_init__(self) -> None:
        """Validate configuration fields."""
        if self.num_layers <= 0:
            raise ValueError(
                "num_layers must be > 0, got {}.".format(self.num_layers)
            )
        if self.num_heads <= 0:
            raise ValueError(
                "num_heads must be > 0, got {}.".format(self.num_heads)
            )
        if self.num_kv_heads <= 0:
            raise ValueError(
                "num_kv_heads must be > 0, got {}.".format(self.num_kv_heads)
            )
        if self.head_dim <= 0:
            raise ValueError(
                "head_dim must be > 0, got {}.".format(self.head_dim)
            )
        if self.max_seq_len <= 0:
            raise ValueError(
                "max_seq_len must be > 0, got {}.".format(self.max_seq_len)
            )
        if self.block_size <= 0:
            raise ValueError(
                "block_size must be > 0, got {}.".format(self.block_size)
            )


# ---------------------------------------------------------------------------
# KVCacheBlock
# ---------------------------------------------------------------------------
@dataclass
class KVCacheBlock:
    """Metadata for a single physical cache block.

    Attributes:
        block_id: Unique integer block identifier.
        seq_ids: Set of sequence IDs that reference this block (for
            shared prefix blocks, this set has more than one member).
        ref_count: Number of live references (equals ``len(seq_ids)``).
        is_prefix: Whether this block is part of a shared prefix.
    """

    block_id: int
    seq_ids: Set[str] = field(default_factory=set)
    ref_count: int = 0
    is_prefix: bool = False


# ---------------------------------------------------------------------------
# Internal physical block storage
# ---------------------------------------------------------------------------
@dataclass
class _PhysicalBlock:
    """Internal storage for a physical K/V block.

    Attributes:
        key: Key tensor of shape
            ``(num_layers, block_size, num_kv_heads, head_dim)``.
        value: Value tensor of the same shape.
        on_cpu: Whether the block has been offloaded to CPU.
        last_used: Monotonic timestamp of last access (for LRU).
        tokens: Token IDs stored in this block (for prefix matching).
            ``None`` when prefix caching is disabled.
    """

    key: torch.Tensor
    value: torch.Tensor
    on_cpu: bool = False
    last_used: float = field(default_factory=time.monotonic)
    tokens: Optional[List[int]] = None


# ---------------------------------------------------------------------------
# PagedKVCache
# ---------------------------------------------------------------------------
class PagedKVCache:
    """Thread-safe paged KV-cache with block-level locking.

    Manages a pool of physical K/V blocks that are mapped to logical
    sequences.  Blocks can be shared between sequences (prefix caching),
    offloaded to CPU, and evicted when memory is tight.

    Thread safety is achieved with two lock layers:

    * **Global lock** (``_global_lock``): protects the block table,
      free list, sequence mappings, and prefix index.
    * **Per-block locks** (``_block_locks``): protect individual block
      data (K/V tensors, ``on_cpu`` flag, ``last_used``).

    This allows concurrent ``append`` / ``get`` operations on
    *different* blocks without global contention.
    """

    def __init__(
        self,
        config: KVCacheConfig,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = _DEFAULT_DTYPE,
        enable_cpu_offload: bool = True,
        max_concurrent_seqs: int = _DEFAULT_MAX_CONCURRENT_SEQS,
    ) -> None:
        """Initialise the paged KV-cache.

        Args:
            config: Cache configuration.
            device: Device for K/V tensors.  Defaults to the device
                reported by :class:`~infrastructure.device_manager.DeviceManager`.
            dtype: Element dtype for K/V tensors.
            enable_cpu_offload: Whether cold blocks can be offloaded
                to CPU pinned memory during eviction.
            max_concurrent_seqs: Maximum number of concurrent sequences
                the cache is sized for.
        """
        self._config: KVCacheConfig = config
        self._device_manager: DeviceManager = DeviceManager()
        self._device: torch.device = (
            device if device is not None else self._device_manager.get_device()
        )
        self._dtype: torch.dtype = dtype
        self._enable_cpu_offload: bool = enable_cpu_offload
        self._logger = get_logger(self.__class__.__name__)

        # Compute the maximum number of physical blocks.
        blocks_per_seq: int = math.ceil(
            config.max_seq_len / config.block_size
        )
        self._max_blocks: int = blocks_per_seq * max_concurrent_seqs

        # Physical block storage.
        self._blocks: Dict[int, _PhysicalBlock] = {}
        self._block_meta: Dict[int, KVCacheBlock] = {}

        # Sequence mappings.
        self._seq_blocks: Dict[str, List[int]] = {}
        self._seq_lengths: Dict[str, int] = {}

        # Free list (stack for LIFO allocation).
        self._free_blocks: List[int] = list(range(self._max_blocks))

        # Per-block locks.
        self._block_locks: Dict[int, threading.RLock] = {
            i: threading.RLock() for i in range(self._max_blocks)
        }

        # Global lock for table-level operations.
        self._global_lock: threading.RLock = threading.RLock()

        # Prefix cache index: token-chunk tuple -> block_id.
        self._prefix_index: Dict[Tuple[int, ...], int] = {}

        # Statistics.
        self._stat_allocations: int = 0
        self._stat_prefix_hits: int = 0
        self._stat_prefix_misses: int = 0
        self._stat_evictions: int = 0
        self._stat_offloads: int = 0

        self._logger.debug(
            "PagedKVCache initialised: max_blocks=%d, block_size=%d, "
            "num_layers=%d, num_kv_heads=%d, head_dim=%d.",
            self._max_blocks, config.block_size,
            config.num_layers, config.num_kv_heads, config.head_dim,
        )

    # ------------------------------------------------------------------
    # Block shape helper
    # ------------------------------------------------------------------
    @property
    def _block_shape(self) -> Tuple[int, int, int, int]:
        """Return the tensor shape of a single physical block."""
        c = self._config
        return (c.num_layers, c.block_size, c.num_kv_heads, c.head_dim)

    # ------------------------------------------------------------------
    # allocate
    # ------------------------------------------------------------------
    def allocate(self, seq_id: str, num_tokens: int) -> List[int]:
        """Allocate physical blocks for a sequence.

        If prefix caching is enabled, the method first attempts to
        reuse existing blocks whose token content matches the
        sequence's prefix.

        Args:
            seq_id: Unique sequence identifier.
            num_tokens: Number of tokens to allocate capacity for.

        Returns:
            A list of allocated block IDs.

        Raises:
            ValueError: If *num_tokens* is non-positive.
            RuntimeError: If no blocks are available even after
                eviction.
        """
        if num_tokens <= 0:
            raise ValueError(
                "num_tokens must be > 0, got {}.".format(num_tokens)
            )

        num_needed: int = math.ceil(
            num_tokens / self._config.block_size
        )

        with self._global_lock:
            allocated: List[int] = []
            for _ in range(num_needed):
                block_id = self._allocate_block()
                allocated.append(block_id)

            self._seq_blocks[seq_id] = allocated
            self._seq_lengths[seq_id] = 0
            self._stat_allocations += num_needed
            return allocated

    def _allocate_block(self) -> int:
        """Get a free block, evicting if necessary.

        Must be called while holding ``_global_lock``.

        Returns:
            A free block ID.

        Raises:
            RuntimeError: If no blocks are available even after
                eviction.
        """
        if self._free_blocks:
            return self._free_blocks.pop()

        # Try to evict cold blocks.
        result = self._evict_cold_blocks_internal(
            _DEFAULT_COLD_THRESHOLD_SECONDS
        )
        if result == EvictionResult.EVICTION_OK:
            # Blocks were freed; try again.
            if self._free_blocks:
                return self._free_blocks.pop()
        elif result == EvictionResult.EVICTION_OFFLOADED:
            # Blocks were offloaded to CPU; their GPU slots are now
            # available for reuse (new tensors will be allocated).
            # Mark the offloaded blocks as free for reuse.
            if self._free_blocks:
                return self._free_blocks.pop()
            # If offloading freed GPU memory but didn't add to the free
            # list (blocks are retained on CPU), we can reuse their
            # block IDs by clearing the CPU copy.
            for bid, pb in list(self._blocks.items()):
                meta = self._block_meta.get(bid)
                if meta is not None and meta.ref_count <= 0 and pb.on_cpu:
                    # Reuse this block ID.
                    del self._blocks[bid]
                    self._block_meta[bid] = KVCacheBlock(block_id=bid)
                    return bid

        raise RuntimeError(
            "PagedKVCache out of blocks: no free blocks and eviction "
            "failed (max_blocks={}).".format(self._max_blocks)
        )

    # ------------------------------------------------------------------
    # append
    # ------------------------------------------------------------------
    def append(
        self,
        seq_id: str,
        key: torch.Tensor,
        value: torch.Tensor,
        tokens: Optional[List[int]] = None,
    ) -> None:
        """Append K/V data for a sequence.

        Writes the provided *key* and *value* tensors into the
        sequence's allocated blocks at the appropriate positions.

        Args:
            seq_id: Sequence identifier.
            key: Key tensor of shape
                ``(num_layers, num_new_tokens, num_kv_heads, head_dim)``.
            value: Value tensor of the same shape.
            tokens: Optional token IDs corresponding to the new K/V
                data (used for prefix cache indexing).

        Raises:
            KeyError: If *seq_id* has no allocated blocks.
            RuntimeError: If more tokens are appended than the
                allocated capacity supports.
        """
        with self._global_lock:
            if seq_id not in self._seq_blocks:
                raise KeyError(
                    "Sequence '{}' has no allocated blocks.".format(seq_id)
                )
            block_ids = list(self._seq_blocks[seq_id])
            current_len = self._seq_lengths[seq_id]
            block_size = self._config.block_size
            num_new = key.shape[1]

            max_capacity = len(block_ids) * block_size
            if current_len + num_new > max_capacity:
                raise RuntimeError(
                    "Sequence '{}' overflow: current={}, new={}, "
                    "capacity={}.".format(
                        seq_id, current_len, num_new, max_capacity
                    )
                )

            # Write data block by block.
            token_offset = 0
            for i, bid in enumerate(block_ids):
                block_start = i * block_size
                block_end = block_start + block_size
                # Does this block overlap with [current_len, current_len+num_new)?
                write_start = max(current_len, block_start)
                write_end = min(current_len + num_new, block_end)
                if write_start >= write_end:
                    continue

                local_start = write_start - block_start
                local_end = write_end - block_start
                data_start = write_start - current_len
                data_end = write_end - current_len

                with self._block_locks[bid]:
                    pb = self._blocks.get(bid)
                    if pb is None:
                        pb = self._create_physical_block()
                        self._blocks[bid] = pb
                    # Reload from CPU if needed.
                    if pb.on_cpu:
                        self._reload_block_from_cpu(bid, pb)
                    pb.key[:, local_start:local_end, :, :] = (
                        key[:, data_start:data_end, :, :].to(self._device)
                    )
                    pb.value[:, local_start:local_end, :, :] = (
                        value[:, data_start:data_end, :, :].to(self._device)
                    )
                    pb.last_used = time.monotonic()

                    # Update prefix index.
                    if (
                        tokens is not None
                        and self._config.enable_prefix_cache
                        and local_start == 0
                        and local_end == block_size
                    ):
                        chunk = tuple(
                            tokens[token_offset: token_offset + block_size]
                        )
                        token_offset += block_size
                        pb.tokens = list(chunk)
                        if chunk not in self._prefix_index:
                            self._prefix_index[chunk] = bid
                            self._block_meta[bid].is_prefix = True

            self._seq_lengths[seq_id] = current_len + num_new

    # ------------------------------------------------------------------
    # get
    # ------------------------------------------------------------------
    def get(self, seq_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve all K/V data for a sequence.

        Args:
            seq_id: Sequence identifier.

        Returns:
            A tuple ``(keys, values)`` where each tensor has shape
            ``(num_layers, seq_len, num_kv_heads, head_dim)``.

        Raises:
            KeyError: If *seq_id* has no allocated blocks.
        """
        with self._global_lock:
            if seq_id not in self._seq_blocks:
                raise KeyError(
                    "Sequence '{}' has no allocated blocks.".format(seq_id)
                )
            block_ids = list(self._seq_blocks[seq_id])
            seq_len = self._seq_lengths[seq_id]
            block_size = self._config.block_size

            keys_list: List[torch.Tensor] = []
            values_list: List[torch.Tensor] = []
            remaining = seq_len

            for bid in block_ids:
                with self._block_locks[bid]:
                    pb = self._blocks.get(bid)
                    if pb is None:
                        # Block never written; return zeros.
                        chunk_len = min(block_size, remaining)
                        keys_list.append(
                            torch.zeros(
                                (self._config.num_layers, chunk_len,
                                 self._config.num_kv_heads, self._config.head_dim),
                                device=self._device, dtype=self._dtype,
                            )
                        )
                        values_list.append(
                            torch.zeros(
                                (self._config.num_layers, chunk_len,
                                 self._config.num_kv_heads, self._config.head_dim),
                                device=self._device, dtype=self._dtype,
                            )
                        )
                    else:
                        if pb.on_cpu:
                            self._reload_block_from_cpu(bid, pb)
                        chunk_len = min(block_size, remaining)
                        keys_list.append(pb.key[:, :chunk_len, :, :].clone())
                        values_list.append(pb.value[:, :chunk_len, :, :].clone())
                        pb.last_used = time.monotonic()
                remaining -= chunk_len
                if remaining <= 0:
                    break

            keys = torch.cat(keys_list, dim=1)
            values = torch.cat(values_list, dim=1)
            return keys, values

    # ------------------------------------------------------------------
    # free
    # ------------------------------------------------------------------
    def free(self, seq_id: str) -> None:
        """Free all blocks held by a sequence.

        Blocks whose reference count drops to zero are returned to the
        free list (unless they are prefix-shared by other sequences).

        Args:
            seq_id: Sequence identifier.

        Raises:
            KeyError: If *seq_id* is not allocated.
        """
        with self._global_lock:
            if seq_id not in self._seq_blocks:
                raise KeyError(
                    "Sequence '{}' is not allocated.".format(seq_id)
                )
            block_ids = self._seq_blocks.pop(seq_id)
            self._seq_lengths.pop(seq_id, None)

            for bid in block_ids:
                meta = self._block_meta.get(bid)
                if meta is None:
                    continue
                meta.seq_ids.discard(seq_id)
                meta.ref_count = len(meta.seq_ids)
                if meta.ref_count <= 0 and not meta.is_prefix:
                    # Return to free list.
                    if bid not in self._free_blocks:
                        self._free_blocks.append(bid)
                    # Look up the physical block *before* dropping it so
                    # the prefix-index cleanup below can actually see
                    # the tokens.  The previous order-of-operations bug
                    # (``del self._blocks[bid]`` then ``self._blocks.get``)
                    # meant the prefix index was never cleaned up.
                    pb = self._blocks.pop(bid, None)
                    if pb is None:
                        # Nothing to clean up; refcount only entry
                        # existed.  Move on.
                        continue
                    # Remove from prefix index now that we still have
                    # the block reference.
                    if pb.tokens is not None:
                        chunk = tuple(pb.tokens)
                        if chunk in self._prefix_index:
                            del self._prefix_index[chunk]

    # ------------------------------------------------------------------
    # evict_cold_blocks  (THE BUG FIX)
    # ------------------------------------------------------------------
    def evict_cold_blocks(
        self, threshold: float = _DEFAULT_COLD_THRESHOLD_SECONDS
    ) -> EvictionResult:
        """Evict cold (unused) blocks to free GPU memory.

        This is the public entry point.  It returns an
        :class:`EvictionResult` indicating the outcome:

        * :attr:`EvictionResult.EVICTION_OK` -- blocks were deleted;
          their IDs are now in the free list.
        * :attr:`EvictionResult.EVICTION_OFFLOADED` -- blocks were
          moved to CPU pinned memory; GPU memory was freed but the
          data is retained.
        * :attr:`EvictionResult.EVICTION_FAILED` -- no cold blocks were
          available.

        **Bug fix (v0.1.0)**: In the old ``_evict_cold_page``, when a
        page was offloaded to CPU the method returned ``None``, causing
        the caller to believe eviction had failed.  The new design
        returns :attr:`EvictionResult.EVICTION_OFFLOADED` so the caller
        knows GPU memory *was* freed.

        Args:
            threshold: A block is "cold" if it has not been accessed
                for at least this many seconds.

        Returns:
            The eviction outcome.
        """
        with self._global_lock:
            return self._evict_cold_blocks_internal(threshold)

    def _evict_cold_blocks_internal(
        self, threshold: float
    ) -> EvictionResult:
        """Internal eviction logic (caller must hold ``_global_lock``).

        Finds blocks with ``ref_count <= 0`` that haven't been accessed
        recently.  If CPU offload is enabled, offloads them and returns
        :attr:`EvictionResult.EVICTION_OFFLOADED`; otherwise frees them
        and returns :attr:`EvictionResult.EVICTION_OK`.  Returns
        :attr:`EvictionResult.EVICTION_FAILED` when no candidates exist.
        """
        now = time.monotonic()
        candidates: List[Tuple[int, _PhysicalBlock]] = []
        for bid, meta in self._block_meta.items():
            if meta.ref_count > 0:
                continue
            pb = self._blocks.get(bid)
            if pb is None:
                # No physical data; just free the ID.
                if bid not in self._free_blocks:
                    self._free_blocks.append(bid)
                continue
            if now - pb.last_used >= threshold:
                candidates.append((bid, pb))

        if not candidates:
            return EvictionResult.EVICTION_FAILED

        # Sort by last_used ascending (coldest first).
        candidates.sort(key=lambda x: x[1].last_used)

        if self._enable_cpu_offload:
            # Offload to CPU pinned memory.
            for bid, pb in candidates:
                with self._block_locks[bid]:
                    if not pb.on_cpu:
                        self._offload_block_to_cpu(bid, pb)
            self._stat_offloads += len(candidates)
            self._logger.debug(
                "Offloaded %d cold block(s) to CPU.", len(candidates)
            )
            return EvictionResult.EVICTION_OFFLOADED
        else:
            # Free the blocks entirely.
            for bid, pb in candidates:
                with self._block_locks[bid]:
                    del self._blocks[bid]
                if bid not in self._free_blocks:
                    self._free_blocks.append(bid)
                # Remove from prefix index.
                if pb.tokens is not None:
                    chunk = tuple(pb.tokens)
                    if chunk in self._prefix_index:
                        del self._prefix_index[chunk]
            self._stat_evictions += len(candidates)
            self._logger.debug(
                "Freed %d cold block(s).", len(candidates)
            )
            return EvictionResult.EVICTION_OK

    # ------------------------------------------------------------------
    # offload_to_cpu
    # ------------------------------------------------------------------
    def offload_to_cpu(self, block_ids: List[int]) -> int:
        """Offload specific blocks to CPU pinned memory.

        Args:
            block_ids: List of block IDs to offload.

        Returns:
            The number of blocks successfully offloaded.
        """
        count = 0
        for bid in block_ids:
            with self._global_lock:
                pb = self._blocks.get(bid)
                if pb is None or pb.on_cpu:
                    continue
            with self._block_locks[bid]:
                pb = self._blocks.get(bid)
                if pb is None or pb.on_cpu:
                    continue
                self._offload_block_to_cpu(bid, pb)
                count += 1
        self._stat_offloads += count
        self._logger.debug("Offloaded %d block(s) to CPU.", count)
        return count

    def _offload_block_to_cpu(
        self, block_id: int, pb: _PhysicalBlock
    ) -> None:
        """Move a single block's tensors to CPU pinned memory.

        Caller must hold the block's per-block lock.
        """
        try:
            pb.key = pb.key.to("cpu", non_blocking=True).pin_memory()
            pb.value = pb.value.to("cpu", non_blocking=True).pin_memory()
        except (RuntimeError, AttributeError):
            # Fallback: regular CPU transfer.
            pb.key = pb.key.cpu()
            pb.value = pb.value.cpu()
        pb.on_cpu = True
        self._logger.debug("Block %d offloaded to CPU.", block_id)

    # ------------------------------------------------------------------
    # reload_from_cpu
    # ------------------------------------------------------------------
    def reload_from_cpu(self, block_ids: List[int]) -> int:
        """Reload specific blocks from CPU back to GPU.

        Args:
            block_ids: List of block IDs to reload.

        Returns:
            The number of blocks successfully reloaded.
        """
        count = 0
        for bid in block_ids:
            with self._block_locks[bid]:
                pb = self._blocks.get(bid)
                if pb is None or not pb.on_cpu:
                    continue
                self._reload_block_from_cpu(bid, pb)
                count += 1
        self._logger.debug("Reloaded %d block(s) from CPU.", count)
        return count

    def _reload_block_from_cpu(
        self, block_id: int, pb: _PhysicalBlock
    ) -> None:
        """Move a single block's tensors back to GPU.

        Caller must hold the block's per-block lock.
        """
        pb.key = pb.key.to(self._device, non_blocking=True)
        pb.value = pb.value.to(self._device, non_blocking=True)
        pb.on_cpu = False
        pb.last_used = time.monotonic()
        self._logger.debug("Block %d reloaded from CPU.", block_id)

    # ------------------------------------------------------------------
    # prefix_match
    # ------------------------------------------------------------------
    def prefix_match(self, tokens: List[int]) -> List[int]:
        """Find reusable prefix blocks for a token sequence.

        Splits *tokens* into block-sized chunks and looks up each
        chunk in the prefix index.  Returns the block IDs of all
        consecutive matching chunks from the beginning.

        Args:
            tokens: Token IDs to match.

        Returns:
            A list of block IDs whose token content matches the prefix
            of *tokens*.  May be empty if no prefix matches.
        """
        if not self._config.enable_prefix_cache:
            return []

        block_size = self._config.block_size
        matched: List[int] = []

        with self._global_lock:
            for i in range(0, len(tokens), block_size):
                chunk = tuple(tokens[i: i + block_size])
                if len(chunk) < block_size:
                    break  # Partial chunk; can't match a full block.
                bid = self._prefix_index.get(chunk)
                if bid is None:
                    self._stat_prefix_misses += 1
                    break
                matched.append(bid)
                self._stat_prefix_hits += 1

        if matched:
            self._logger.debug(
                "Prefix match: %d block(s) reused.", len(matched)
            )
        return matched

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics.

        Returns:
            A dictionary with the following keys:

            * ``blocks_used``: Number of blocks currently in use.
            * ``blocks_free``: Number of free blocks.
            * ``hit_rate``: Prefix cache hit rate (0.0 -- 1.0).
            * ``evictions``: Total number of evicted blocks.
            * ``offloads``: Total number of CPU offloads.
            * ``total_blocks``: Total block capacity.
        """
        with self._global_lock:
            blocks_used = self._max_blocks - len(self._free_blocks)
            blocks_free = len(self._free_blocks)
            total_lookups = self._stat_prefix_hits + self._stat_prefix_misses
            hit_rate = (
                self._stat_prefix_hits / total_lookups
                if total_lookups > 0
                else 0.0
            )
            return {
                "blocks_used": blocks_used,
                "blocks_free": blocks_free,
                "hit_rate": hit_rate,
                "evictions": self._stat_evictions,
                "offloads": self._stat_offloads,
                "total_blocks": self._max_blocks,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_physical_block(self) -> _PhysicalBlock:
        """Create a new zero-initialised physical block."""
        return _PhysicalBlock(
            key=torch.zeros(
                self._block_shape, device=self._device, dtype=self._dtype
            ),
            value=torch.zeros(
                self._block_shape, device=self._device, dtype=self._dtype
            ),
        )

    def __repr__(self) -> str:
        s = self.stats()
        return (
            "PagedKVCache(blocks_used={}, blocks_free={}, "
            "hit_rate={:.3f}, evictions={})".format(
                s["blocks_used"], s["blocks_free"],
                s["hit_rate"], s["evictions"],
            )
        )
