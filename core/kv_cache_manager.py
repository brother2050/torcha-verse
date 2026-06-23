"""Key-value cache management for autoregressive generation.

This module provides :class:`KVCacheManager`, the core optimisation layer
for autoregressive text (and multi-modal) generation.  It supports two
caching strategies:

* **Static** -- Pre-allocates a contiguous block of GPU memory sized by
  ``max_batch_size x max_seq_len x num_layers x head_dim``.  This avoids
  per-step allocation overhead but wastes memory when the actual sequence
  is shorter than the maximum.

* **Paged** -- Inspired by vLLM's PagedAttention, divides the KV cache
  into fixed-size *pages* that are allocated on demand.  This eliminates
  internal fragmentation and enables sharing of prefix pages across
  requests with common prefixes.

Additional features:

* **Prefix sharing** -- Requests with identical token prefixes share the
  same physical pages, avoiding redundant computation.
* **CPU offloading** -- Low-frequency (cold) pages can be evicted to CPU
  memory and brought back on demand.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

__all__ = ["KVCacheManager", "CacheStrategy"]

#: Type alias for a (key, value) tuple.
KVPair = Tuple[torch.Tensor, torch.Tensor]


class CacheStrategy:
    """Supported KV cache strategies."""

    STATIC: str = "static"
    PAGED: str = "paged"


# ---------------------------------------------------------------------------
# Page table entry (for paged strategy)
# ---------------------------------------------------------------------------
class _PageBlock:
    """A single page block in the paged cache."""

    __slots__ = ("page_id", "key", "value", "ref_count", "on_cpu", "last_used")

    def __init__(
        self,
        page_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.page_id: int = page_id
        self.key: torch.Tensor = key
        self.value: torch.Tensor = value
        self.ref_count: int = 0
        self.on_cpu: bool = False
        self.last_used: float = 0.0


# ---------------------------------------------------------------------------
# KVCacheManager
# ---------------------------------------------------------------------------
class KVCacheManager:
    """Manage key-value caches for autoregressive generation.

    The manager supports both static (pre-allocated) and paged
    (on-demand) strategies.  It tracks per-batch, per-layer caches and
    provides a uniform ``update`` / ``get`` / ``free`` API.

    Args:
        strategy: ``"static"`` or ``"paged"``.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads (KV heads for GQA).
        head_dim: Dimension of each attention head.
        max_batch_size: Maximum batch size (static strategy).
        max_seq_len: Maximum sequence length (static strategy).
        page_size: Number of tokens per page (paged strategy).
        max_pages: Maximum number of pages (paged strategy).
        device: Device for the cache tensors.
        dtype: Data type for the cache tensors.
        cpu_offload: Whether to enable CPU offloading for cold pages.
        offload_threshold: GPU memory usage fraction that triggers offload.
    """

    def __init__(
        self,
        strategy: str = CacheStrategy.STATIC,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        page_size: int = 16,
        max_pages: int = 1024,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        cpu_offload: bool = False,
        offload_threshold: float = 0.85,
    ) -> None:
        if strategy not in (CacheStrategy.STATIC, CacheStrategy.PAGED):
            raise ValueError(
                f"Unknown strategy '{strategy}'. Use 'static' or 'paged'."
            )

        self.strategy: str = strategy
        self.num_layers: int = num_layers
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.max_batch_size: int = max_batch_size
        self.max_seq_len: int = max_seq_len
        self.page_size: int = page_size
        self.max_pages: int = max_pages
        self.dtype: torch.dtype = dtype
        self.cpu_offload: bool = cpu_offload
        self.offload_threshold: float = offload_threshold

        self._device_manager: DeviceManager = DeviceManager()
        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device or self._device_manager.get_device()
        )
        self._logger = get_logger(self.__class__.__name__)

        # State for static strategy.
        self._static_k: Optional[torch.Tensor] = None
        self._static_v: Optional[torch.Tensor] = None
        self._static_lengths: List[int] = []  # current length per batch entry
        self._static_allocated: bool = False

        # State for paged strategy.
        self._pages: Dict[int, _PageBlock] = {}
        self._next_page_id: int = 0
        # Per-batch page tables: batch_idx -> list of page_ids per layer.
        self._page_tables: Dict[int, List[List[int]]] = {}
        # Prefix sharing: token_hash -> page_id.
        self._prefix_cache: Dict[int, List[int]] = {}

        if strategy == CacheStrategy.STATIC:
            self._preallocate_static()

    # ------------------------------------------------------------------
    # Static strategy
    # ------------------------------------------------------------------
    def _preallocate_static(self) -> None:
        """Pre-allocate the static KV cache tensors.

        Shapes:
            key_cache: (max_batch_size, num_layers, num_heads, max_seq_len, head_dim)
            value_cache: same shape
        """
        shape = (
            self.max_batch_size,
            self.num_layers,
            self.num_heads,
            self.max_seq_len,
            self.head_dim,
        )
        self._static_k = torch.zeros(shape, dtype=self.dtype, device=self._device)
        self._static_v = torch.zeros(shape, dtype=self.dtype, device=self._device)
        self._static_lengths = [0] * self.max_batch_size
        self._static_allocated = True
        self._logger.debug(
            "Pre-allocated static KV cache: shape=%s, dtype=%s, device=%s "
            "(%.2f MB)",
            shape,
            self.dtype,
            self._device,
            (self._static_k.nelement() * self._static_k.element_size() * 2) / 1e6,
        )

    # ------------------------------------------------------------------
    # Public API: allocation
    # ------------------------------------------------------------------
    def allocate(self, batch_size: int, seq_len: int) -> None:
        """Allocate cache space for a batch.

        For the **static** strategy this resets the per-batch length
        counters.  For the **paged** strategy this initialises the page
        tables for each batch entry.

        Args:
            batch_size: Number of sequences in the batch.
            seq_len: Expected sequence length (used for capacity checks).

        Raises:
            RuntimeError: If the requested size exceeds the pre-allocated
                capacity (static strategy).
        """
        if self.strategy == CacheStrategy.STATIC:
            if not self._static_allocated:
                self._preallocate_static()
            if batch_size > self.max_batch_size:
                raise RuntimeError(
                    f"batch_size {batch_size} exceeds max_batch_size "
                    f"{self.max_batch_size}."
                )
            if seq_len > self.max_seq_len:
                raise RuntimeError(
                    f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}."
                )
            self._static_lengths = [0] * batch_size
        else:
            # Paged: initialise page tables.
            for i in range(batch_size):
                if i not in self._page_tables:
                    self._page_tables[i] = [[] for _ in range(self.num_layers)]

    # ------------------------------------------------------------------
    # Public API: update
    # ------------------------------------------------------------------
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_idx: int = 0,
        start_pos: Optional[int] = None,
    ) -> None:
        """Update the KV cache for a given layer and batch entry.

        Args:
            layer_idx: Index of the transformer layer.
            key: Key tensor of shape ``(batch, num_heads, seq_len, head_dim)``
                or ``(num_heads, seq_len, head_dim)``.
            value: Value tensor with the same shape as ``key``.
            batch_idx: Index of the batch entry to update.
            start_pos: Position at which to write the new tokens.  When
                ``None`` the current length is used (append mode).

        Raises:
            IndexError: If ``layer_idx`` or ``batch_idx`` is out of range.
            RuntimeError: If the cache is not allocated.
        """
        if self.strategy == CacheStrategy.STATIC:
            self._update_static(layer_idx, key, value, batch_idx, start_pos)
        else:
            self._update_paged(layer_idx, key, value, batch_idx)

    def _update_static(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_idx: int,
        start_pos: Optional[int],
    ) -> None:
        """Update the static cache."""
        if self._static_k is None or self._static_v is None:
            raise RuntimeError("Static cache has not been allocated. Call allocate() first.")

        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers}).")
        if batch_idx < 0 or batch_idx >= self.max_batch_size:
            raise IndexError(f"batch_idx {batch_idx} out of range [0, {self.max_batch_size}).")

        # Normalise key/value to (num_heads, seq_len, head_dim).
        if key.dim() == 4:
            # (batch, num_heads, seq_len, head_dim) -> take the right batch.
            key = key[batch_idx]
            value = value[batch_idx]

        seq_len = key.shape[1]
        pos = start_pos if start_pos is not None else self._static_lengths[batch_idx]

        if pos + seq_len > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: pos={pos}, seq_len={seq_len}, "
                f"max_seq_len={self.max_seq_len}."
            )

        # Write into the pre-allocated tensor.
        self._static_k[batch_idx, layer_idx, :, pos : pos + seq_len, :] = key.to(
            self._device, self.dtype
        )
        self._static_v[batch_idx, layer_idx, :, pos : pos + seq_len, :] = value.to(
            self._device, self.dtype
        )
        self._static_lengths[batch_idx] = pos + seq_len

    def _update_paged(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """Update the paged cache by allocating pages as needed."""
        if batch_idx not in self._page_tables:
            self._page_tables[batch_idx] = [[] for _ in range(self.num_layers)]

        if key.dim() == 4:
            key = key[batch_idx]
            value = value[batch_idx]

        num_heads, seq_len, head_dim = key.shape
        key = key.to(self._device, self.dtype)
        value = value.to(self._device, self.dtype)

        # Split the new tokens into pages.
        num_new_pages = (seq_len + self.page_size - 1) // self.page_size
        page_table = self._page_tables[batch_idx][layer_idx]

        for page_num in range(num_new_pages):
            start = page_num * self.page_size
            end = min(start + self.page_size, seq_len)
            chunk_len = end - start

            # Allocate or reuse a page.
            page_id = self._allocate_page(num_heads, head_dim)
            page_block = self._pages[page_id]
            page_block.ref_count += 1
            page_block.key[:, :chunk_len, :] = key[:, start:end, :]
            page_block.value[:, :chunk_len, :] = value[:, start:end, :]
            page_table.append(page_id)

    # ------------------------------------------------------------------
    # Public API: get
    # ------------------------------------------------------------------
    def get(
        self,
        layer_idx: int,
        batch_idx: int = 0,
    ) -> KVPair:
        """Retrieve the cached (key, value) for a layer and batch entry.

        Args:
            layer_idx: Index of the transformer layer.
            batch_idx: Index of the batch entry.

        Returns:
            A tuple ``(key, value)`` of tensors.  For the static strategy
            the shape is ``(num_heads, current_len, head_dim)``; for the
            paged strategy the pages are concatenated.

        Raises:
            IndexError: If indices are out of range.
            RuntimeError: If the cache is not allocated.
        """
        if self.strategy == CacheStrategy.STATIC:
            return self._get_static(layer_idx, batch_idx)
        return self._get_paged(layer_idx, batch_idx)

    def _get_static(self, layer_idx: int, batch_idx: int) -> KVPair:
        """Retrieve from the static cache."""
        if self._static_k is None or self._static_v is None:
            raise RuntimeError("Static cache has not been allocated. Call allocate() first.")
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range.")
        if batch_idx < 0 or batch_idx >= self.max_batch_size:
            raise IndexError(f"batch_idx {batch_idx} out of range.")

        length = self._static_lengths[batch_idx]
        key = self._static_k[batch_idx, layer_idx, :, :length, :]
        value = self._static_v[batch_idx, layer_idx, :, :length, :]
        return key, value

    def _get_paged(self, layer_idx: int, batch_idx: int) -> KVPair:
        """Retrieve from the paged cache, concatenating pages."""
        if batch_idx not in self._page_tables:
            raise RuntimeError(f"No cache allocated for batch_idx {batch_idx}.")

        page_ids = self._page_tables[batch_idx][layer_idx]
        if not page_ids:
            # Return empty tensors.
            return (
                torch.empty(self.num_heads, 0, self.head_dim, device=self._device, dtype=self.dtype),
                torch.empty(self.num_heads, 0, self.head_dim, device=self._device, dtype=self.dtype),
            )

        keys: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        for pid in page_ids:
            block = self._pages[pid]
            # Bring back from CPU if offloaded.
            if block.on_cpu:
                self._reload_from_cpu(pid)
            keys.append(block.key)
            values.append(block.value)
            block.last_used = torch.tensor(0.0)  # mark as recently used

        # Concatenate along the sequence dimension.
        full_key = torch.cat(keys, dim=1)
        full_value = torch.cat(values, dim=1)
        return full_key, full_value

    # ------------------------------------------------------------------
    # Public API: free
    # ------------------------------------------------------------------
    def free(self, batch_idx: int) -> None:
        """Release the cache for a specific batch entry.

        For the paged strategy this decrements page reference counts and
        frees pages whose count drops to zero.  For the static strategy
        this simply resets the length counter.

        Args:
            batch_idx: Index of the batch entry to free.
        """
        if self.strategy == CacheStrategy.STATIC:
            if 0 <= batch_idx < len(self._static_lengths):
                self._static_lengths[batch_idx] = 0
        else:
            if batch_idx in self._page_tables:
                for layer_pages in self._page_tables[batch_idx]:
                    for pid in layer_pages:
                        if pid in self._pages:
                            self._pages[pid].ref_count -= 1
                            if self._pages[pid].ref_count <= 0:
                                del self._pages[pid]
                del self._page_tables[batch_idx]

    def free_all(self) -> None:
        """Release all cached data."""
        if self.strategy == CacheStrategy.STATIC:
            self._static_lengths = [0] * self.max_batch_size
        else:
            self._pages.clear()
            self._page_tables.clear()
            self._prefix_cache.clear()
            self._next_page_id = 0

    # ------------------------------------------------------------------
    # Paged strategy internals
    # ------------------------------------------------------------------
    def _allocate_page(self, num_heads: int, head_dim: int) -> int:
        """Allocate a new page block and return its id."""
        if len(self._pages) >= self.max_pages:
            # Try to evict a cold page.
            evicted = self._evict_cold_page()
            if evicted is None:
                raise RuntimeError(
                    f"Page limit reached ({self.max_pages}) and no evictable pages."
                )
            return evicted

        page_id = self._next_page_id
        self._next_page_id += 1

        page_key = torch.zeros(
            num_heads, self.page_size, head_dim,
            dtype=self.dtype, device=self._device,
        )
        page_value = torch.zeros_like(page_key)
        self._pages[page_id] = _PageBlock(page_id, page_key, page_value)
        return page_id

    def _evict_cold_page(self) -> Optional[int]:
        """Find and evict the coldest (least recently used) free page."""
        candidates = [
            (pid, block) for pid, block in self._pages.items()
            if block.ref_count <= 0
        ]
        if not candidates:
            return None

        # Sort by last_used (ascending = coldest first).
        candidates.sort(key=lambda x: x[1].last_used)
        pid, block = candidates[0]

        # If CPU offload is enabled, move to CPU instead of deleting.
        if self.cpu_offload and not block.on_cpu:
            block.key = block.key.cpu()
            block.value = block.value.cpu()
            block.on_cpu = True
            self._logger.debug("Offloaded page %d to CPU.", pid)
            # Still return None because the page is retained (just on CPU).
            # We need to allocate a new page.
            return None

        del self._pages[pid]
        return pid

    def _reload_from_cpu(self, page_id: int) -> None:
        """Move a page back from CPU to GPU."""
        block = self._pages[page_id]
        if block.on_cpu:
            block.key = block.key.to(self._device)
            block.value = block.value.to(self._device)
            block.on_cpu = False
            self._logger.debug("Reloaded page %d from CPU to GPU.", page_id)

    # ------------------------------------------------------------------
    # Prefix sharing
    # ------------------------------------------------------------------
    def share_prefix(
        self,
        batch_idx: int,
        prefix_tokens: torch.Tensor,
    ) -> int:
        """Attempt to share a prefix cache across requests.

        Checks whether the given ``prefix_tokens`` have been cached
        before.  If so, the existing pages are referenced (ref-count
        incremented) for the new batch entry, avoiding recomputation.

        Args:
            batch_idx: The batch entry that wants to reuse the prefix.
            prefix_tokens: Token ids of the prefix to look up.

        Returns:
            The number of tokens that were served from the shared cache.
        """
        if self.strategy != CacheStrategy.PAGED:
            return 0

        # Hash the prefix tokens.
        prefix_hash = hash(tuple(prefix_tokens.tolist()))

        if prefix_hash in self._prefix_cache:
            shared_page_ids = self._prefix_cache[prefix_hash]
            if batch_idx not in self._page_tables:
                self._page_tables[batch_idx] = [[] for _ in range(self.num_layers)]
            for layer_idx, pid in enumerate(shared_page_ids):
                if pid in self._pages:
                    self._pages[pid].ref_count += 1
                    self._page_tables[batch_idx][layer_idx].append(pid)
            self._logger.debug(
                "Shared %d prefix pages for batch %d.", len(shared_page_ids), batch_idx
            )
            return len(shared_page_ids) * self.page_size

        return 0

    def record_prefix(
        self,
        prefix_tokens: torch.Tensor,
        page_ids: List[int],
    ) -> None:
        """Record a prefix -> pages mapping for future sharing.

        Args:
            prefix_tokens: Token ids of the prefix.
            page_ids: Page ids that store this prefix.
        """
        if self.strategy != CacheStrategy.PAGED:
            return
        prefix_hash = hash(tuple(prefix_tokens.tolist()))
        self._prefix_cache[prefix_hash] = list(page_ids)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_cache_length(self, batch_idx: int = 0) -> int:
        """Return the current cached sequence length for ``batch_idx``."""
        if self.strategy == CacheStrategy.STATIC:
            if 0 <= batch_idx < len(self._static_lengths):
                return self._static_lengths[batch_idx]
            return 0
        else:
            if batch_idx not in self._page_tables:
                return 0
            total = 0
            for layer_pages in self._page_tables[batch_idx]:
                total += len(layer_pages) * self.page_size
            return total

    def memory_usage(self) -> Dict[str, int]:
        """Return a dictionary describing current memory usage.

        Returns:
            A dict with keys ``num_pages``, ``num_active_batches``,
            ``gpu_bytes``, and ``cpu_bytes``.
        """
        if self.strategy == CacheStrategy.STATIC:
            if self._static_k is not None:
                gpu_bytes = self._static_k.nelement() * self._static_k.element_size() * 2
            else:
                gpu_bytes = 0
            return {
                "num_pages": 0,
                "num_active_batches": sum(1 for l in self._static_lengths if l > 0),
                "gpu_bytes": gpu_bytes,
                "cpu_bytes": 0,
            }

        gpu_bytes = 0
        cpu_bytes = 0
        for block in self._pages.values():
            sz = block.key.nelement() * block.key.element_size() * 2
            if block.on_cpu:
                cpu_bytes += sz
            else:
                gpu_bytes += sz
        return {
            "num_pages": len(self._pages),
            "num_active_batches": len(self._page_tables),
            "gpu_bytes": gpu_bytes,
            "cpu_bytes": cpu_bytes,
        }

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all cache state (does not deallocate static tensors)."""
        if self.strategy == CacheStrategy.STATIC:
            self._static_lengths = [0] * self.max_batch_size
        else:
            self._pages.clear()
            self._page_tables.clear()
            self._prefix_cache.clear()
            self._next_page_id = 0
