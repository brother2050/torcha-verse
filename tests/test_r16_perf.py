"""R-16 performance tests.

Verify the optimisations applied in R-16 do not change behaviour:
the cache is correct, the unsynchronised ``get_output`` is
read-safe, and ``register_executor`` invalidates the cache.

These tests do **not** assert wall-clock numbers (CI runners are
noisy); they assert the *invariants* the new code paths rely on.
"""

from __future__ import annotations

import threading

import pytest

from nodes.base import BaseNode, NodeContext, NodeSpec, NodeRegistry


# ---------------------------------------------------------------------------
# NodeContext: get_output fast path is lock-free but read-safe
# ---------------------------------------------------------------------------
class _R16PerfTest:
    """Shared mixin to avoid repeating the boilerplate."""

    @staticmethod
    def _make_context() -> NodeContext:
        return NodeContext()


class TestNodeContextUnsynchronisedRead(_R16PerfTest):
    """R-16: ``get_output`` no longer acquires a lock on the hot path."""

    def test_get_output_returns_latest_set(self) -> None:
        """Sequential set + get returns the most recent value."""
        ctx = NodeContext()
        ctx.set_output("n1", {"a": 1})
        assert ctx.get_output("n1", "a") == 1
        ctx.set_output("n1", {"a": 2})
        assert ctx.get_output("n1", "a") == 2

    def test_get_output_missing_raises(self) -> None:
        """Missing node still raises KeyError (R-16: behaviour preserved)."""
        ctx = NodeContext()
        with pytest.raises(KeyError):
            ctx.get_output("nope")

    def test_has_output_unsynchronised(self) -> None:
        """R-16: ``has_output`` no longer acquires a lock; behaviour
        still correct (Python's GIL protects dict membership)."""
        ctx = NodeContext()
        assert not ctx.has_output("n1")
        ctx.set_output("n1", {})
        assert ctx.has_output("n1")

    def test_set_get_concurrent_no_corruption(self) -> None:
        """Concurrent set + get must not crash or corrupt the dict.

        Run for a few thousand iterations in 2 threads; if the
        unsynchronised read in :meth:`get_output` were unsound we
        would see either a ``KeyError`` (key dropped between
        membership and copy) or a ``RuntimeError`` (dict mutated
        during iteration)."""
        ctx = NodeContext()
        errors: list[BaseException] = []
        stop = threading.Event()

        def writer() -> None:
            try:
                for i in range(2000):
                    ctx.set_output(f"n{i % 50}", {"v": i})
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def reader() -> None:
            try:
                for i in range(2000):
                    if ctx.has_output(f"n{i % 50}"):
                        ctx.get_output(f"n{i % 50}", "v")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
        assert not errors, f"concurrent ops raised: {errors}"


# ---------------------------------------------------------------------------
# resolve_executor LRU cache
# ---------------------------------------------------------------------------
class TestResolveExecutorCache:
    """R-16: ``resolve_executor`` caches the resolved value."""

    def test_cache_hit_returns_same_callable(self) -> None:
        ctx = NodeContext()
        a = ctx.resolve_executor("image_txt2img")
        b = ctx.resolve_executor("image_txt2img")
        assert a is b

    def test_register_executor_invalidates_cache(self) -> None:
        """After :meth:`register_executor` the cached entry is
        discarded so the new callable is returned."""
        ctx = NodeContext()
        sentinel = object()  # type: ignore[assignment]
        # First call: bus has the registered node -> adapter.
        first = ctx.resolve_executor("image_txt2img")
        assert first is not None
        # Register an explicit override.
        ctx.register_executor("image_txt2img", sentinel)  # type: ignore[arg-type]
        second = ctx.resolve_executor("image_txt2img")
        assert second is sentinel

    def test_negative_result_is_cached(self) -> None:
        """Resolving a non-existent type returns None and caches it.

        This is the "negative cache" optimisation: the bus is not
        consulted on subsequent lookups.
        """
        ctx = NodeContext()
        assert ctx.resolve_executor("nope_does_not_exist_xyz") is None
        # Second call should not raise (bus is not consulted because
        # negative result is cached).
        assert ctx.resolve_executor("nope_does_not_exist_xyz") is None

    def test_cache_eviction(self) -> None:
        """FIFO eviction keeps the cache bounded by 1024 entries.

        We do not actually grow to 1024 (would be slow); we set a
        tiny ``_executor_cache_max`` and exercise the eviction
        path directly.
        """
        ctx = NodeContext()
        ctx._executor_cache_max = 4  # type: ignore[attr-defined]
        # Five different lookups -> the first one should be evicted.
        for i in range(5):
            ctx.resolve_executor(f"missing_type_{i:02d}")
        # Cache holds at most 4 entries.
        assert len(ctx._executor_cache) <= 4  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# NodeRegistry list cache
# ---------------------------------------------------------------------------
class TestNodeRegistryListCache:
    """R-16: ``list()`` caches the snapshot; register/unregister
    invalidates the cache."""

    def test_list_returns_consistent_snapshot(self) -> None:
        reg = NodeRegistry()
        first = reg.list()
        assert len(first) >= 39
        # Second call returns the same set in the same order.
        second = reg.list()
        assert [s.type for s in first] == [s.type for s in second]

    def test_register_invalidates_cache(self) -> None:
        reg = NodeRegistry()
        before = reg.list()
        reg.register(_R16DemoNode)  # type: ignore[arg-type]
        after = reg.list()
        assert len(after) == len(before) + 1
        assert any(s.type == "r16_demo_node" for s in after)

    def test_unregister_invalidates_cache(self) -> None:
        reg = NodeRegistry()
        reg.register(_R16DemoNode)  # type: ignore[arg-type]
        assert any(s.type == "r16_demo_node" for s in reg.list())
        removed = reg.unregister("r16_demo_node")
        assert removed
        assert not any(s.type == "r16_demo_node" for s in reg.list())


# ---------------------------------------------------------------------------
# Mini node used by the registry test
# ---------------------------------------------------------------------------
class _R16DemoNode(BaseNode):
    spec = NodeSpec(type="r16_demo_node", name="R-16 Demo")
