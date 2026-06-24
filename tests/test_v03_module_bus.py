"""Tests for v0.3.0 ModuleBus singleton registry.

Covers register / resolve / list / invalidate / has / get_spec, the
``@register_module`` decorator, factory caching, ``ModuleNotFoundError``,
``reset()`` and basic thread safety.
"""
from __future__ import annotations

import threading

import pytest

from core.module_bus import (
    ModuleBus,
    ModuleNotFoundError,
    ModuleSpec,
    register_module,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_bus():
    """Ensure a clean ModuleBus singleton before and after each test."""
    ModuleBus.reset()
    yield
    ModuleBus.reset()


# ---------------------------------------------------------------------------
# Singleton & reset
# ---------------------------------------------------------------------------
class TestModuleBusSingleton:
    """Singleton identity and reset behaviour."""

    def test_singleton_identity(self):
        """ModuleBus() always returns the same instance."""
        b1 = ModuleBus()
        b2 = ModuleBus()
        assert b1 is b2

    def test_reset_creates_fresh_instance(self):
        """reset() drops the singleton so the next call is fresh."""
        b1 = ModuleBus()
        b1.register("test.ns", "item", lambda: 42)
        assert b1.count() == 1

        ModuleBus.reset()
        b2 = ModuleBus()
        assert b2 is not b1
        assert b2.count() == 0


# ---------------------------------------------------------------------------
# Register / resolve / has / get_spec
# ---------------------------------------------------------------------------
class TestModuleBusRegisterResolve:
    """Core register / resolve / has / get_spec surface."""

    def test_register_and_resolve(self):
        """A registered factory is invoked and its result cached."""
        bus = ModuleBus()
        bus.register("model.text", "llama", lambda: {"weights": 42})
        instance = bus.resolve("model.text", "llama")
        assert instance == {"weights": 42}

    def test_factory_cached(self):
        """resolve() invokes the factory at most once."""
        bus = ModuleBus()
        call_count = [0]

        def factory():
            call_count[0] += 1
            return object()

        bus.register("svc", "counter", factory)
        obj1 = bus.resolve("svc", "counter")
        obj2 = bus.resolve("svc", "counter")
        assert obj1 is obj2
        assert call_count[0] == 1

    def test_has(self):
        """has() returns True for registered, False for unknown."""
        bus = ModuleBus()
        bus.register("model.text", "llama", lambda: {})
        assert bus.has("model.text", "llama") is True
        assert bus.has("model.text", "nonexistent") is False

    def test_get_spec(self):
        """get_spec() returns the ModuleSpec for a registered module."""
        bus = ModuleBus()
        bus.register(
            "model.text", "llama", lambda: {},
            version="1.0.0", description="A text model",
            tags=["llm"],
        )
        spec = bus.get_spec("model.text", "llama")
        assert isinstance(spec, ModuleSpec)
        assert spec.kind == "model.text"
        assert spec.name == "llama"
        assert spec.version == "1.0.0"
        assert spec.description == "A text model"
        assert "llm" in spec.tags

    def test_get_spec_raises_for_unknown(self):
        """get_spec() raises ModuleNotFoundError for unknown modules."""
        bus = ModuleBus()
        with pytest.raises(ModuleNotFoundError):
            bus.get_spec("unknown.kind", "unknown.name")

    def test_resolve_raises_module_not_found(self):
        """resolve() raises ModuleNotFoundError for unregistered modules."""
        bus = ModuleBus()
        with pytest.raises(ModuleNotFoundError):
            bus.resolve("missing.kind", "missing.name")


# ---------------------------------------------------------------------------
# List & invalidate
# ---------------------------------------------------------------------------
class TestModuleBusListInvalidate:
    """list() filtering and cache invalidation."""

    def test_list_all(self):
        """list() with no kind returns every registered spec."""
        bus = ModuleBus()
        bus.register("model.text", "a", lambda: {})
        bus.register("model.image", "b", lambda: {})
        specs = bus.list()
        assert len(specs) == 2

    def test_list_by_kind_prefix(self):
        """list(kind) filters by exact kind or namespace prefix."""
        bus = ModuleBus()
        bus.register("model.text", "a", lambda: {})
        bus.register("model.image", "b", lambda: {})
        bus.register("node", "c", lambda: {})

        model_specs = bus.list("model")
        assert len(model_specs) == 2
        assert all(s.kind.startswith("model") for s in model_specs)

    def test_invalidate_re_invokes_factory(self):
        """invalidate() drops the cache so the factory runs again."""
        bus = ModuleBus()
        call_count = [0]

        def factory():
            call_count[0] += 1
            return {"n": call_count[0]}

        bus.register("svc", "fresh", factory)
        obj1 = bus.resolve("svc", "fresh")
        assert call_count[0] == 1

        bus.invalidate("svc", "fresh")
        obj2 = bus.resolve("svc", "fresh")
        assert call_count[0] == 2
        assert obj2["n"] == 2

    def test_invalidate_all(self):
        """invalidate_all() clears every cached instance."""
        bus = ModuleBus()
        calls = [0]

        def factory():
            calls[0] += 1
            return calls[0]

        bus.register("a", "x", factory)
        bus.register("b", "y", factory)
        bus.resolve("a", "x")
        bus.resolve("b", "y")
        assert calls[0] == 2

        bus.invalidate_all()
        bus.resolve("a", "x")
        bus.resolve("b", "y")
        assert calls[0] == 4


# ---------------------------------------------------------------------------
# @register_module decorator
# ---------------------------------------------------------------------------
class TestRegisterModuleDecorator:
    """The @register_module decorator registers with the global bus."""

    def test_decorator_registers_factory(self):
        """A decorated function is registered and resolvable."""
        @register_module("test.decorator", "widget")
        def make_widget():
            return {"type": "widget"}

        bus = ModuleBus()
        assert bus.has("test.decorator", "widget")
        instance = bus.resolve("test.decorator", "widget")
        assert instance == {"type": "widget"}

    def test_decorator_returns_callable_unchanged(self):
        """The decorator returns the original callable unchanged."""
        @register_module("test.decorator", "gadget")
        def make_gadget():
            return "gadget"

        assert make_gadget() == "gadget"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------
class TestModuleBusThreadSafety:
    """Concurrent resolve() calls are safe and cache correctly."""

    def test_concurrent_resolve_single_instance(self):
        """Multiple threads resolving the same key get the same instance."""
        bus = ModuleBus()
        call_count = [0]
        barrier = threading.Barrier(8)

        def slow_factory():
            call_count[0] += 1
            return {"id": id(object())}

        bus.register("threaded", "shared", slow_factory)

        results = [None] * 8

        def worker(idx):
            barrier.wait()
            results[idx] = bus.resolve("threaded", "shared")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads got the same cached instance.
        assert all(r is results[0] for r in results)
        # Factory was invoked exactly once.
        assert call_count[0] == 1
