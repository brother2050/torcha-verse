"""Tests for ``infrastructure.error_helper`` (D3 stage three).

Covers the v0.4.x D3 stage three work-stream: ``safe_call`` is now
forensic (always logs a warning + bumps a counter) and the module
exposes :data:`DEGRADE_COUNTERS` plus a :func:`record_degrade` helper
for ``finally`` blocks and sandbox-generated code.

Coverage map:

* :func:`safe_call` returns the value when ``fn`` succeeds.
* :func:`safe_call` returns ``fallback`` when ``fn`` raises the
  expected exception.
* :func:`safe_call` re-raises (does **not** catch) when the raised
  exception is not in ``expected``.
* :func:`safe_call` always emits a ``logger.warning`` on the
  fallback path -- callers cannot silence it.
* :func:`safe_call` increments :data:`DEGRADE_COUNTERS` keyed by
  ``op_id`` (or a derived key when ``op_id`` is omitted).
* :func:`record_degrade` increments :data:`DEGRADE_COUNTERS` and
  logs a warning, with or without an attached exception.
* :data:`DEGRADE_COUNTERS` is a :class:`collections.Counter`; tests
  can read it freely.
* The :mod:`infrastructure.error_helper` module is import-safe and
  does not crash on a fresh process.
"""
from __future__ import annotations

import logging
from collections import Counter

import pytest

from infrastructure.error_helper import (
    DEGRADE_COUNTERS,
    record_degrade,
    safe_call,
)


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.error_helper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_degrade_counters():
    """Reset the global DEGRADE_COUNTERS before each test so that
    each test sees a fresh slate."""
    DEGRADE_COUNTERS.clear()
    yield
    DEGRADE_COUNTERS.clear()


@pytest.fixture
def caplog_error_helper(caplog):
    """Restrict the caplog handler to error_helper logger only."""
    caplog.set_level(logging.WARNING, logger="infrastructure.error_helper")
    return caplog


# ---------------------------------------------------------------------------
# safe_call - happy path
# ---------------------------------------------------------------------------
class TestSafeCallHappyPath:
    """safe_call returns the value of fn when fn succeeds."""

    def test_returns_value(self):
        assert safe_call(lambda: 42) == 42

    def test_passes_args(self):
        def add(a, b):
            return a + b
        assert safe_call(add, 2, 3) == 5

    def test_passes_kwargs(self):
        def greet(name, *, greeting="hi"):
            return f"{greeting}, {name}"
        assert safe_call(greet, "world", greeting="hello") == "hello, world"

    def test_no_warning_on_success(self, caplog_error_helper):
        safe_call(lambda: "ok")
        assert len(caplog_error_helper.records) == 0

    def test_no_counter_increment_on_success(self):
        safe_call(lambda: "ok", op_id="success_op")
        assert DEGRADE_COUNTERS["success_op"] == 0


# ---------------------------------------------------------------------------
# safe_call - fallback path
# ---------------------------------------------------------------------------
class TestSafeCallFallback:
    """safe_call returns fallback and logs/bump on the expected exception."""

    def test_returns_fallback_on_expected_exception(self):
        def boom():
            raise RuntimeError("kaboom")
        result = safe_call(boom, fallback="down", expected=RuntimeError)
        assert result == "down"

    def test_returns_none_fallback_by_default(self):
        def boom():
            raise RuntimeError("kaboom")
        assert safe_call(boom) is None

    def test_emits_warning(self, caplog_error_helper):
        def boom():
            raise RuntimeError("kaboom")
        safe_call(boom, op="my_op")
        warnings = [r for r in caplog_error_helper.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "my_op" in warnings[0].getMessage()
        assert "kaboom" in warnings[0].getMessage()

    def test_emits_warning_with_fallback_label_when_no_op(self, caplog_error_helper):
        def my_func():
            raise RuntimeError("x")
        safe_call(my_func)
        warnings = [r for r in caplog_error_helper.records if r.levelno == logging.WARNING]
        assert "my_func" in warnings[0].getMessage()

    def test_warning_cannot_be_silenced_by_logger_none(self, caplog_error_helper):
        """``logger=None`` (the default sentinel) must use the
        module-level logger -- the warning is never silent."""
        def boom():
            raise RuntimeError("kaboom")
        safe_call(boom, logger=None)
        warnings = [r for r in caplog_error_helper.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1

    def test_counter_increments_under_op_id(self):
        def boom():
            raise RuntimeError("x")
        safe_call(boom, op_id="my_op", expected=RuntimeError)
        assert DEGRADE_COUNTERS["my_op"] == 1

    def test_counter_increments_twice(self):
        def boom():
            raise RuntimeError("x")
        safe_call(boom, op_id="my_op", expected=RuntimeError)
        safe_call(boom, op_id="my_op", expected=RuntimeError)
        assert DEGRADE_COUNTERS["my_op"] == 2

    def test_counter_uses_derived_key_when_op_id_omitted(self):
        def my_func():
            raise RuntimeError("x")
        safe_call(my_func, op="loading", expected=RuntimeError)
        # derived key format: "<op>::<fn.__name__>"
        assert DEGRADE_COUNTERS["loading::my_func"] == 1

    def test_narrowed_expected_only_catches_listed(self):
        """``expected=ValueError`` must NOT catch ``RuntimeError``."""
        def boom():
            raise RuntimeError("kaboom")
        with pytest.raises(RuntimeError):
            safe_call(boom, fallback="down", expected=ValueError)
        # And no counter bump -- the exception was re-raised.
        assert all(v == 0 for v in DEGRADE_COUNTERS.values())

    def test_expected_tuple_catches_any(self):
        def boom():
            raise KeyError("k")
        result = safe_call(boom, fallback="down",
                           expected=(ValueError, KeyError, TypeError))
        assert result == "down"
        assert "safe_call::boom" in DEGRADE_COUNTERS or any(
            k.endswith("::boom") for k in DEGRADE_COUNTERS
        )

    def test_unexpected_exception_propagates(self):
        """A non-expected exception must propagate AND must not bump
        the counter."""
        def boom():
            raise ValueError("not handled")
        with pytest.raises(ValueError):
            safe_call(boom, expected=KeyError, op_id="propagate")
        assert DEGRADE_COUNTERS["propagate"] == 0

    def test_custom_logger(self, caplog):
        caplog.set_level(logging.WARNING)
        custom = logging.getLogger("test.custom.degrade")
        def boom():
            raise RuntimeError("x")
        safe_call(boom, op="x", logger=custom)
        # The custom logger must have received the record.
        assert any(r.name == "test.custom.degrade" for r in caplog.records)


# ---------------------------------------------------------------------------
# record_degrade
# ---------------------------------------------------------------------------
class TestRecordDegrade:
    """record_degrade bumps the counter and logs a warning."""

    def test_bumps_counter(self):
        record_degrade("my_op")
        assert DEGRADE_COUNTERS["my_op"] == 1

    def test_logs_warning(self, caplog_error_helper):
        record_degrade("my_op", op="my label")
        warnings = [r for r in caplog_error_helper.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "my label" in warnings[0].getMessage()

    def test_logs_with_exception(self, caplog_error_helper):
        record_degrade("my_op", op="my label", exc=RuntimeError("nope"))
        warnings = [r for r in caplog_error_helper.records if r.levelno == logging.WARNING]
        assert "nope" in warnings[0].getMessage()

    def test_logs_without_exception(self, caplog_error_helper):
        record_degrade("my_op")
        warnings = [r for r in caplog_error_helper.records if r.levelno == logging.WARNING]
        assert "no exception attached" in warnings[0].getMessage()

    def test_default_label_is_op_id(self, caplog_error_helper):
        record_degrade("just_the_id")
        warnings = [r for r in caplog_error_helper.records if r.levelno == logging.WARNING]
        assert "just_the_id" in warnings[0].getMessage()

    def test_custom_logger(self, caplog):
        caplog.set_level(logging.WARNING)
        custom = logging.getLogger("test.custom.record")
        record_degrade("op_x", logger=custom)
        assert any(r.name == "test.custom.record" for r in caplog.records)


# ---------------------------------------------------------------------------
# DEGRADE_COUNTERS shape
# ---------------------------------------------------------------------------
class TestDegradeCounters:
    """DEGRADE_COUNTERS is a Counter; multiple ops are independent."""

    def test_is_counter(self):
        assert isinstance(DEGRADE_COUNTERS, Counter)

    def test_independent_keys(self):
        record_degrade("op_a")
        record_degrade("op_a")
        record_degrade("op_b")
        assert DEGRADE_COUNTERS["op_a"] == 2
        assert DEGRADE_COUNTERS["op_b"] == 1

    def test_total(self):
        record_degrade("op_a")
        record_degrade("op_b")
        record_degrade("op_b")
        assert sum(DEGRADE_COUNTERS.values()) == 3

    def test_clear_for_tests(self):
        record_degrade("op_x")
        assert DEGRADE_COUNTERS["op_x"] == 1
        DEGRADE_COUNTERS.clear()
        assert "op_x" not in DEGRADE_COUNTERS


# ---------------------------------------------------------------------------
# import safety
# ---------------------------------------------------------------------------
class TestImportSafety:
    """The module is importable and has the documented public surface."""

    def test_module_imports(self):
        import infrastructure.error_helper
        assert hasattr(infrastructure.error_helper, "safe_call")
        assert hasattr(infrastructure.error_helper, "record_degrade")
        assert hasattr(infrastructure.error_helper, "DEGRADE_COUNTERS")

    def test_dunder_all(self):
        import infrastructure.error_helper
        # __all__ must list the three public names.
        assert "safe_call" in infrastructure.error_helper.__all__
        assert "record_degrade" in infrastructure.error_helper.__all__
        assert "DEGRADE_COUNTERS" in infrastructure.error_helper.__all__
