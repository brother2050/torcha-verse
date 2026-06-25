"""Tests for the v0.6.x ``@config_schema`` + :class:`Field` DSL.

Covers the new :mod:`infrastructure.config_center._schema` module:
``Field.coerce`` validation, ``@config_schema`` registration, and
the automatic seeding of defaults into the :class:`ConfigCenter`
singleton.
"""

from __future__ import annotations

import pytest

from infrastructure.config_center import (
    ConfigCenter,
    ConfigSchema,
    ConfigSchemaError,
    Field,
    config_schema,
    default_registry,
    get_config,
)


class TestField:
    """Tests for the :class:`Field` descriptor and ``coerce`` validator."""

    def test_int_field_accepts_in_range(self):
        f = Field(default=5, type_=int, min=1, max=10)
        assert f.coerce(7) == 7
        assert f.coerce(1) == 1
        assert f.coerce(10) == 10

    def test_int_field_rejects_out_of_range(self):
        f = Field(default=5, type_=int, min=1, max=10)
        with pytest.raises(ConfigSchemaError):
            f.coerce(0)
        with pytest.raises(ConfigSchemaError):
            f.coerce(11)

    def test_str_field_enforces_choices(self):
        f = Field(default="cpu", type_=str, choices=("cpu", "gpu", "mps"))
        assert f.coerce("gpu") == "gpu"
        with pytest.raises(ConfigSchemaError):
            f.coerce("tpu")

    def test_int_coerces_from_string(self):
        f = Field(default=5, type_=int)
        assert f.coerce("42") == 42

    def test_float_coerces_from_string(self):
        f = Field(default=0.0, type_=float)
        assert f.coerce("3.14") == 3.14

    def test_bool_coerces_from_int(self):
        f = Field(default=False, type_=bool)
        assert f.coerce(1) is True
        assert f.coerce(0) is False

    def test_wrong_type_raises(self):
        f = Field(default=5, type_=int)
        with pytest.raises(ConfigSchemaError):
            f.coerce("not a number")

    def test_list_coerces_from_csv(self):
        f = Field(default=["a"], type_=list)
        assert f.coerce("a,b,c") == ["a", "b", "c"]


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the ConfigCenter singleton + schema registry between tests."""
    yield
    ConfigCenter.reset()


class TestConfigSchema:
    """Tests for ``@config_schema`` and the auto-seeding behaviour."""

    def test_schema_registers_with_registry(self):
        @config_schema
        class _SampleFlag:
            """A sample feature flag."""
            enabled: bool = Field(default=True, doc="enable?")
            rate: float = Field(default=0.1, doc="rate", min=0.0, max=1.0)

        schema = default_registry.get("_SampleFlag")
        assert schema is not None
        assert schema.name == "_SampleFlag"
        assert "A sample feature flag" in schema.doc
        field_names = [n for n, _ in schema.fields]
        assert field_names == ["enabled", "rate"]

    def test_schema_seeds_config_center(self):
        @config_schema
        class _MyFlags:
            """Auto-seed test."""
            on: bool = Field(default=True)
            size: int = Field(default=42, min=1)

        assert get_config("_MyFlags.on") is True
        assert get_config("_MyFlags.size") == 42

    def test_schema_does_not_overwrite_existing(self):
        ConfigCenter().set("_AlreadySet.value", 999)
        ConfigCenter().set("_AlreadySet2.x", 7)

        @config_schema
        class _AlreadySet:
            """Don't clobber."""
            value: int = Field(default=0)

        @config_schema
        class _AlreadySet2:
            """Don't clobber."""
            x: int = Field(default=0)

        assert get_config("_AlreadySet.value") == 999
        assert get_config("_AlreadySet2.x") == 7

    def test_describe_returns_serialisable_list(self):
        @config_schema
        class _MetaSample:
            """For describe test."""
            threshold: float = Field(default=0.5, min=0.0, max=1.0)

        descriptions = default_registry.describe()
        names = [d["name"] for d in descriptions]
        assert "_MetaSample" in names
        for d in descriptions:
            assert isinstance(d["doc"], str)
            assert isinstance(d["fields"], list)
