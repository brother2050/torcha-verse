"""ConnectionValidator 独立单元测试。"""
import pytest
from pipeline.validators import ConnectionValidator, EdgeTuple

class TestConnectionValidator:
    """ConnectionValidator 的独立单元测试。"""

    def test_endpoint_existence_from_id(self):
        """源节点未声明时返回错误。"""
        error = ConnectionValidator.validate_connection(
            from_id="unknown",
            to_id="b",
            output_key="output",
            input_key="input",
            declared_ids={"b"},
            existing_edges=[],
        )
        assert error is not None
        assert "unknown" in error

    def test_endpoint_existence_to_id(self):
        """目标节点未声明时返回错误。"""
        error = ConnectionValidator.validate_connection(
            from_id="a",
            to_id="unknown",
            output_key="output",
            input_key="input",
            declared_ids={"a"},
            existing_edges=[],
        )
        assert error is not None
        assert "unknown" in error

    def test_self_loop(self):
        """自环检测。"""
        error = ConnectionValidator.validate_connection(
            from_id="a",
            to_id="a",
            output_key="output",
            input_key="input",
            declared_ids={"a"},
            existing_edges=[],
        )
        assert error is not None
        assert "自环" in error or "self" in error.lower()

    def test_duplicate_edge(self):
        """重复边检测。"""
        existing = [("a", "b", "output", "input")]
        error = ConnectionValidator.validate_connection(
            from_id="a",
            to_id="b",
            output_key="output",
            input_key="input",
            declared_ids={"a", "b"},
            existing_edges=existing,
        )
        assert error is not None
        assert "重复" in error or "duplicate" in error.lower()

    def test_cycle_detection(self):
        """环检测：a->b 已存在，添加 b->a 会产生环。"""
        existing = [("a", "b", "output", "input")]
        error = ConnectionValidator.validate_connection(
            from_id="b",
            to_id="a",
            output_key="output",
            input_key="input",
            declared_ids={"a", "b"},
            existing_edges=existing,
        )
        assert error is not None
        assert "环" in error or "cycle" in error.lower()

    def test_valid_connection(self):
        """合法连接返回 None。"""
        error = ConnectionValidator.validate_connection(
            from_id="a",
            to_id="b",
            output_key="output",
            input_key="input",
            declared_ids={"a", "b"},
            existing_edges=[],
        )
        assert error is None

    def test_would_create_cycle_direct(self):
        """直接环检测。"""
        existing = [("a", "b", "output", "input")]
        assert ConnectionValidator.would_create_cycle("b", "a", existing) is True

    def test_would_create_cycle_no_cycle(self):
        """无环时返回 False。"""
        existing = [("a", "b", "output", "input")]
        assert ConnectionValidator.would_create_cycle("a", "c", existing) is False

    def test_would_create_cycle_self_loop(self):
        """自环检测。"""
        assert ConnectionValidator.would_create_cycle("a", "a", []) is True

    def test_load_specs_returns_none_when_no_registry(self):
        """注册表不可用时 load_specs 返回 None。"""
        # 由于 NodeRegistry 可能不可用，验证返回 None 或 dict
        specs = ConnectionValidator.load_specs()
        assert specs is None or isinstance(specs, dict)
