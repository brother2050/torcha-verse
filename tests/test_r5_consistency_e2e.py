"""generate_via_pipeline 端到端测试。"""
import pytest
from consistency.pipeline import ConsistencyPipeline, ConsistencyProfile
from nodes.base import NodeContext

class TestGenerateViaPipeline:
    """generate_via_pipeline 端到端测试。"""

    def test_generate_via_pipeline_basic(self):
        """基本调用应返回结果。"""
        profile = ConsistencyProfile()
        pipe = ConsistencyPipeline(profile=profile)
        ctx = NodeContext()
        results = pipe.generate_via_pipeline("a cat", width=64, height=64, ctx=ctx)
        assert isinstance(results, dict)
        assert "base" in results

    def test_generate_via_pipeline_default_ctx(self):
        """ctx=None 时自动创建默认上下文。"""
        profile = ConsistencyProfile()
        pipe = ConsistencyPipeline(profile=profile)
        results = pipe.generate_via_pipeline("a dog", width=64, height=64)
        assert isinstance(results, dict)
        assert "base" in results

    def test_to_pipeline_structure(self):
        """to_pipeline 构建的 Pipeline 结构正确。"""
        profile = ConsistencyProfile()
        pipe = ConsistencyPipeline(profile=profile)
        pipeline = pipe.to_pipeline("test prompt", width=64, height=64)
        # 应有 base 节点
        assert "base" in pipeline.dag.node_ids

    def test_generate_via_pipeline_with_steps(self):
        """传递 steps 参数。"""
        profile = ConsistencyProfile()
        pipe = ConsistencyPipeline(profile=profile)
        results = pipe.generate_via_pipeline("a cat", width=64, height=64, steps=5)
        assert "base" in results

    def test_generate_via_pipeline_invalid_dimensions(self):
        """无效维度应抛出 ValueError。"""
        profile = ConsistencyProfile()
        pipe = ConsistencyPipeline(profile=profile)
        with pytest.raises(ValueError):
            pipe.generate_via_pipeline("test", width=0, height=64)

    def test_consistency_pipeline_close(self):
        """close() 不应抛异常。"""
        profile = ConsistencyProfile()
        pipe = ConsistencyPipeline(profile=profile)
        pipe.close()  # 不应抛异常

    def test_consistency_pipeline_context_manager(self):
        """上下文管理器协议。"""
        profile = ConsistencyProfile()
        with ConsistencyPipeline(profile=profile) as pipe:
            assert pipe is not None
        # 退出后不应抛异常
