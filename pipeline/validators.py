"""连接验证器,提取 PipelineBuilder 与 Canvas 共用的验证逻辑。

本模块将原先散落在 :class:`pipeline.composer.PipelineBuilder.connect` 与
:class:`canvas.canvas.Canvas.connect` 中的连接校验逻辑统一为
:class:`ConnectionValidator`,消除重复代码并为后续扩展(如自定义校验规则)
提供单一入口。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

__all__ = ["ConnectionValidator"]

#: 一条连接的四元组表示:``(from_node, to_node, output_key, input_key)``。
EdgeTuple = Tuple[str, str, str, str]


class ConnectionValidator:
    """连接验证器,供 PipelineBuilder 和 Canvas 共用。

    封装以下检查(按顺序执行,遇到第一个错误即返回):

    1. **端点存在性** —— ``from_id`` 和 ``to_id`` 必须已在 ``declared_ids``
       中声明。
    2. **自环检测** —— ``from_id != to_id``。
    3. **重复边检测** —— 同一条 ``(from, to, output_key, input_key)`` 不能
       重复声明。
    4. **端口存在性**(可选,需要 ``specs`` 与 ``node_type_map``)——
       ``output_key`` 是 ``from_id`` 类型的声明输出,``input_key`` 是
       ``to_id`` 类型的声明输入。
    5. **类型兼容性**(可选,需要 ``specs`` 与 ``node_type_map``)—— 输出端口
       类型与输入端口类型兼容(通过 :class:`~nodes.type_system.TypeSystem`
       判定)。
    6. **环检测**(DFS)—— 新边不引入环。

    所有方法均为 ``@staticmethod``,验证器本身无状态,可在多线程下安全调用。
    """

    # ------------------------------------------------------------------
    # Spec 缓存(供 PipelineBuilder / Canvas 共用,避免每次 connect 都遍历注册表)
    # ------------------------------------------------------------------
    _spec_cache: Optional[Dict[str, Any]] = None
    _spec_cache_time: float = 0.0
    _SPEC_CACHE_TTL: float = 5.0  # 秒

    @classmethod
    def load_specs(cls) -> Optional[Dict[str, Any]]:
        """惰性加载节点规格(``{node_type: NodeSpec}``),带 TTL 缓存。

        从 L4 :class:`~nodes.base.NodeRegistry` 加载已注册节点的规格。当注册表
        不可用时返回 ``None``,调用方应据此跳过端口 / 类型校验。结果缓存
        :attr:`_SPEC_CACHE_TTL` 秒,避免每次 ``connect()`` 都遍历注册表。
        """
        now = time.monotonic()
        if (
            cls._spec_cache is not None
            and (now - cls._spec_cache_time) < cls._SPEC_CACHE_TTL
        ):
            return cls._spec_cache
        try:
            from nodes import NodeRegistry  # type: ignore[import-not-found]

            registry = NodeRegistry()
            cls._spec_cache = {spec.type: spec for spec in registry.list()}
            cls._spec_cache_time = now
            return cls._spec_cache
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 核心验证入口
    # ------------------------------------------------------------------
    @staticmethod
    def validate_connection(
        from_id: str,
        to_id: str,
        output_key: str,
        input_key: str,
        declared_ids: Set[str],
        existing_edges: List[EdgeTuple],
        node_type_map: Optional[Dict[str, str]] = None,
        specs: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """验证一条连接,返回错误消息(``None`` 表示通过)。

        Args:
            from_id: 产生输出的(上游)节点 id。
            to_id: 消费输入的(下游)节点 id。
            output_key: 上游节点的输出端口名。
            input_key: 下游节点的输入端口名。
            declared_ids: 已声明节点 id 的集合。
            existing_edges: 现有边的四元组列表
                ``[(from, to, output_key, input_key), ...]``。
            node_type_map: 可选的 ``{node_id: node_type}`` 映射,用于端口 /
                类型校验。
            specs: 可选的 ``{node_type: NodeSpec}`` 映射,用于端口 / 类型校验。

        Returns:
            ``None`` 表示验证通过;否则返回描述失败原因的可读字符串。
        """
        # 1. 端点存在性
        if from_id not in declared_ids:
            return "源节点 {!r} 未声明。".format(from_id)
        if to_id not in declared_ids:
            return "目标节点 {!r} 未声明。".format(to_id)

        # 2. 自环检测
        if from_id == to_id:
            return "连接 {}.{} -> {}.{} 是自环。".format(
                from_id, output_key, to_id, input_key
            )

        # 3. 重复边检测
        new_key: EdgeTuple = (from_id, to_id, output_key, input_key)
        for edge in existing_edges:
            if (edge[0], edge[1], edge[2], edge[3]) == new_key:
                return "重复连接 {}.{} -> {}.{}。".format(
                    from_id, output_key, to_id, input_key
                )

        # 4 & 5. 端口存在性 + 类型兼容性(可选,需要 specs 与 node_type_map)
        if specs is not None and node_type_map is not None:
            from_type = node_type_map.get(from_id)
            to_type = node_type_map.get(to_id)
            from_spec = specs.get(from_type) if from_type else None
            to_spec = specs.get(to_type) if to_type else None

            # 端口存在性
            if from_spec is not None and output_key not in from_spec.outputs:
                available = (
                    ", ".join(sorted(from_spec.outputs.keys())) or "(none)"
                )
                return (
                    "端口 {!r} 不是节点类型 {!r}(节点 {!r})的声明输出。"
                    "可用输出端口: {}。".format(
                        output_key, from_type, from_id, available
                    )
                )
            if to_spec is not None and input_key not in to_spec.inputs:
                available = (
                    ", ".join(sorted(to_spec.inputs.keys())) or "(none)"
                )
                return (
                    "端口 {!r} 不是节点类型 {!r}(节点 {!r})的声明输入。"
                    "可用输入端口: {}。".format(
                        input_key, to_type, to_id, available
                    )
                )

            # 类型兼容性
            if from_spec is not None and to_spec is not None:
                from nodes.type_system import TypeSystem

                out_type = from_spec.outputs[output_key]
                in_type = to_spec.inputs[input_key]
                if not TypeSystem.is_compatible(out_type, in_type):
                    compatible = TypeSystem.compatible_inputs(out_type)
                    return (
                        "类型不匹配: 节点 {!r} 的输出端口 {!r}(类型 {!r})与"
                        "节点 {!r} 的输入端口 {!r}(类型 {!r})不兼容。"
                        "兼容的输入类型: {}。".format(
                            from_id,
                            output_key,
                            out_type,
                            to_id,
                            input_key,
                            in_type,
                            ", ".join(compatible),
                        )
                    )

        # 6. 环检测(DFS)
        if ConnectionValidator.would_create_cycle(
            from_id, to_id, existing_edges
        ):
            return "连接 {}.{} -> {}.{} 会引入环。".format(
                from_id, output_key, to_id, input_key
            )

        return None

    # ------------------------------------------------------------------
    # 环检测
    # ------------------------------------------------------------------
    @staticmethod
    def would_create_cycle(
        from_id: str,
        to_id: str,
        existing_edges: List[EdgeTuple],
    ) -> bool:
        """返回添加 ``from_id -> to_id`` 是否会产生环。

        当 ``from_id`` 已经能从 ``to_id`` 经现有边到达时(即存在有向路径
        ``to_id -> ... -> from_id``),添加新边会闭合一个环。使用迭代式 DFS
        遍历连接图。

        Args:
            from_id: 拟添加边的上游节点。
            to_id: 拟添加边的下游节点。
            existing_edges: 现有边的四元组列表(不含新边)。

        Returns:
            ``True`` 表示新边会闭合环。
        """
        if from_id == to_id:
            return True
        # 构建邻接表(仅现有边)
        adj: Dict[str, List[str]] = {}
        for edge in existing_edges:
            adj.setdefault(edge[0], []).append(edge[1])
        # DFS: 检查从 to_id 出发能否回到 from_id
        visited: Set[str] = set()
        stack: List[str] = [to_id]
        while stack:
            current = stack.pop()
            if current == from_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adj.get(current, []))
        return False
