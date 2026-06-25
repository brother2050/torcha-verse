"""L4 capability-layer node base classes and registry for TorchaVerse v0.3.0.

This module is the foundation of the node system that replaces the v0.1.0
"god-class" engines (``text_engine.py`` 999 lines, ``image_engine.py`` 915
lines, ...) with small, single-responsibility, composable nodes.

A *node* is the smallest unit of generative capability: it accepts a set of
typed inputs, executes one well-defined operation against a
:class:`NodeContext`, and returns a set of typed outputs.  Nodes are wired
together by the L5 pipeline layer; the contract defined here is
what makes that composition possible.

Public surface
--------------

* :class:`NodeSpec` -- declarative description of a node (type, name,
  inputs, outputs, tags).  Attached to every node class as the ``spec``
  class attribute.
* :class:`NodeContext` -- the runtime context handed to every node
  ``execute`` call (module bus, asset store, resource budget, logger,
  audit logger, run config, run id).
* :class:`BaseNode` -- the abstract base class every node derives from.
  Provides real implementations of :meth:`BaseNode.validate_inputs` and
  :meth:`BaseNode.estimate_resources` plus an abstract
  :meth:`BaseNode.execute`.
* :class:`NodeRegistry` -- a thin facade over :class:`ModuleBus` that
  discovers, instantiates and searches nodes by type.
* :func:`register_node` -- class decorator that registers a node with the
  global :class:`ModuleBus` so it is discoverable by every
  :class:`NodeRegistry`.

Design notes
------------
The registry deliberately delegates storage to :class:`ModuleBus`
(the v0.3.0 single assembly point) so that nodes, models, tokenizers and
tools all live in one namespace-aware registry.  A small module-level
index (``_NODE_CLASSES``) is kept only so that :class:`NodeSpec` objects
can be retrieved without instantiating the node.

This module has **no third-party dependencies** -- it only imports from
the dependency-free L1/L2/L3 layers (``ModuleBus``, ``AssetStore``,
``ResourceBudget``, ``AuditLogger``), so it is importable in any
environment, including minimal CI sandboxes.
"""

from __future__ import annotations

import abc
import logging
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Callable, ClassVar, Dict, List, Optional
from uuid import uuid4

from core.module_bus import ModuleBus, ModuleNotFoundError as _BusNotFoundError
from assets.base import AssetRef
from assets.store import AssetStore
from infrastructure.audit_log import AuditLogger
from infrastructure.logger import get_logger
from infrastructure.resource_budget import ResourceBudget

from .type_system import is_optional

__all__ = [
    "NodeSpec",
    "NodeContext",
    "NodeExecutor",
    "BaseNode",
    "NodeRegistry",
    "register_node",
]

#: ModuleBus ``kind`` namespace under which every node is registered.
_NODE_KIND: str = "node"

#: Module-level index of ``node_type -> node class``.  Populated by
#: :func:`register_node` and :meth:`NodeRegistry.register`; used so that
#: :class:`NodeSpec` objects can be returned by :meth:`NodeRegistry.list`
#: without instantiating the node.  The :class:`ModuleBus` remains the
#: authoritative discovery surface.
_NODE_CLASSES: Dict[str, type[BaseNode]] = {}

#: Re-entrant lock guarding the module-level ``_NODE_CLASSES`` index so
#: that concurrent registration / unregistration / lookup is safe.
_NODE_CLASSES_LOCK: threading.RLock = threading.RLock()

#: Module-level logger for the node system (stdlib only -- no torch).
_logger: logging.Logger = get_logger("nodes")


# ---------------------------------------------------------------------------
# NodeSpec
# ---------------------------------------------------------------------------
@dataclass
class NodeSpec:
    """Declarative description of a node.

    A :class:`NodeSpec` is attached to every :class:`BaseNode` subclass
    as the ``spec`` class attribute.  It is the single source of truth for
    a node's identity, its typed input/output contract and its tags, and
    is what the pipeline layer (L5) and the web canvas use to validate
    and render a node.

    Attributes:
        type: Stable, unique node type identifier, e.g.
            ``"image_txt2img"`` or ``"text_chat"``.  Used as the
            :class:`ModuleBus` name under the ``"node"`` kind.
        name: Human-readable display name.
        description: One-line description of what the node does.
        inputs: Mapping of input name to its declared port type string
            (e.g. ``"TEXT"``, ``"IMAGE"``, ``"Optional[SEED]"``).
            Optional inputs are expressed with the ``Optional[T]``
            wrapper.
        outputs: Mapping of output name to its declared port type string.
        tags: Free-form tags used for discovery / filtering.
    """

    type: str
    name: str
    description: str = ""
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the spec fields after dataclass initialisation."""
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError("NodeSpec.type must be a non-empty string.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("NodeSpec.name must be a non-empty string.")
        if not isinstance(self.description, str):
            raise ValueError("NodeSpec.description must be a string.")
        if not isinstance(self.inputs, dict):
            raise ValueError("NodeSpec.inputs must be a dict[str, str].")
        if not isinstance(self.outputs, dict):
            raise ValueError("NodeSpec.outputs must be a dict[str, str].")
        if not isinstance(self.tags, list):
            raise ValueError("NodeSpec.tags must be a list[str].")

    def __repr__(self) -> str:
        return (
            "NodeSpec(type={!r}, name={!r}, "
            "inputs={}, outputs={}, tags={!r})".format(
                self.type,
                self.name,
                list(self.inputs.keys()),
                list(self.outputs.keys()),
                self.tags,
            )
        )


# ---------------------------------------------------------------------------
# NodeContext
# ---------------------------------------------------------------------------
#: :meth:`NodeContext.get_output` 在键不存在时返回的哨兵对象,
#: 用于区分"存储了 ``None``"与"该条目不存在"两种情况。
_MISSING: Any = object()

#: L5 管道层默认的最大并发工作线程数。
_DEFAULT_MAX_WORKERS: int = 4

#: 节点执行器可调用对象的类型别名:``(inputs, ctx) -> outputs``。
#: 定义在 :class:`NodeContext` 之前,以便 ``NodeContext`` 的字段注解
#: (``executors: Dict[str, NodeExecutor]``)能直接引用该别名。
NodeExecutor = Callable[[Dict[str, Any], "NodeContext"], Dict[str, Any]]


@dataclass
class NodeContext:
    """运行期上下文,同时承载 L4 节点执行与 L5 管道编排所需的服务。

    本类是 v0.3.0 架构中 **唯一** 的 ``NodeContext``:它合并了原先
    ``nodes/base.py`` 的 L4 节点上下文与 ``pipeline/composer.py`` 的 L5
    管道上下文,消除两个同名类带来的歧义。

    **L4 节点上下文职责** —— 传递给每个节点 ``execute`` 调用的横切服务:
    :class:`ModuleBus`(解析模型 / 分词器 / 同伴节点)、:class:`AssetStore`
    (读写版本化资产)、:class:`ResourceBudget`(硬性资源上限)、日志器、
    :class:`AuditLogger`、运行配置字典与唯一运行 id。

    **L5 管道上下文职责** —— 在同一次管道运行中跨节点共享:

    1. *输出存储* —— 线程安全的 ``node_id -> outputs`` 映射,节点完成后写入,
       下游节点据此读取上游结果。
    2. *执行器解析* —— 为给定 ``node_type`` 查找可调用对象的查找链。显式
       注册的执行器优先,其次 :class:`ModuleBus`(``"node"`` kind),最后
       返回 ``None``(passthrough)。
    3. *元数据* —— 任意可变的键值袋,作为 ``config`` 的别名保留 L5 历史调用
       习惯。

    所有字段都有合理默认值,因此可以无参构造(便于测试与 dry-run)。

    Attributes:
        bus: 用于解析依赖的模块装配总线。
        assets: 分层资产存储(dry-run 时可为 ``None``)。
        budget: 本次运行的硬性资源预算。
        logger: 节点诊断用的日志器。
        audit: 安全 / 运维事件的审计日志器。
        config: 自由格式的运行配置字典。节点从此读取默认值
            (如 ``"default_text_model"``)。同时也是 L5 ``metadata`` 的别名。
        run_id: 当前运行的唯一标识。
        executors: ``node_type -> 可调用对象`` 的显式映射。每个可调用对象
            接收 ``(inputs, ctx)`` 并返回输出字典。
        max_workers: 并行执行的默认工作线程上限。
        strict_mode: 为 ``True`` 时,缺失执行器会抛异常而非 passthrough。
    """

    # --- L4 节点上下文字段 ---
    bus: ModuleBus = field(default_factory=ModuleBus)
    assets: Optional[AssetStore] = None
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    logger: logging.Logger = field(
        default_factory=lambda: get_logger("nodes.context")
    )
    audit: Optional[AuditLogger] = None
    config: Dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid4().hex)
    # --- L5 管道层字段 ---
    executors: Dict[str, "NodeExecutor"] = field(default_factory=dict)
    max_workers: int = _DEFAULT_MAX_WORKERS
    strict_mode: bool = False

    def __post_init__(self) -> None:
        """初始化 L5 管道内部状态(输出存储与线程锁)并规范化字段。"""
        # 输出存储与保护它的可重入锁。
        self._lock: threading.RLock = threading.RLock()
        self._outputs: Dict[str, Dict[str, Any]] = {}
        # 规范化 max_workers,保证至少为 1。
        self.max_workers = max(1, int(self.max_workers))

    # ------------------------------------------------------------------
    # L5 兼容属性
    # ------------------------------------------------------------------
    @property
    def metadata(self) -> Dict[str, Any]:
        """``config`` 的别名,保留 L5 管道层的历史调用习惯。"""
        return self.config

    # ------------------------------------------------------------------
    # 输出存储
    # ------------------------------------------------------------------
    def set_output(self, node_id: str, outputs: Dict[str, Any]) -> None:
        """记录节点 ``node_id`` 产生的输出。

        Args:
            node_id: 产生输出的节点 id。
            outputs: 待存储的输出字典。
        """
        with self._lock:
            self._outputs[node_id] = dict(outputs)

    def get_output(
        self, node_id: str, key: Optional[str] = None
    ) -> Any:
        """获取节点 ``node_id`` 的输出。

        Args:
            node_id: 产生输出的节点 id。
            key: 给定时返回该具体输出键;为 ``None`` 时返回节点完整输出字典。

        Returns:
            请求的值(或输出字典)。缺失的键返回 ``None``;缺失的节点抛
            :class:`KeyError`。

        Raises:
            KeyError: 若 ``node_id`` 没有记录任何输出。
        """
        with self._lock:
            outputs = self._outputs.get(node_id, _MISSING)
        if outputs is _MISSING:
            raise KeyError(
                "No outputs recorded for node {!r}.".format(node_id)
            )
        if key is None:
            return dict(outputs)
        return outputs.get(key)

    def has_output(self, node_id: str) -> bool:
        """返回是否已为 ``node_id`` 记录输出。"""
        with self._lock:
            return node_id in self._outputs

    def all_outputs(self) -> Dict[str, Dict[str, Any]]:
        """返回所有已记录节点输出的浅拷贝。"""
        with self._lock:
            return {nid: dict(out) for nid, out in self._outputs.items()}

    def reset_outputs(self) -> None:
        """清空所有已记录的输出,重置运行状态。"""
        with self._lock:
            self._outputs.clear()

    # ------------------------------------------------------------------
    # 执行器解析
    # ------------------------------------------------------------------
    def register_executor(
        self, node_type: str, executor: "NodeExecutor"
    ) -> None:
        """为节点类型注册一个显式执行器。

        Args:
            node_type: 执行器所处理的节点类型。
            executor: ``(inputs, ctx) -> outputs`` 的可调用对象。
        """
        with self._lock:
            self.executors[node_type] = executor

    def resolve_executor(
        self, node_type: str
    ) -> Optional["NodeExecutor"]:
        """解析 ``node_type`` 对应的执行器可调用对象。

        查找顺序:

        1. 显式 ``executors`` 映射。
        2. :class:`ModuleBus` 的 ``"node"`` kind(配置了 bus 时)。当解析
           结果是一个 :class:`BaseNode` 实例(即具有 ``execute`` 方法)时,
           包装为适配器闭包,将管道调用签名 ``(inputs, ctx)`` 转换为节点
           签名 ``execute(ctx, **inputs)``。

        Returns:
            执行器可调用对象;未注册时返回 ``None``(管道将回退到 passthrough)。
        """
        # 1. 先查 executors dict。
        with self._lock:
            executor = self.executors.get(node_type)
        if executor is not None:
            return executor

        # 2. 再查 ModuleBus 的 "node" kind。
        if self.bus is not None:
            try:
                if self.bus.has(_NODE_KIND, node_type):
                    resolved = self.bus.resolve(_NODE_KIND, node_type)
                    # 若解析结果是一个 BaseNode 实例(具有 execute 方法),
                    # 包装为适配器闭包,将 (inputs, ctx) 转为
                    # node.execute(ctx, **inputs)。
                    if hasattr(resolved, "execute") and callable(
                        getattr(resolved, "execute")
                    ):
                        def _node_adapter(
                            inputs: Dict[str, Any],
                            ctx: "NodeContext",
                        ) -> Dict[str, Any]:
                            # 优先使用 _safe_execute(S2-4),获得统一的异常
                            # 处理与日志记录;回退到 execute 以兼容非 BaseNode。
                            if hasattr(resolved, "_safe_execute"):
                                return resolved._safe_execute(ctx, **inputs)
                            return resolved.execute(ctx, **inputs)
                        return _node_adapter
                    return resolved
            except Exception:  # pragma: no cover - 防御性处理
                _logger.debug(
                    "ModuleBus 查找 %s 失败", node_type, exc_info=True
                )
        return None

    def __repr__(self) -> str:
        return (
            "NodeContext(run_id={!r}, bus={!r}, assets={!r}, "
            "budget={!r}, outputs={}, executors={})".format(
                self.run_id,
                self.bus,
                "set" if self.assets is not None else "None",
                self.budget,
                len(self._outputs),
                len(self.executors),
            )
        )


# ---------------------------------------------------------------------------
# BaseNode
# ---------------------------------------------------------------------------
class BaseNode(abc.ABC):
    """Abstract base class for every TorchaVerse capability node.

    A node is the smallest unit of generative capability.  Subclasses
    declare their contract through the ``spec`` class attribute (a
    :class:`NodeSpec`) and implement :meth:`execute`.  The base class
    provides real, reusable implementations of :meth:`validate_inputs`
    and :meth:`estimate_resources` that operate on ``spec.inputs``;
    subclasses typically extend them with domain-specific checks.

    Class attributes:
        spec: The :class:`NodeSpec` describing this node.  Subclasses
            *must* assign a :class:`NodeSpec` instance.
    """

    #: Declarative node contract.  Subclasses assign a :class:`NodeSpec`.
    spec: ClassVar[NodeSpec]

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def execute(self, ctx: NodeContext, **inputs: Any) -> Dict[str, Any]:
        """Run the node and return its outputs.

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: Keyword inputs matching ``spec.inputs``.

        Returns:
            A dictionary mapping output names (per ``spec.outputs``) to
            their produced values.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 安全执行包装 (S2-4)
    # ------------------------------------------------------------------
    def _safe_execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """安全执行包装:捕获常见运行时异常并记录日志后重新抛出。

        本方法包裹 :meth:`execute`,捕获 :class:`OSError`、
        :class:`RuntimeError` 与 :class:`MemoryError`,在 ctx 的日志器上
        记录 error 级别日志后重新抛出,以便上层
        :class:`~pipeline.composer.Pipeline` 处理部分结果保留(R0-7)。

        Args:
            ctx: 运行期 :class:`NodeContext`。
            **inputs: 与 :meth:`execute` 相同的关键字输入。

        Returns:
            节点输出的字典。

        Raises:
            Exception: 重新抛出 :meth:`execute` 抛出的异常。
        """
        # 在执行前校验输入,提前发现缺失/非法的必填输入。
        # 放在 try 块之外,使校验失败直接抛出 ValueError 而不被
        # 当作"执行失败"记录。
        errors = self.validate_inputs(inputs)
        if errors:
            raise ValueError(
                "Input validation failed: {}".format(errors)
            )
        try:
            return self.execute(ctx, **inputs)
        except Exception as exc:
            logger = getattr(ctx, "logger", None) or _logger
            logger.error(
                "节点 %s 执行失败 (%s): %s",
                getattr(self.spec, "type", self.__class__.__name__),
                type(exc).__name__,
                exc,
            )
            raise

    # ------------------------------------------------------------------
    # Reusable validation
    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate ``inputs`` against :attr:`spec`.

        The base implementation checks that every *required* input
        (one whose declared type string is not wrapped in
        ``Optional[...]``) is present and not ``None``.  Because port
        types are now opaque strings (e.g. ``"IMAGE"``, ``"INT"``),
        runtime ``isinstance`` checks are no longer possible; the value
        is only checked for ``None``.  Unknown inputs are ignored
        (lenient) so that pipelines can pass extra metadata through a
        node without erroring.

        Subclasses are expected to call ``super().validate_inputs(inputs)``
        first and then append any domain-specific errors.

        Args:
            inputs: The input dictionary to validate.

        Returns:
            A list of human-readable error strings; empty when valid.
        """
        errors: List[str] = []
        spec = self.spec
        for name, type_str in spec.inputs.items():
            optional = is_optional(type_str)
            if name not in inputs:
                if not optional:
                    errors.append(
                        "Missing required input {!r} for node {!r}.".format(
                            name, spec.type
                        )
                    )
                continue
            value = inputs[name]
            if value is None and not optional:
                errors.append(
                    "Required input {!r} for node {!r} is None.".format(
                        name, spec.type
                    )
                )
        return errors

    # ------------------------------------------------------------------
    # Reusable resource estimation
    # ------------------------------------------------------------------
    def estimate_resources(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        """Estimate the resources this node would consume for ``inputs``.

        Returns a dictionary with three keys:

        * ``vram_gb`` -- estimated GPU memory in gigabytes.
        * ``ram_gb`` -- estimated host memory in gigabytes.
        * ``time_s`` -- estimated wall-clock time in seconds.

        The base implementation applies a generic heuristic: a small
        base overhead plus, when the inputs carry spatial dimensions
        (``width`` / ``height``) and a step count (``steps``), a
        pixel-and-step scaling term.  Subclasses override with
        domain-specific formulas (see e.g. :class:`nodes.image.ImageTxt2ImgNode`).

        Args:
            inputs: The input dictionary the node would be executed with.

        Returns:
            A ``{"vram_gb", "ram_gb", "time_s"}`` dictionary.
        """
        vram_gb: float = _BASE_VRAM_GB
        ram_gb: float = _BASE_RAM_GB
        time_s: float = _BASE_TIME_S

        width = inputs.get("width")
        height = inputs.get("height")
        steps = inputs.get("steps")
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            pixels = float(width) * float(height)
            megapixels = pixels / (_MEGAPIXEL_PIXELS)
            vram_gb += megapixels * _VRAM_PER_MEGAPIXEL_GB
            ram_gb += megapixels * _RAM_PER_MEGAPIXEL_GB
            if isinstance(steps, (int, float)) and steps > 0:
                time_s += float(steps) * _TIME_PER_STEP_S * (
                    pixels / (_REFERENCE_PIXELS)
                )

        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return "<{cls} type={type!r} name={name!r}>".format(
            cls=self.__class__.__name__,
            type=self.spec.type,
            name=self.spec.name,
        )


# ---------------------------------------------------------------------------
# Estimation coefficients (module-level, overridable by subclasses)
# ---------------------------------------------------------------------------
#: Base VRAM overhead (GB) assumed for any node before scaling.
_BASE_VRAM_GB: float = 0.5
#: Base host-RAM overhead (GB) assumed for any node before scaling.
_BASE_RAM_GB: float = 0.25
#: Base wall-clock time (s) assumed for any node before scaling.
_BASE_TIME_S: float = 1.0
#: Number of pixels in one megapixel (used to normalise spatial estimates).
_MEGAPIXEL_PIXELS: float = 1_000_000.0
#: Additional VRAM (GB) per megapixel of output resolution.
_VRAM_PER_MEGAPIXEL_GB: float = 0.25
#: Additional host RAM (GB) per megapixel of output resolution.
_RAM_PER_MEGAPIXEL_GB: float = 0.10
#: Reference resolution (512x512) used to normalise per-step time.
_REFERENCE_PIXELS: float = 512.0 * 512.0
#: Wall-clock seconds per denoising step at the reference resolution.
_TIME_PER_STEP_S: float = 0.05


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------
def _register_node_class(
    cls: type[BaseNode],
    bus: Optional[ModuleBus] = None,
) -> type[BaseNode]:
    """Register a node class with the bus and the module-level index.

    Args:
        cls: The :class:`BaseNode` subclass to register.
        bus: Optional explicit :class:`ModuleBus`.  When ``None`` the
            process-wide singleton is used.

    Returns:
        The class unchanged (so it can be used as a decorator return).

    Raises:
        TypeError: If ``cls.spec`` is not a :class:`NodeSpec`.
        ValueError: If ``cls.spec.type`` is empty.
    """
    spec = getattr(cls, "spec", None)
    if not isinstance(spec, NodeSpec):
        raise TypeError(
            "{}.spec must be a NodeSpec instance, got {!r}.".format(
                cls.__name__, spec
            )
        )
    if not spec.type:
        raise ValueError(
            "{}.spec.type must be a non-empty string.".format(cls.__name__)
        )

    registry_bus = bus if bus is not None else ModuleBus()
    with _NODE_CLASSES_LOCK:
        _NODE_CLASSES[spec.type] = cls
    registry_bus.register(
        kind=_NODE_KIND,
        name=spec.type,
        factory=cls,
        description=spec.description,
        tags=list(spec.tags),
    )
    _logger.debug(
        "Registered node type=%s class=%s.", spec.type, cls.__name__
    )
    return cls


def _unregister_node_class(
    node_type: str,
    bus: Optional[ModuleBus] = None,
) -> bool:
    """Remove a node class from the bus and the module-level index.

    This is the inverse of :func:`_register_node_class` and is used by the
    plugin system to unload a plugin's nodes.  After the call the node
    type is no longer discoverable through :class:`NodeRegistry` /
    :class:`ModuleBus`.

    Args:
        node_type: The node type identifier to remove.
        bus: Optional explicit :class:`ModuleBus`.  When ``None`` the
            process-wide singleton is used.

    Returns:
        ``True`` if a node was removed from the bus, ``False`` otherwise.
    """
    registry_bus = bus if bus is not None else ModuleBus()
    existed = registry_bus.unregister(_NODE_KIND, node_type)
    with _NODE_CLASSES_LOCK:
        cls = _NODE_CLASSES.pop(node_type, None)
    if cls is not None:
        _logger.debug(
            "Unregistered node type=%s class=%s.", node_type,
            getattr(cls, "__name__", "?"),
        )
    return existed or cls is not None


def register_node(node_type: str) -> Callable[[type[BaseNode]], type[BaseNode]]:
    """Class decorator that registers a :class:`BaseNode` subclass.

    The decorated class must define a ``spec`` :class:`NodeSpec`.  The
    ``node_type`` argument is authoritative: it is written back into
    ``cls.spec.type`` (via :func:`dataclasses.replace`) so the spec and
    the registration key can never drift apart.

    Example::

        @register_node("text_chat")
        class TextNode(BaseNode):
            spec = NodeSpec(type="text_chat", name="Text Chat", ...)

    Args:
        node_type: The unique node type identifier to register under.

    Returns:
        A decorator that registers the class and returns it unchanged.

    Raises:
        TypeError: If the decorated object has no valid ``spec``.
        ValueError: If ``node_type`` is empty.
    """
    if not isinstance(node_type, str) or not node_type.strip():
        raise ValueError("node_type must be a non-empty string.")

    def decorator(cls: type[BaseNode]) -> type[BaseNode]:
        spec = getattr(cls, "spec", None)
        if not isinstance(spec, NodeSpec):
            raise TypeError(
                "@register_node can only decorate classes with a NodeSpec "
                "'spec' attribute; {} has {!r}.".format(cls.__name__, spec)
            )
        if spec.type != node_type:
            cls.spec = replace(spec, type=node_type)
        _register_node_class(cls)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# NodeRegistry
# ---------------------------------------------------------------------------
class NodeRegistry:
    """Discovery and instantiation facade for nodes, backed by ModuleBus.

    :class:`NodeRegistry` is a thin wrapper over :class:`ModuleBus` (the
    v0.3.0 single assembly point).  Nodes are registered under the
    ``"node"`` kind; this class exposes node-centric operations --
    ``register``, ``get``, ``list`` and ``search`` -- on top of the bus.

    Because :class:`ModuleBus` is a process-wide singleton, a freshly
    constructed :class:`NodeRegistry` immediately sees every node that
    was registered via the :func:`register_node` decorator at import
    time.  A custom bus may be supplied (e.g. for an isolated test bus).

    Example::

        @register_node("text_chat")
        class TextNode(BaseNode): ...

        registry = NodeRegistry()
        specs = registry.list()                 # all registered nodes
        node = registry.get("text_chat")        # a TextNode instance
        hits = registry.search("image")         # nodes mentioning "image"
    """

    def __init__(self, bus: Optional[ModuleBus] = None) -> None:
        self._bus: ModuleBus = bus if bus is not None else ModuleBus()

    @property
    def bus(self) -> ModuleBus:
        """The underlying :class:`ModuleBus` used for resolution."""
        return self._bus

    # ------------------------------------------------------------------
    def register(self, node_class: type[BaseNode]) -> None:
        """Register a node class with this registry's bus.

        Args:
            node_class: A :class:`BaseNode` subclass with a valid
                ``spec``.
        """
        _register_node_class(node_class, bus=self._bus)

    # ------------------------------------------------------------------
    def unregister(self, node_type: str) -> bool:
        """Remove a node from this registry's bus.

        Args:
            node_type: The node type identifier to remove.

        Returns:
            ``True`` if a node was removed, ``False`` otherwise.
        """
        return _unregister_node_class(node_type, bus=self._bus)

    # ------------------------------------------------------------------
    def get(self, node_type: str) -> BaseNode:
        """Return a (cached) instance of the node registered as ``node_type``.

        Resolution delegates to :class:`ModuleBus`; the factory (the node
        class itself) is invoked at most once per type and the instance is
        cached by the bus.  When the node is not on the bus the
        module-level index is consulted as a fallback so that nodes
        registered only in-memory are still reachable.

        Args:
            node_type: The node type identifier (e.g. ``"image_txt2img"``).

        Returns:
            A :class:`BaseNode` instance.

        Raises:
            KeyError: If no node is registered for ``node_type``.
        """
        try:
            instance = self._bus.resolve(_NODE_KIND, node_type)
            return instance  # type: ignore[return-value]
        except _BusNotFoundError:
            with _NODE_CLASSES_LOCK:
                cls = _NODE_CLASSES.get(node_type)
            if cls is None:
                raise KeyError(
                    "No node registered for type {!r}.".format(node_type)
                )
            return cls()

    # ------------------------------------------------------------------
    def list(self) -> List[NodeSpec]:
        """Return the :class:`NodeSpec` of every registered node.

        The :class:`ModuleBus` is the authoritative discovery surface;
        the module-level index is consulted only to retrieve the
        :class:`NodeSpec` without instantiating the node, and as a
        defensive fallback for nodes registered in-memory only.

        Returns:
            A list of :class:`NodeSpec` sorted by ``type``.
        """
        specs: List[NodeSpec] = []
        seen: set[str] = set()

        for module_spec in self._bus.list(_NODE_KIND):
            if module_spec.kind != _NODE_KIND:
                # Skip nested "node.*" namespaces -- only exact "node".
                continue
            with _NODE_CLASSES_LOCK:
                cls = _NODE_CLASSES.get(module_spec.name)
            if cls is not None:
                specs.append(cls.spec)
                seen.add(module_spec.name)

        # Defensive: include any in-memory-only registrations.
        with _NODE_CLASSES_LOCK:
            node_classes_items = list(_NODE_CLASSES.items())
        for node_type, cls in node_classes_items:
            if node_type not in seen:
                specs.append(cls.spec)
                seen.add(node_type)

        specs.sort(key=lambda s: s.type)
        return specs

    # ------------------------------------------------------------------
    def search(self, query: str) -> List[NodeSpec]:
        """Fuzzy-search nodes by type, name, description or tags.

        The match is case-insensitive and matches any node whose type,
        name, description or any tag *contains* the query substring.
        An empty query returns every node (same as :meth:`list`).

        Args:
            query: Substring to search for.

        Returns:
            A list of matching :class:`NodeSpec` sorted by ``type``.
        """
        needle = (query or "").strip().lower()
        if not needle:
            return self.list()

        results: List[NodeSpec] = []
        for spec in self.list():
            haystack = " ".join(
                [
                    spec.type,
                    spec.name,
                    spec.description,
                    " ".join(spec.tags),
                ]
            ).lower()
            if needle in haystack:
                results.append(spec)
        return results

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return "NodeRegistry(bus={!r}, nodes={})".format(
            self._bus, len(self.list())
        )
