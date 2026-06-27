"""The :class:`PipelineService` class (v0.6.x).

The constructor, the executor-bridge (``_make_executor`` /
``_run``) and the property accessors live in this module.  The
capability methods are attached at import time:

* :func:`serving.service._service_text.attach_text_methods` --
  ``text_completion`` / ``text_chat``
* :func:`serving.service._service_media.attach_media_methods` --
  ``image_txt2img`` / ``image_img2img`` / ``audio_tts`` /
  ``video_txt2vid``
* :func:`serving.service._service_llm.attach_llm_methods` --
  ``multimodal_understand`` / ``rag_query`` / ``agent_run``
* :func:`serving.service._service_llm.attach_list_models` --
  ``list_models``

This pattern (decorator-style duck-punching) preserves the
public surface of :class:`PipelineService` -- callers and tests
can still use ``PipelineService.text_chat = my_override`` and
``monkeypatch.setattr(service, "image_txt2img", ...)`` because
the methods are real attributes on the class object.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from infrastructure.cache_store import CacheStore
from infrastructure.config_center import ConfigCenter
from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger
from infrastructure.rate_limiter import RateLimiter
from security.input_sanitizer import InputSanitizer
from security.output_filter import OutputFilter

from nodes import NodeRegistry
from nodes.base import NodeContext
from pipeline.composer import PipelineBuilder

from serving.metrics import MetricsCollector

__all__ = ["PipelineService"]


class PipelineService:
    """Service layer that bridges the REST API to the Pipeline/Node system.

    Each capability method builds a short single-node :class:`Pipeline`
    via :class:`PipelineBuilder`, runs it through the L5 composer and
    returns the produced node outputs as a dictionary.  Node executors
    are resolved lazily: a per-node-type callable is registered on a
    fresh :class:`NodeContext` for every run, bridging the L5
    composer to the L4 node system.

    Methods return the node's output dictionary on success (e.g.
    ``{"text": ..., "usage": ...}``) or ``{"error": ..., "error_type":
    ...}`` when the pipeline cannot be built or run.

    Capabilities without a backing node (multimodal understanding, RAG
    query, agent execution) return a ``not_implemented`` error response
    so the REST contract stays intact while the node backends mature.
    """

    def __init__(self) -> None:
        self._cfg: ConfigCenter = ConfigCenter()
        self._device_manager: DeviceManager = DeviceManager()
        self._logger = get_logger("PipelineService")
        self._metrics: MetricsCollector = MetricsCollector()
        self._registry: NodeRegistry = NodeRegistry()

        # Security gates (Gate 1 input sanitiser + Gate 3 output filter).
        self._sanitizer: InputSanitizer = InputSanitizer()
        self._filter: OutputFilter = OutputFilter()

        # Build a reusable executor map (node_type -> callable).  Each
        # executor creates a lightweight L4 NodeContext and dispatches to
        # the registered node, reading run-level config from the L5
        # composer context's metadata.
        self._executors: Dict[str, Any] = {}
        for spec in self._registry.list():
            self._executors[spec.type] = self._make_executor(spec.type)

        # RAG stack: vector store + retriever.  Built lazily on first
        # ``rag_query`` call (cheap to construct, no eager I/O).
        self._rag_store = None
        self._rag_retriever = None

        # LLM provider: the same instance is shared by the ReAct agent
        # used by ``agent_run``.  Eagerly constructed as a
        # :class:`ChatTemplateProvider` wrapping the project-owned
        # :class:`LocalTorchTextProvider` (micro-transformer with
        # a randomly-initialised but trainable PyTorch forward
        # path).  Operators can override it at runtime by setting
        # ``service.llm_provider = ...`` before serving traffic --
        # e.g. to plug a user-downloaded Qwen2.5 checkpoint in
        # via ``ChatTemplateProvider(TransformerLM.from_pretrained(...))``
        # (v0.11.0 item) or any other LLMProvider-compatible backend.
        from models.interfaces.llm_provider import ChatTemplateProvider
        from models.providers.local_text import LocalTorchTextProvider

        self._llm_provider: Any = ChatTemplateProvider(
            LocalTorchTextProvider.from_random(),
            name="torcha-verse-micro-transformer",
        )

        # Cache for idempotent generation results.
        cache_cfg = self._cfg.get("serving.cache", {})
        self._cache: CacheStore = CacheStore(
            max_size=cache_cfg.get("max_size", 256),
            ttl=cache_cfg.get("ttl", 300),
        )

        # Rate limiter.
        rate_cfg = self._cfg.get("serving.rate_limit", {})
        self._rate_limiter: RateLimiter = RateLimiter(
            rate=rate_cfg.get("rate", 100),
            burst=rate_cfg.get("burst", 200),
        )

        self._logger.info(
            "PipelineService initialised with %d node executors.",
            len(self._executors),
        )

    # ------------------------------------------------------------------
    # Executor bridge
    # ------------------------------------------------------------------
    def _make_executor(self, node_type: str) -> Any:
        """Create an L5 executor that dispatches to the L4 node ``node_type``.

        The returned callable has the signature ``(inputs, ctx) ->
        outputs`` expected by :class:`pipeline.composer.Pipeline`.  It
        reads the run-level config (model defaults) from the composer
        context's ``config["node_config"]`` bag.
        """
        registry = self._registry

        def _executor(inputs: Dict[str, Any], ctx: NodeContext) -> Dict[str, Any]:
            run_config: Dict[str, Any] = ctx.config.get("node_config", {})
            node_ctx = NodeContext(config=dict(run_config))
            node = registry.get(node_type)
            return node.execute(node_ctx, **inputs)

        return _executor

    def _run(
        self,
        name: str,
        node_type: str,
        node_id: str,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build, run and return the output of a single-node pipeline.

        Args:
            name: Pipeline name (for logging / serialisation).
            node_type: The L4 node type to execute.
            node_id: The node id used to look up the result.
            inputs: Static inputs forwarded to the node.
            config: Optional run-level config (model defaults).

        Returns:
            The node's output dictionary, or ``{"error": ...}`` on
            failure.
        """
        try:
            ctx = NodeContext(
                executors=self._executors,
                config={"node_config": config or {}},
            )
            pipeline = (
                PipelineBuilder(name)
                .node(node_type, id=node_id, **inputs)
                .build()
            )
            results = pipeline.run(ctx)
            return results.get(node_id, {})
        except Exception as exc:  # noqa: BLE001 - surface as API error
            self._logger.error("Pipeline '%s' failed: %s", name, exc)
            return {"error": str(exc), "error_type": "pipeline_error"}

    # ------------------------------------------------------------------
    # Introspection (properties)
    # ------------------------------------------------------------------
    @property
    def metrics(self) -> MetricsCollector:
        """The metrics collector."""
        return self._metrics

    @property
    def cache(self) -> CacheStore:
        """The result cache."""
        return self._cache

    @property
    def rate_limiter(self) -> RateLimiter:
        """The rate limiter."""
        return self._rate_limiter

    @property
    def device_manager(self) -> DeviceManager:
        """The device manager."""
        return self._device_manager

    @property
    def llm_provider(self) -> Any:
        """The shared :class:`LLMProvider` used by RAG and agent paths.

        Operators can override the default :class:`EchoProvider` by
        assigning to this property before serving traffic::

            service.llm_provider = ChatTemplateProvider(my_transformer)
        """
        return self._llm_provider

    @llm_provider.setter
    def llm_provider(self, provider: Any) -> None:
        self._llm_provider = provider
        # Reset the RAG stack so the next rag_query picks up the new
        # provider's ``embed`` function.
        self._rag_store = None
        self._rag_retriever = None


# ---------------------------------------------------------------------------
# Attach capability methods to the class
# ---------------------------------------------------------------------------
# Done at import time so the public surface of
# :class:`PipelineService` is unchanged: every method is a real
# attribute of the class, so ``PipelineService.text_chat = ...``
# and ``monkeypatch.setattr(service, "text_chat", ...)`` keep
# working in tests.
from ._service_text import attach_text_methods  # noqa: E402
from ._service_media import attach_media_methods  # noqa: E402
from ._service_llm import attach_llm_methods, attach_list_models  # noqa: E402

attach_text_methods(PipelineService)
attach_media_methods(PipelineService)
attach_llm_methods(PipelineService)
attach_list_models(PipelineService)
