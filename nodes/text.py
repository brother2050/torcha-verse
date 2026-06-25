"""Text generation nodes for the TorchaVerse L4 capability layer.

This module decomposes the v0.1.0 ``text_engine.py`` "god class" (999
lines) into small, single-responsibility nodes:

* :class:`TextNode` (``text_chat``) -- conversational / chat-style
  generation driven by a prompt, model, sampling budget and temperature.
* :class:`TextCompletionNode` (``text_completion``) -- raw prompt
  completion without the chat template.

Both nodes share the same typed contract (``prompt`` / ``model`` /
``max_tokens`` -> ``text`` / ``usage``) and the same resource-estimation
heuristic.  Their :meth:`execute` implementations are intentionally
placeholder stubs that return deterministic mock data -- the *interface*
is complete and ready for the real model backend to be wired in via the
:class:`ModuleBus`; only the body needs replacing.

The nodes honour the v0.3.0 "no hardcoded constants" rule: the model
identifier is read from the ``model`` input and falls back to
``ctx.config["default_text_model"]``; sampling parameters come from the
inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseNode, NodeContext, NodeSpec, register_node

__all__ = ["TextNode", "TextCompletionNode"]


# ---------------------------------------------------------------------------
# Estimation coefficients (text-specific)
# ---------------------------------------------------------------------------
#: Approximate VRAM (GB) occupied per billion parameters of a text model.
_TEXT_VRAM_PER_BILLION_PARAMS_GB: float = 2.0
#: Default assumed parameter count (in billions) when the model is unknown.
_DEFAULT_TEXT_MODEL_PARAMS_B: float = 7.0
#: Wall-clock seconds per generated token at the reference throughput.
_TEXT_TIME_PER_TOKEN_S: float = 0.02
#: Host RAM (GB) reserved for tokenisation buffers / KV cache accounting.
_TEXT_RAM_GB: float = 0.5


@register_node("text_chat")
class TextNode(BaseNode):
    """Conversational text generation node (``text_chat``).

    Produces a chat-style response for the given prompt.  The model is
    resolved through the :class:`ModuleBus` (kind ``"model.text"``);
    sampling is controlled by ``max_tokens`` and ``temperature``.

    Inputs:
        prompt: The user prompt / instruction (required).
        model: Registered text model name.  Optional; falls back to
            ``ctx.config["default_text_model"]``.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature in ``[0, 2]``.

    Outputs:
        text: The generated text.
        usage: Token-usage dictionary with ``prompt_tokens``,
            ``completion_tokens`` and ``total_tokens``.
    """

    spec = NodeSpec(
        type="text_chat",
        name="Text Chat",
        description="Conversational text generation from a prompt.",
        inputs={
            "prompt": "PROMPT",
            "model": "Optional[TEXT]",
            "max_tokens": "INT",
            "temperature": "FLOAT",
        },
        outputs={
            "text": "TEXT",
            "usage": "TEXT",
        },
        tags=["text", "generation", "llm", "chat"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate text-chat inputs.

        Extends the base type checks with:

        * ``temperature`` must be in ``[0, 2]``.
        * ``max_tokens`` must be a positive integer.
        * ``prompt`` must be a non-empty string.
        """
        errors = super().validate_inputs(inputs)

        temperature = inputs.get("temperature")
        if isinstance(temperature, (int, float)) and not (
            0.0 <= float(temperature) <= 2.0
        ):
            errors.append(
                "Input 'temperature' for node 'text_chat' must be in "
                "[0, 2], got {}.".format(temperature)
            )

        max_tokens = inputs.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens <= 0:
            errors.append(
                "Input 'max_tokens' for node 'text_chat' must be > 0, "
                "got {}.".format(max_tokens)
            )

        prompt = inputs.get("prompt")
        if isinstance(prompt, str) and not prompt.strip():
            errors.append(
                "Input 'prompt' for node 'text_chat' must be a non-empty "
                "string."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate VRAM / RAM / time for a text-chat run.

        VRAM scales with the (assumed) parameter count of the model;
        time scales with the requested ``max_tokens``.
        """
        max_tokens = inputs.get("max_tokens", 0)
        max_tokens = max_tokens if isinstance(max_tokens, (int, float)) else 0
        params_b = _DEFAULT_TEXT_MODEL_PARAMS_B

        vram_gb = _TEXT_VRAM_PER_BILLION_PARAMS_GB * params_b
        ram_gb = _TEXT_RAM_GB
        time_s = float(max_tokens) * _TEXT_TIME_PER_TOKEN_S

        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Generate a chat response (placeholder implementation).

        .. note::
            This is a stub that returns deterministic mock data.  The real
            implementation will resolve the model via
            ``ctx.bus.resolve("model.text", model)`` and run sampling.

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``prompt``, ``model``, ``max_tokens``, ``temperature``.

        Returns:
            A dict with ``text`` and ``usage``.
        """
        prompt = str(inputs.get("prompt", ""))
        model = inputs.get("model") or ctx.config.get("default_text_model")
        _mt = inputs.get("max_tokens", 256)
        max_tokens = int(_mt) if _mt is not None else 256
        _temp = inputs.get("temperature", 0.7)
        temperature = float(_temp) if _temp is not None else 0.7

        ctx.logger.debug(
            "text_chat run_id=%s model=%s max_tokens=%d temperature=%.2f",
            ctx.run_id,
            model,
            max_tokens,
            temperature,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.text_chat",
                action="generate",
                resource_id=model,
                details={
                    "run_id": ctx.run_id,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, min(max_tokens, 16))
        text = (
            "[text_chat placeholder] model={!r}: ".format(model)
            + prompt[: 64]
        )
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "model": model,
        }
        return {"text": text, "usage": usage}


@register_node("text_completion")
class TextCompletionNode(BaseNode):
    """Raw prompt-completion node (``text_completion``).

    Unlike :class:`TextNode`, this node performs plain completion without
    applying a chat template -- suitable for code completion, fill-in-the
    -middle and other non-conversational tasks.

    Inputs:
        prompt: The prompt to complete (required).
        model: Registered text model name.  Optional; falls back to
            ``ctx.config["default_text_model"]``.
        max_tokens: Maximum number of tokens to generate.

    Outputs:
        text: The completed text.
        usage: Token-usage dictionary.
    """

    spec = NodeSpec(
        type="text_completion",
        name="Text Completion",
        description="Raw prompt completion without a chat template.",
        inputs={
            "prompt": "PROMPT",
            "model": "Optional[TEXT]",
            "max_tokens": "INT",
        },
        outputs={
            "text": "TEXT",
            "usage": "TEXT",
        },
        tags=["text", "generation", "llm", "completion"],
    )

    # ------------------------------------------------------------------
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate text-completion inputs.

        Extends the base checks with:

        * ``max_tokens`` must be a positive integer.
        * ``prompt`` must be a non-empty string.
        """
        errors = super().validate_inputs(inputs)

        max_tokens = inputs.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens <= 0:
            errors.append(
                "Input 'max_tokens' for node 'text_completion' must be > 0, "
                "got {}.".format(max_tokens)
            )

        prompt = inputs.get("prompt")
        if isinstance(prompt, str) and not prompt.strip():
            errors.append(
                "Input 'prompt' for node 'text_completion' must be a "
                "non-empty string."
            )

        return errors

    # ------------------------------------------------------------------
    def estimate_resources(
        self, inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate VRAM / RAM / time for a text-completion run."""
        max_tokens = inputs.get("max_tokens", 0)
        max_tokens = max_tokens if isinstance(max_tokens, (int, float)) else 0
        params_b = _DEFAULT_TEXT_MODEL_PARAMS_B

        vram_gb = _TEXT_VRAM_PER_BILLION_PARAMS_GB * params_b
        ram_gb = _TEXT_RAM_GB
        time_s = float(max_tokens) * _TEXT_TIME_PER_TOKEN_S

        return {
            "vram_gb": round(vram_gb, 4),
            "ram_gb": round(ram_gb, 4),
            "time_s": round(time_s, 4),
        }

    # ------------------------------------------------------------------
    def execute(
        self, ctx: NodeContext, **inputs: Any
    ) -> Dict[str, Any]:
        """Complete a prompt (placeholder implementation).

        .. note::
            Stub returning deterministic mock data; the real backend will
            be wired through the :class:`ModuleBus`.

        Args:
            ctx: The runtime :class:`NodeContext`.
            **inputs: ``prompt``, ``model``, ``max_tokens``.

        Returns:
            A dict with ``text`` and ``usage``.
        """
        prompt = str(inputs.get("prompt", ""))
        model = inputs.get("model") or ctx.config.get("default_text_model")
        _mt = inputs.get("max_tokens", 128)
        max_tokens = int(_mt) if _mt is not None else 128

        ctx.logger.debug(
            "text_completion run_id=%s model=%s max_tokens=%d",
            ctx.run_id,
            model,
            max_tokens,
        )
        if ctx.audit is not None:
            ctx.audit.log(
                "INFER",
                actor="node.text_completion",
                action="complete",
                resource_id=model,
                details={
                    "run_id": ctx.run_id,
                    "max_tokens": max_tokens,
                },
                severity="info",
            )

        # --- placeholder body -------------------------------------------------
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, min(max_tokens, 16))
        text = (
            "[text_completion placeholder] model={!r}: ".format(model)
            + prompt[: 64]
        )
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "model": model,
        }
        return {"text": text, "usage": usage}
