"""Text generation engine for TorchaVerse.

This module provides :class:`TextEngine`, the capability-layer entry point
for all text generation tasks.  It composes a large language model (loaded
from the :class:`ModelRegistry`) with a :class:`TextTokenizer` (from the
:class:`TokenizerHub`), a :class:`KVCacheManager` for efficient
autoregressive decoding, and a :class:`SamplingStrategy` that encapsulates
temperature / top-k / top-p / repetition-penalty logic.

Advanced features:

* **Multi-turn chat** -- conversation history management with sliding-window
  compression.
* **Function calling** -- parse LLM output for tool invocations and execute
  them through the :class:`ToolRegistry`.
* **Structured output** -- constrain decoding to a JSON schema.
* **Speculative decoding** -- draft tokens with a small model and verify
  them with the large model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from core.kv_cache_manager import KVCacheManager
from core.model_registry import BaseModel, ModelRegistry
from core.tokenizer_hub import TextTokenizer, TokenizerHub
from core.tool_registry import ToolRegistry, ToolResult
from infrastructure.config_manager import ConfigManager
from infrastructure.device_manager import DeviceManager
from infrastructure.error_handler import ErrorHandler
from infrastructure.logger import get_logger

__all__ = [
    "Message",
    "ToolCall",
    "SamplingStrategy",
    "TextEngine",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    """Represents a single function-call request emitted by the LLM.

    Attributes:
        name: The tool / function name to invoke.
        arguments: Keyword arguments parsed from the LLM output.
        id: Optional call identifier used to correlate results.
    """

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: One of ``"system"``, ``"user"``, ``"assistant"``, or
            ``"tool"``.
        content: The textual content of the message.
        tool_calls: Optional list of :class:`ToolCall` objects requested
            by the assistant.
        tool_call_id: When ``role == "tool"``, the id of the tool call
            this message responds to.
        name: Optional name of the tool that produced this message.
    """

    role: str
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary (OpenAI-style)."""
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id or "",
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        """Deserialise from a dictionary."""
        tool_calls: List[ToolCall] = []
        for raw in d.get("tool_calls", []):
            fn = raw.get("function", raw)
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            tool_calls.append(
                ToolCall(name=fn.get("name", ""), arguments=args, id=raw.get("id"))
            )
        return cls(
            role=d.get("role", "user"),
            content=d.get("content", ""),
            tool_calls=tool_calls,
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
        )


# ---------------------------------------------------------------------------
# SamplingStrategy
# ---------------------------------------------------------------------------
class SamplingStrategy:
    """Encapsulates token sampling logic.

    Supports temperature scaling, top-k filtering, top-p (nucleus)
    filtering, and repetition penalty.

    Args:
        temperature: Sampling temperature (``> 0``).  ``0`` or ``None``
            selects greedy decoding.
        top_k: Keep only the *k* highest-probability tokens.  ``0``
            disables top-k filtering.
        top_p: Nucleus sampling threshold in ``(0, 1]``.  ``1.0``
            disables nucleus filtering.
        repetition_penalty: Penalty applied to already-seen tokens.
            ``1.0`` disables the penalty.
    """

    def __init__(
        self,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> None:
        self.temperature: float = temperature
        self.top_k: int = top_k
        self.top_p: float = top_p
        self.repetition_penalty: float = repetition_penalty

    # ------------------------------------------------------------------
    def apply(
        self,
        logits: torch.Tensor,
        prev_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the sampling strategy and return next-token ids.

        Args:
            logits: Logits of shape ``(batch, vocab_size)``.
            prev_tokens: Optional tensor of previously generated token
                ids ``(batch, seq_len)`` used for repetition penalty.

        Returns:
            Next-token ids of shape ``(batch,)``.
        """
        # Greedy decoding.
        if self.temperature is None or self.temperature <= 0:
            return torch.argmax(logits, dim=-1)

        logits = logits / self.temperature

        # Repetition penalty.
        if (
            self.repetition_penalty != 1.0
            and prev_tokens is not None
            and prev_tokens.numel() > 0
        ):
            for b in range(logits.shape[0]):
                gathered = prev_tokens[b]
                gathered = gathered[gathered >= 0]
                if gathered.numel() == 0:
                    continue
                score = torch.gather(logits[b], 0, gathered)
                score = torch.where(
                    score < 0,
                    score * self.repetition_penalty,
                    score / self.repetition_penalty,
                )
                logits[b].scatter_(0, gathered, score)

        # Top-k filtering.
        if self.top_k and self.top_k > 0:
            k = min(self.top_k, logits.size(-1))
            values, _ = torch.topk(logits, k, dim=-1)
            min_values = values[:, -1, None]
            logits = logits.masked_fill(logits < min_values, float("-inf"))

        # Top-p (nucleus) filtering.
        if self.top_p is not None and 0.0 < self.top_p < 1.0:
            logits = self._top_p_filter(logits, self.top_p)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)

    # ------------------------------------------------------------------
    @staticmethod
    def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Filter logits using nucleus (top-p) sampling."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        return logits.masked_fill(indices_to_remove, float("-inf"))

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return the strategy parameters as a dictionary."""
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }

    def __repr__(self) -> str:
        return (
            f"SamplingStrategy(temperature={self.temperature}, top_k={self.top_k}, "
            f"top_p={self.top_p}, repetition_penalty={self.repetition_penalty})"
        )


# ---------------------------------------------------------------------------
# TextEngine
# ---------------------------------------------------------------------------
class TextEngine:
    """Text generation engine.

    Composes an LLM, a text tokenizer, a KV-cache manager, and a sampling
    strategy to provide a high-level text-generation API.

    Args:
        model_name: Registered model name in the :class:`ModelRegistry`.
        config: Optional configuration dictionary.  When ``None`` the
            global :class:`ConfigManager` is consulted.
        device: Optional device override.
        dtype: Optional dtype override.
        checkpoint_path: Optional path to model weights.
        draft_model_name: Optional name of a small draft model for
            speculative decoding.
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        checkpoint_path: Optional[str] = None,
        draft_model_name: Optional[str] = None,
    ) -> None:
        self.model_name: str = model_name
        self._config: Dict[str, Any] = config or {}
        self._cfg_manager: ConfigManager = ConfigManager()
        self._device_manager: DeviceManager = DeviceManager()
        self._error_handler: ErrorHandler = ErrorHandler()
        self._logger = get_logger(f"TextEngine[{model_name}]")

        self._device: torch.device = (
            torch.device(device) if isinstance(device, str)
            else device or self._device_manager.get_device()
        )

        # Resolve model configuration from the YAML config.
        model_cfg = self._cfg_manager.get(f"text_models.{model_name}", {})
        merged_cfg: Dict[str, Any] = {**model_cfg, **self._config}

        # Load the model from the registry.
        self._registry: ModelRegistry = ModelRegistry()
        self._tokenizer_hub: TokenizerHub = TokenizerHub()

        try:
            self.model: BaseModel = self._registry.load(
                model_name,
                checkpoint_path=checkpoint_path,
                device=self._device,
                dtype=dtype,
                config=merged_cfg,
            )
        except KeyError:
            # Fallback: instantiate a small default transformer so the
            # engine is always usable.
            self._logger.warning(
                "Model '%s' is not registered. Instantiating a small "
                "default TransformerDecoder.", model_name,
            )
            from models.text.transformer import TransformerDecoder

            self.model = TransformerDecoder(
                vocab_size=merged_cfg.get("vocab_size", 256),
                hidden_size=merged_cfg.get("hidden_size", 128),
                num_layers=merged_cfg.get("num_layers", 2),
                num_heads=merged_cfg.get("num_heads", 4),
                num_kv_heads=merged_cfg.get("num_kv_heads", 4),
                intermediate_size=merged_cfg.get("intermediate_size", 256),
                max_seq_len=merged_cfg.get("max_seq_len", 512),
                config=merged_cfg,
            )
            self.model = self._device_manager.to_device(self.model, self._device)

        # Tokenizer.
        self.tokenizer: TextTokenizer = self._tokenizer_hub.get_tokenizer(  # type: ignore[assignment]
            "text",
            vocab_size=merged_cfg.get("vocab_size", 256),
            max_length=merged_cfg.get("max_seq_len", 512),
            device=self._device,
        )

        # KV cache manager.
        kv_cfg = self._cfg_manager.get("kv_cache", {})
        self.kv_cache: KVCacheManager = KVCacheManager(
            strategy=kv_cfg.get("strategy", "static"),
            num_layers=getattr(self.model, "num_layers", 2),
            num_heads=getattr(
                self.model, "num_kv_heads", getattr(self.model, "num_heads", 4)
            ),
            head_dim=getattr(self.model, "head_dim", 32),
            max_batch_size=1,
            max_seq_len=merged_cfg.get("max_seq_len", 512),
            device=self._device,
        )

        # Default sampling strategy.
        sampling_cfg = self._cfg_manager.get("sampling.default", {})
        self.sampling: SamplingStrategy = SamplingStrategy(
            temperature=sampling_cfg.get("temperature", 0.7),
            top_k=sampling_cfg.get("top_k", 50),
            top_p=sampling_cfg.get("top_p", 0.9),
            repetition_penalty=sampling_cfg.get("repetition_penalty", 1.1),
        )

        # Tool registry for function calling.
        self._tool_registry: ToolRegistry = ToolRegistry()

        # Conversation history.
        self._history: List[Message] = []
        self._max_history: int = merged_cfg.get("max_history", 20)

        # Draft model for speculative decoding.
        self._draft_model: Optional[BaseModel] = None
        if draft_model_name is not None:
            try:
                self._draft_model = self._registry.load(
                    draft_model_name, device=self._device, dtype=dtype
                )
            except KeyError:
                self._logger.warning(
                    "Draft model '%s' not registered; speculative "
                    "decoding will be disabled.", draft_model_name,
                )

        self._logger.info("TextEngine initialised with model '%s'.", model_name)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, model_name: str) -> "TextEngine":
        """Create a :class:`TextEngine` from the global configuration.

        Args:
            model_name: Registered model name.

        Returns:
            A configured :class:`TextEngine` instance.
        """
        cfg = ConfigManager()
        model_cfg = cfg.get(f"text_models.{model_name}", {})
        return cls(model_name, config=model_cfg)

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------
    def tokenize(self, text: str) -> List[int]:
        """Tokenise ``text`` into a list of token ids.

        Args:
            text: Input text.

        Returns:
            A list of integer token ids (without batch dimension).
        """
        ids = self.tokenizer.encode(text, return_tensors=False)
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return ids[0]
        if isinstance(ids, list):
            return ids  # type: ignore[return-value]
        return ids.tolist()  # type: ignore[union-attr]

    def detokenize(self, token_ids: List[int]) -> str:
        """Decode token ids back into text.

        Args:
            token_ids: List of token ids.

        Returns:
            The decoded text string.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        """Compute a dense embedding vector for ``text``.

        Uses the model's hidden states (mean-pooled over the sequence)
        as the embedding.

        Args:
            text: Input text.

        Returns:
            A 1-D embedding tensor of shape ``(hidden_size,)``.
        """
        input_ids = self.tokenizer.encode(text, return_tensors=True).to(self._device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # Run forward and extract hidden states before the LM head.
        hidden = self.model.forward(input_ids)
        if hidden.dim() == 3:
            embedding = hidden.mean(dim=1).squeeze(0)
        else:
            embedding = hidden.squeeze(0)
        return embedding.detach().cpu()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
    ) -> Union[str, Iterator[str]]:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering threshold.
            top_p: Nucleus sampling threshold.
            repetition_penalty: Repetition penalty factor.
            stream: When ``True`` returns an iterator yielding tokens
                one at a time.
            stop: Optional stop string or list of stop strings.

        Returns:
            The generated text string, or an iterator of text chunks
            when ``stream`` is ``True``.
        """
        strategy = SamplingStrategy(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        stop_list: List[str] = [stop] if isinstance(stop, str) else (stop or [])

        input_ids = self.tokenizer.encode(prompt, return_tensors=True).to(self._device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if stream:
            return self._generate_stream(input_ids, strategy, max_tokens, stop_list)

        generated_ids = self._generate_tokens(input_ids, strategy, max_tokens, stop_list)
        prompt_len = input_ids.shape[1]
        new_ids = generated_ids[0, prompt_len:].tolist()
        return self.detokenize(new_ids)

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        strategy: SamplingStrategy,
        max_tokens: int,
        stop_list: List[str],
    ) -> torch.Tensor:
        """Run the autoregressive generation loop.

        Args:
            input_ids: Prompt token ids ``(batch, seq_len)``.
            strategy: Sampling strategy.
            max_tokens: Maximum tokens to generate.
            stop_list: Stop strings.

        Returns:
            The full token tensor ``(batch, seq_len + n_generated)``.
        """
        self.model.eval()
        generated = input_ids
        prompt_len = input_ids.shape[1]
        eos_id = self.tokenizer.eos_token_id

        # Try the model's built-in generate for efficiency.
        try:
            output = self.model.generate(
                input_ids,
                max_tokens=max_tokens,
                temperature=strategy.temperature,
                top_k=strategy.top_k,
                top_p=strategy.top_p,
                eos_token_id=eos_id,
            )
            return output
        except Exception:
            pass

        # Manual generation loop.
        for _ in range(max_tokens):
            logits = self.model.forward(generated)
            next_logits = logits[:, -1, :]
            next_token = strategy.apply(next_logits, prev_tokens=generated)
            next_token = next_token.unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=-1)

            if stop_list:
                new_text = self.detokenize(generated[0, prompt_len:].tolist())
                if any(s in new_text for s in stop_list):
                    break

            if eos_id is not None and (next_token.item() == eos_id):
                break

        return generated

    def _generate_stream(
        self,
        input_ids: torch.Tensor,
        strategy: SamplingStrategy,
        max_tokens: int,
        stop_list: List[str],
    ) -> Iterator[str]:
        """Yield generated text chunks one at a time.

        Args:
            input_ids: Prompt token ids.
            strategy: Sampling strategy.
            max_tokens: Maximum tokens.
            stop_list: Stop strings.

        Yields:
            Decoded text chunks.
        """
        self.model.eval()
        generated = input_ids
        prompt_len = input_ids.shape[1]
        eos_id = self.tokenizer.eos_token_id
        chunk_size = self._cfg_manager.get("streaming.chunk_size", 4)

        buffer: List[int] = []
        for _ in range(max_tokens):
            logits = self.model.forward(generated)
            next_logits = logits[:, -1, :]
            next_token = strategy.apply(next_logits, prev_tokens=generated)
            tid = next_token.item()
            buffer.append(tid)
            generated = torch.cat(
                [generated, next_token.unsqueeze(-1)], dim=-1
            )

            if len(buffer) >= chunk_size:
                text = self.detokenize(buffer)
                if any(s in text for s in stop_list):
                    text = self._truncate_at_stop(text, stop_list)
                    if text:
                        yield text
                    return
                yield text
                buffer = []

            if eos_id is not None and tid == eos_id:
                break

        if buffer:
            text = self.detokenize(buffer)
            text = self._truncate_at_stop(text, stop_list)
            if text:
                yield text

    @staticmethod
    def _truncate_at_stop(text: str, stop_list: List[str]) -> str:
        """Truncate ``text`` at the first stop string."""
        for s in stop_list:
            idx = text.find(s)
            if idx != -1:
                return text[:idx]
        return text

    # ------------------------------------------------------------------
    # Chat (multi-turn)
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        max_tokens: int = 512,
    ) -> Message:
        """Run a multi-turn chat conversation.

        Args:
            messages: The conversation messages (system, user, assistant).
            tools: Optional list of tool schemas for function calling.
            stream: Whether to stream the output.
            max_tokens: Maximum tokens to generate.

        Returns:
            The assistant's reply as a :class:`Message`.
        """
        prompt = self._build_chat_prompt(messages)

        # If tools are provided, inject their descriptions.
        if tools:
            tool_descs = self._tool_registry.get_tool_descriptions()
            if tool_descs:
                tool_text = json.dumps(tool_descs, indent=2)
                prompt += f"\n[Available Tools]\n{tool_text}\n"

        response_text = self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=self.sampling.temperature,
            top_k=self.sampling.top_k,
            top_p=self.sampling.top_p,
            repetition_penalty=self.sampling.repetition_penalty,
            stream=stream,
        )

        if stream:
            chunks: List[str] = []
            assert isinstance(response_text, Iterator)
            for chunk in response_text:
                chunks.append(chunk)
            response_text = "".join(chunks)

        # Parse tool calls from the response.
        tool_calls = self._parse_tool_calls(response_text)

        # Execute tool calls if any.
        if tool_calls:
            for tc in tool_calls:
                result = self._tool_registry.execute_tool(tc.name, tc.arguments)
                tc.id = tc.id or f"call_{tc.name}_{id(tc)}"

        assistant_msg = Message(
            role="assistant",
            content=response_text,
            tool_calls=tool_calls,
        )

        # Update conversation history.
        self._history.extend(messages)
        self._history.append(assistant_msg)
        self._compress_history()

        return assistant_msg

    def _build_chat_prompt(self, messages: Sequence[Message]) -> str:
        """Build a text prompt from a list of messages.

        Args:
            messages: Conversation messages.

        Returns:
            The formatted prompt string.
        """
        parts: List[str] = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"[SYSTEM] {msg.content}")
            elif msg.role == "user":
                parts.append(f"[USER] {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"[ASSISTANT] {msg.content}")
            elif msg.role == "tool":
                parts.append(f"[TOOL_RESULT] {msg.content}")
        parts.append("[ASSISTANT]")
        return "\n".join(parts)

    def _compress_history(self) -> None:
        """Compress conversation history using a sliding window.

        When the history exceeds ``_max_history`` messages, older
        messages are summarised to keep the context within limits.
        """
        if len(self._history) <= self._max_history:
            return

        excess = len(self._history) - self._max_history
        old_messages = self._history[:excess]
        self._history = self._history[excess:]

        summary_parts = [f"[{m.role}] {m.content[:100]}" for m in old_messages]
        summary = Message(
            role="system",
            content="Previous conversation summary: " + " | ".join(summary_parts),
        )
        self._history.insert(0, summary)
        self._logger.debug("Compressed %d old messages.", excess)

    # ------------------------------------------------------------------
    # Function calling
    # ------------------------------------------------------------------
    def _parse_tool_calls(self, text: str) -> List[ToolCall]:
        """Parse tool-call requests from LLM output.

        Looks for JSON blocks matching the pattern::

            ```json
            {"name": "tool_name", "arguments": {...}}
            ```

        Args:
            text: The LLM output text.

        Returns:
            A list of :class:`ToolCall` objects.
        """
        tool_calls: List[ToolCall] = []

        # Pattern 1: fenced JSON blocks.
        json_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
        for match in json_pattern.finditer(text):
            try:
                data = json.loads(match.group(1))
                if "name" in data:
                    tool_calls.append(
                        ToolCall(
                            name=data["name"],
                            arguments=data.get("arguments", data.get("parameters", {})),
                        )
                    )
            except (json.JSONDecodeError, KeyError):
                continue

        return tool_calls

    def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[Message]:
        """Execute a list of tool calls and return tool-result messages.

        Args:
            tool_calls: The tool calls to execute.

        Returns:
            A list of :class:`Message` objects with ``role="tool"``.
        """
        results: List[Message] = []
        for tc in tool_calls:
            result: ToolResult = self._tool_registry.execute_tool(tc.name, tc.arguments)
            content = json.dumps(
                {"output": result.output, "error": result.error}
                if not result.success
                else {"output": result.output},
                default=str,
            )
            results.append(
                Message(
                    role="tool",
                    content=content,
                    tool_call_id=tc.id,
                    name=tc.name,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Structured output (JSON schema constrained decoding)
    # ------------------------------------------------------------------
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Generate text constrained to a JSON schema.

        Uses a post-generation validation and repair approach: generate
        freely, then parse and validate against ``schema``.

        Args:
            prompt: The input prompt.
            schema: A JSON-schema-style dictionary describing the
                expected output structure.
            max_tokens: Maximum tokens to generate.

        Returns:
            A dictionary matching the schema.
        """
        schema_hint = (
            f"\n[Respond with a JSON object matching this schema]\n"
            f"{json.dumps(schema, indent=2)}\n"
        )
        full_prompt = prompt + schema_hint

        text = self.generate(full_prompt, max_tokens=max_tokens, temperature=0.1)

        # Extract JSON from the response.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find a JSON block.
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        self._logger.warning("Failed to parse structured output; returning default.")
        return self._build_default_from_schema(schema)

    @staticmethod
    def _build_default_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Build a default object from a JSON schema."""
        result: Dict[str, Any] = {}
        properties = schema.get("properties", {})
        for key, spec in properties.items():
            prop_type = spec.get("type", "string")
            if prop_type == "string":
                result[key] = spec.get("default", "")
            elif prop_type == "integer":
                result[key] = spec.get("default", 0)
            elif prop_type == "float":
                result[key] = spec.get("default", 0.0)
            elif prop_type == "boolean":
                result[key] = spec.get("default", False)
            elif prop_type == "list":
                result[key] = spec.get("default", [])
            elif prop_type == "dict":
                result[key] = spec.get("default", {})
        return result

    # ------------------------------------------------------------------
    # Speculative decoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_speculative(
        self,
        prompt: str,
        max_tokens: int = 256,
        num_draft_tokens: int = 4,
        temperature: float = 0.7,
    ) -> str:
        """Generate text using speculative decoding.

        A small draft model proposes ``num_draft_tokens`` tokens, which
        are then verified by the large model in a single forward pass.
        Accepted tokens are kept; the first rejected token is replaced by
        the large model's own sample.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            num_draft_tokens: Number of tokens the draft model proposes
                per round.
            temperature: Sampling temperature.

        Returns:
            The generated text.
        """
        if self._draft_model is None:
            self._logger.warning(
                "No draft model available; falling back to standard generation."
            )
            return self.generate(
                prompt, max_tokens=max_tokens, temperature=temperature
            )

        self.model.eval()
        self._draft_model.eval()
        device = self._device

        input_ids = self.tokenizer.encode(prompt, return_tensors=True).to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        generated = input_ids
        strategy = SamplingStrategy(temperature=temperature, top_k=0, top_p=1.0)

        tokens_generated = 0
        while tokens_generated < max_tokens:
            # 1. Draft model proposes tokens.
            draft_tokens = self._draft_model.generate(
                generated,
                max_tokens=min(num_draft_tokens, max_tokens - tokens_generated),
                temperature=temperature,
            )
            draft_new = draft_tokens[0, generated.shape[1]:]

            if draft_new.numel() == 0:
                break

            # 2. Large model verifies.
            verify_input = torch.cat(
                [generated, draft_new.unsqueeze(0)], dim=1
            )
            logits = self.model.forward(verify_input)

            accepted = 0
            for i, draft_id in enumerate(draft_new):
                pos = generated.shape[1] + i
                large_logits = logits[:, pos, :]
                large_token = strategy.apply(large_logits)

                if large_token.item() == draft_id.item():
                    accepted += 1
                else:
                    # Reject: replace with the large model's token.
                    generated = torch.cat(
                        [generated, draft_new[:accepted].unsqueeze(0)], dim=1
                    )
                    generated = torch.cat(
                        [generated, large_token.unsqueeze(0).unsqueeze(0)], dim=1
                    )
                    tokens_generated += accepted + 1
                    break
            else:
                # All draft tokens accepted; sample one more.
                generated = torch.cat([generated, draft_new.unsqueeze(0)], dim=1)
                extra_logits = logits[:, -1, :]
                extra_token = strategy.apply(extra_logits)
                generated = torch.cat(
                    [generated, extra_token.unsqueeze(0).unsqueeze(0)], dim=1
                )
                tokens_generated += accepted + 1

        prompt_len = input_ids.shape[1]
        new_ids = generated[0, prompt_len:].tolist()
        return self.detokenize(new_ids)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def reset_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()

    def register_tool(
        self,
        name: str,
        func: Any,
        description: str = "",
        parameter_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a tool with the engine's tool registry.

        Args:
            name: Tool name.
            func: Callable to execute.
            description: Human-readable description.
            parameter_schema: Parameter schema.
        """
        self._tool_registry.register_tool(name, func, description, parameter_schema)

    def __repr__(self) -> str:
        return f"TextEngine(model={self.model_name!r}, device={self._device})"
