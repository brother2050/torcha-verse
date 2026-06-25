"""AutoDirector v2 -- intelligent canvas generation for TorchaVerse v0.3.0 (L5).

The :class:`AutoDirector` is the "smart" layer above the canvas system.  Given
a natural-language *topic* (e.g. ``"a cat playing piano"``), it:

1. **Selects** the most relevant built-in pipeline template (or, in a full
   deployment, retrieves a community template via vector search).
2. **Fills** the template's placeholder variables (character names, scenes,
   plot, shot durations, prompts, ...) using an LLM callback.  By default a
   rule-based callback is used so that AutoDirector works out of the box
   without any external LLM API.
3. **Expands** the filled template into a fully-laid-out :class:`Canvas`
   object that the user can continue to adjust in the visual editor.
4. Optionally **optimises** the canvas (merging redundant nodes, adjusting
   resource allocation).

The AutoDirector layer is *torch-free* and depends only on the canvas core
(:mod:`canvas.canvas`) and the L5 pipeline layer
(:mod:`pipeline.templates`).

Public surface
--------------

* :class:`AutoDirector` -- the intelligent canvas generator.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from pipeline.templates import TemplateRegistry

from .canvas import Canvas, CanvasConnection, CanvasNode

__all__ = ["AutoDirector"]

# ---------------------------------------------------------------------------
# Module-level logger.
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger("canvas.autodirector")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Input keys whose values should be customised based on the topic.
_FILLABLE_INPUT_KEYS: frozenset[str] = frozenset({
    "prompt",
    "negative_prompt",
    "theme",
    "topic",
    "text",
    "lyrics",
    "character",
    "scene",
    "plot",
    "subject",
    "content",
    "description",
    "story",
    "script",
    "lines",
    "context",
    "narration",
})

#: Input keys that control resource consumption and may be adjusted by
#: :meth:`AutoDirector.optimize`.
_RESOURCE_INPUT_KEYS: frozenset[str] = frozenset({
    "steps",
    "width",
    "height",
    "max_tokens",
    "scale",
})

#: Mapping of node type -> list of input keys that should be customised
#: based on the topic.  When a template node of a given type does not
#: already have one of these inputs (and the input is not wired via a
#: connection), :meth:`AutoDirector.fill_template` adds it so that the
#: LLM callback can fill it.
_NODE_TYPE_FILLABLE_INPUTS: Dict[str, List[str]] = {
    # Image nodes
    "image_txt2img": ["prompt", "negative_prompt"],
    "image_img2img": ["prompt", "negative_prompt"],
    "image_inpaint": ["prompt", "negative_prompt"],
    "image_upscale": ["prompt"],
    "image_relight": ["prompt"],
    # Text nodes
    "text_chat": ["prompt"],
    "text_translate": ["text"],
    "text_summarize": ["text"],
    "text_classify": ["text"],
    "text_extract": ["text"],
    # Audio nodes
    "tts_single": ["text"],
    "tts_multi_speaker": ["text"],
    "asr_transcribe": [],
    "music_generate": ["prompt"],
    "sfx_generate": ["prompt"],
    # Video nodes
    "video_txt2video": ["prompt"],
    "video_img2video": ["prompt"],
    "video_first_frame": [],
    "video_interpolate": [],
    "video_upscale": [],
    # Pipeline / orchestration nodes
    "subtitle_burn": [],
    "subtitle_generate": ["text"],
    "script_generate": ["topic"],
    "character_design": ["description"],
    "storyboard_layout": ["description"],
    "theme_expand": ["theme"],
    "topic_expand": ["topic"],
}

#: Bonus score added when a template's category appears in the topic.
_CATEGORY_MATCH_BONUS: int = 2
#: Bonus score added when a template's tag appears in the topic.
_TAG_MATCH_BONUS: int = 1
#: Factor by which non-critical node steps are reduced during optimisation.
_OPTIMIZE_STEP_REDUCTION: float = 0.8
#: Default canvas name prefix for generated canvases.
_GENERATED_CANVAS_PREFIX: str = "autodirector:"


# ---------------------------------------------------------------------------
# Default LLM callback (rule-based, no external API)
# ---------------------------------------------------------------------------
def _default_llm_callback(prompt: str) -> str:
    """Rule-based LLM callback that generates values without external APIs.

    This function parses a structured prompt (containing a ``Topic:``
    line and a ``Variables:`` line) and returns a JSON object mapping each
    variable name to a topic-derived value.  It is used as the default
    ``llm_callback`` when no real LLM is configured.

    Args:
        prompt: A structured prompt string.

    Returns:
        A JSON string mapping variable names to filled values.
    """
    # Extract the topic.
    topic_match = re.search(r"Topic:\s*(.+)", prompt)
    topic = topic_match.group(1).strip() if topic_match else "unknown topic"

    # Extract the variable names.
    var_match = re.search(r"Variables:\s*(.+)", prompt)
    if var_match:
        raw_vars = var_match.group(1).strip()
        variables = [v.strip() for v in raw_vars.split(",") if v.strip()]
    else:
        # Fallback: look for "- varname:" patterns.
        variables = re.findall(r"-\s*(\w+):", prompt)

    result: Dict[str, str] = {}
    topic_words = topic.split()
    primary_subject = topic_words[0].capitalize() if topic_words else "Subject"

    for var in variables:
        if var == "prompt":
            result[var] = "{}, high quality, detailed, professional".format(
                topic
            )
        elif var == "negative_prompt":
            result[var] = "low quality, blurry, distorted, deformed"
        elif var in ("character",):
            result[var] = primary_subject
        elif var in ("scene",):
            result[var] = "A scene featuring {}".format(topic)
        elif var in ("plot", "story", "script"):
            result[var] = "A creative story about {}.".format(topic)
        elif var in ("theme", "topic", "subject"):
            result[var] = topic
        elif var == "text":
            result[var] = "Once upon a time, {}...".format(topic)
        elif var == "lyrics":
            result[var] = "La la la, {}, la la la.".format(topic)
        elif var == "lines":
            result[var] = "Dialogue inspired by {}.".format(topic)
        elif var == "context":
            result[var] = "Context: {}.".format(topic)
        elif var == "narration":
            result[var] = "Narration about {}.".format(topic)
        elif var == "description":
            result[var] = "A vivid description of {}.".format(topic)
        elif var == "content":
            result[var] = "Content about {}.".format(topic)
        else:
            result[var] = topic

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# AutoDirector
# ---------------------------------------------------------------------------
class AutoDirector:
    """Intelligent canvas generator (AutoDirector v2).

    The AutoDirector turns a natural-language topic into a ready-to-edit
    :class:`Canvas` by selecting a pipeline template, filling its
    placeholder variables via an LLM callback, and expanding the result
    into a visual canvas.

    The ``llm_callback`` is a :data:`typing.Callable[[str], str]` that
    receives a structured prompt and returns a response string.  When
    ``None``, a built-in rule-based callback is used so that AutoDirector
    works without any external LLM API.

    Args:
        template_registry: A :class:`~pipeline.templates.TemplateRegistry`
            providing the built-in (and optionally community) templates.
        llm_callback: Optional callable ``(prompt: str) -> str``.  When
            ``None``, :func:`_default_llm_callback` is used.
    """

    def __init__(
        self,
        template_registry: TemplateRegistry,
        llm_callback: Optional[Callable[[str], str]] = None,
    ) -> None:
        if not isinstance(template_registry, TemplateRegistry):
            raise TypeError(
                "template_registry must be a TemplateRegistry instance."
            )
        self._registry: TemplateRegistry = template_registry
        self._llm_callback: Callable[[str], str] = (
            llm_callback if llm_callback is not None else _default_llm_callback
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def registry(self) -> TemplateRegistry:
        """The template registry used for template selection."""
        return self._registry

    @property
    def llm_callback(self) -> Callable[[str], str]:
        """The LLM callback used for template variable filling."""
        return self._llm_callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        topic: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Canvas:
        """Generate a canvas from a natural-language topic.

        This is the main entry point.  It selects the best-matching
        template, fills its variables via the LLM callback, expands the
        result into a canvas, and optionally optimises it.

        Args:
            topic: A natural-language description of the desired content
                (e.g. ``"a cat playing piano"``).
            constraints: Optional dictionary of parameter overrides
                (e.g. ``{"duration_minutes": 5}``).

        Returns:
            A :class:`Canvas` ready for the user to continue adjusting.
        """
        template_name = self.suggest_template(topic)
        _logger.info(
            "AutoDirector: topic=%r -> template=%r", topic, template_name
        )
        filled = self.fill_template(template_name, topic, constraints)
        canvas = self.expand_to_canvas(filled)
        canvas = self.optimize(canvas)
        return canvas

    def suggest_template(self, topic: str) -> str:
        """Recommend the most matching template for the given topic.

        The recommendation uses a keyword-overlap scoring algorithm: the
        topic is tokenised and each template's name, description, category
        and tags are scored by the number of overlapping tokens.  Category
        and tag matches receive bonus weight.

        Args:
            topic: The natural-language topic.

        Returns:
            The name of the best-matching template.

        Raises:
            ValueError: If the template registry is empty.
        """
        templates = self._registry.list()
        if not templates:
            raise ValueError("Template registry is empty.")

        topic_lower = topic.lower()
        # Use re.findall to tokenise on word boundaries so that
        # punctuation-adjacent words (e.g. "cat,playing") are split
        # correctly instead of being treated as a single token.
        topic_words = set(re.findall(r"\w+", topic_lower))

        best_name: Optional[str] = None
        best_score: int = -1

        for tmpl in templates:
            haystack = " ".join(
                [tmpl.name, tmpl.description, tmpl.category]
                + list(tmpl.tags)
            ).lower()
            haystack_words = set(haystack.split())

            # Word overlap score.
            score: int = len(topic_words & haystack_words)

            # Category bonus.
            if tmpl.category and tmpl.category.lower() in topic_lower:
                score += _CATEGORY_MATCH_BONUS

            # Tag bonuses.
            for tag in tmpl.tags:
                if tag.lower() in topic_lower:
                    score += _TAG_MATCH_BONUS

            if score > best_score:
                best_score = score
                best_name = tmpl.name

        # If nothing matched, default to the first template.
        if best_name is None:
            best_name = templates[0].name

        return best_name

    def fill_template(
        self,
        template_name: str,
        topic: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fill a template's placeholder variables using the LLM callback.

        The method scans the template's DAG for *fillable* input keys from
        two sources:

        1. **Existing fillable inputs** -- node inputs whose key is in
           :data:`_FILLABLE_INPUT_KEYS` (e.g. ``prompt``, ``theme``,
           ``text``).
        2. **Type-based fillable inputs** -- inputs declared in
           :data:`_NODE_TYPE_FILLABLE_INPUTS` for the node's type.  When a
           node of a given type does not already have one of these inputs
           and the input is not wired via a connection, the method adds it
           so that the LLM callback can fill it.

        A structured prompt is built from all collected keys, the LLM
        callback is invoked, and the returned values are applied to the
        template's node inputs.  Constraint overrides are applied on top
        of the LLM-filled values.

        Args:
            template_name: The name of the template to fill.
            topic: The natural-language topic.
            constraints: Optional dictionary of parameter overrides.

        Returns:
            A dictionary with keys:

            * ``template_name`` -- the source template name.
            * ``topic`` -- the topic string.
            * ``constraints`` -- the constraints dict (or ``{}``).
            * ``variables`` -- the LLM-filled variable mapping.
            * ``dag_dict`` -- the template's DAG with filled inputs.

        Raises:
            KeyError: If no template with ``template_name`` is registered.
        """
        template = self._registry.get(template_name)
        # Deep-copy the dag_dict so we don't mutate the original template.
        dag_dict = copy.deepcopy(template.dag_dict)
        constraints = dict(constraints or {})

        # Build a set of (to_node, input_key) pairs that are wired via
        # connections, so we don't add static values for connected inputs.
        connected_inputs: set[tuple[str, str]] = set()
        for edge_d in dag_dict.get("edges", []):
            connected_inputs.add(
                (edge_d.get("to_node", ""), edge_d.get("input_key", ""))
            )

        # Collect fillable input keys from two sources:
        #   1. Existing inputs whose key is in _FILLABLE_INPUT_KEYS.
        #   2. Node-type-based fillable inputs from _NODE_TYPE_FILLABLE_INPUTS.
        fillable_keys: List[str] = []
        seen_keys: set[str] = set()
        for node_d in dag_dict.get("nodes", []):
            node_type = node_d.get("node_type", "")
            inputs = node_d.get("inputs") or {}
            # Source 1: existing fillable inputs.
            for key in inputs:
                if key in _FILLABLE_INPUT_KEYS and key not in seen_keys:
                    fillable_keys.append(key)
                    seen_keys.add(key)
            # Source 2: type-based fillable inputs.
            type_fillable = _NODE_TYPE_FILLABLE_INPUTS.get(node_type, [])
            for key in type_fillable:
                if key not in seen_keys:
                    fillable_keys.append(key)
                    seen_keys.add(key)

        # Build the LLM prompt and call the callback.
        variables: Dict[str, Any] = {}
        if fillable_keys:
            prompt = self._build_fill_prompt(
                template_name, topic, fillable_keys
            )
            response = self._llm_callback(prompt)
            variables = self._parse_llm_response(response, fillable_keys)

        # Apply filled variables to node inputs.
        for node_d in dag_dict.get("nodes", []):
            node_id = node_d.get("id", "")
            node_type = node_d.get("node_type", "")
            inputs = node_d.setdefault("inputs", {})

            # Update existing fillable inputs with LLM values.
            for key in list(inputs.keys()):
                if key in variables:
                    inputs[key] = variables[key]

            # Add type-based fillable inputs that don't exist and aren't
            # connected, so the LLM-filled value becomes a static input.
            type_fillable = _NODE_TYPE_FILLABLE_INPUTS.get(node_type, [])
            for key in type_fillable:
                if (
                    key not in inputs
                    and (node_id, key) not in connected_inputs
                    and key in variables
                ):
                    inputs[key] = variables[key]

        # Apply constraint overrides.
        for node_d in dag_dict.get("nodes", []):
            inputs = node_d.setdefault("inputs", {})
            for ckey, cval in constraints.items():
                if ckey in inputs:
                    inputs[ckey] = cval

        return {
            "template_name": template_name,
            "topic": topic,
            "constraints": constraints,
            "variables": variables,
            "dag_dict": dag_dict,
        }

    def expand_to_canvas(
        self, filled_template: Dict[str, Any]
    ) -> Canvas:
        """Expand a filled template into a :class:`Canvas`.

        Each node in the filled template's ``dag_dict`` becomes a
        :class:`CanvasNode`, and each edge becomes a
        :class:`CanvasConnection`.  The canvas is then auto-laid-out.

        Args:
            filled_template: The dict returned by :meth:`fill_template`
                (or a compatible dict with a ``dag_dict`` key).

        Returns:
            A new :class:`Canvas` with nodes, connections and auto-layout.
        """
        dag_dict = filled_template.get("dag_dict", {})
        template_name = filled_template.get(
            "template_name", "generated"
        )
        canvas_name = _GENERATED_CANVAS_PREFIX + template_name
        canvas = Canvas(canvas_name)

        # Add nodes.
        for node_d in dag_dict.get("nodes", []):
            canvas.add_node(
                node_d.get("node_type", "unknown"),
                id=node_d.get("id"),
                **dict(node_d.get("inputs") or {}),
            )

        # Add connections.
        # ``Canvas.connect`` raises ``ValueError`` on validation failure
        # (e.g. duplicate edge, type mismatch).  We skip such edges with a
        # warning rather than aborting the whole canvas generation, so that
        # a single bad edge in a template does not prevent the rest of the
        # canvas from being built.
        for edge_d in dag_dict.get("edges", []):
            try:
                canvas.connect(
                    edge_d.get("from_node", ""),
                    edge_d.get("output_key", "output"),
                    edge_d.get("to_node", ""),
                    edge_d.get("input_key", "input"),
                )
            except ValueError:
                _logger.warning(
                    "expand_to_canvas: skipped edge %s.%s -> %s.%s "
                    "(validation failed)",
                    edge_d.get("from_node", ""),
                    edge_d.get("output_key", "output"),
                    edge_d.get("to_node", ""),
                    edge_d.get("input_key", "input"),
                )

        # Auto-layout for a clean visual result.
        canvas.auto_layout()
        return canvas

    def optimize(self, canvas: Canvas) -> Canvas:
        """Optimise a canvas by merging redundant nodes and adjusting resources.

        Two optimisations are applied:

        1. **Merge redundant nodes** -- nodes with the same ``type`` and
           identical ``inputs`` are merged into one.  Connections from the
           removed nodes are rewired to the surviving node, and duplicate
           connections are de-duplicated.
        2. **Adjust resource allocation** -- for non-critical nodes
           (those with no downstream dependents) that have a ``steps``
           input, the step count is reduced to save compute.

        The original canvas is not modified; a new optimised canvas is
        returned.

        Args:
            canvas: The canvas to optimise.

        Returns:
            A new :class:`Canvas` with optimisations applied.
        """
        # Work on a fork so the original is untouched.
        optimized = canvas.fork(canvas.name + ":optimized")

        # --- 1. Merge redundant nodes ---
        self._merge_redundant_nodes(optimized)

        # --- 2. Adjust resource allocation ---
        self._adjust_resources(optimized)

        return optimized

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_fill_prompt(
        template_name: str,
        topic: str,
        fillable_keys: List[str],
    ) -> str:
        """Build the structured prompt sent to the LLM callback.

        Args:
            template_name: The name of the template being filled.
            topic: The natural-language topic.
            fillable_keys: The list of variable names to fill.

        Returns:
            A structured prompt string.
        """
        vars_str = ", ".join(fillable_keys)
        return (
            "You are a creative director. Fill in the following template "
            "variables for the given topic.\n"
            "\n"
            "Topic: {}\n"
            "Template: {}\n"
            "Variables: {}\n"
            "\n"
            "Respond with a JSON object mapping each variable name to its "
            "filled value."
        ).format(topic, template_name, vars_str)

    @staticmethod
    def _parse_llm_response(
        response: str,
        fillable_keys: List[str],
    ) -> Dict[str, Any]:
        """Parse the LLM callback's response into a variable mapping.

        Attempts to parse ``response`` as a JSON object.  If parsing fails,
        falls back to mapping every fillable key to the raw response string.

        Args:
            response: The raw response string from the LLM callback.
            fillable_keys: The expected variable names.

        Returns:
            A dictionary mapping variable names to filled values.
        """
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                # Only keep keys that are in the fillable set.
                return {
                    k: v for k, v in parsed.items() if k in fillable_keys
                }
        except (json.JSONDecodeError, TypeError):
            _logger.debug(
                "LLM response was not valid JSON; using fallback."
            )

        # Fallback: only assign the response to "prompt"; other keys
        # get an empty string so they don't receive irrelevant content.
        return {k: (response if k == "prompt" else "") for k in fillable_keys}

    @staticmethod
    def _merge_redundant_nodes(canvas: Canvas) -> None:
        """Merge nodes with the same type and identical inputs.

        For each group of redundant nodes, the first node is kept and the
        rest are removed.  Connections referencing the removed nodes are
        rewired to the surviving node, and duplicate connections are
        de-duplicated.

        Args:
            canvas: The canvas to optimise in place.
        """
        nodes = canvas.list_nodes()
        # Group nodes by (type, frozenset of inputs).
        groups: Dict[tuple, List[str]] = {}
        for node in nodes:
            key = (node.type, repr(sorted(node.inputs.items()) if isinstance(node.inputs, dict) else node.inputs))
            groups.setdefault(key, []).append(node.id)

        for key, node_ids in groups.items():
            if len(node_ids) <= 1:
                continue
            survivor = node_ids[0]
            redundant = node_ids[1:]
            redundant_set = set(redundant)
            survivor_map = {nid: survivor for nid in redundant}

            # Rewire connections.
            connections = canvas.list_connections()
            for conn in connections:
                new_from = conn.from_node
                new_to = conn.to_node
                if conn.from_node in redundant_set:
                    new_from = survivor_map[conn.from_node]
                if conn.to_node in redundant_set:
                    new_to = survivor_map[conn.to_node]
                # If neither endpoint changed, leave the connection as-is.
                if (
                    new_from == conn.from_node
                    and new_to == conn.to_node
                ):
                    continue
                # Disconnect the old connection before rewiring.
                canvas.disconnect(conn.id)
                # If both endpoints map to the survivor, it's a self-loop -- skip.
                if new_from == new_to:
                    continue
                try:
                    canvas.connect(
                        new_from,
                        conn.from_port,
                        new_to,
                        conn.to_port,
                    )
                except ValueError:
                    # duplicate after rewiring
                    _logger.debug(
                        "Merge: skipped duplicate connection "
                        "%s.%s -> %s.%s after rewiring.",
                        new_from,
                        conn.from_port,
                        new_to,
                        conn.to_port,
                    )

            # Remove redundant nodes.
            for nid in redundant:
                canvas.remove_node(nid)

    @staticmethod
    def _adjust_resources(canvas: Canvas) -> None:
        """Adjust resource allocation for non-critical nodes.

        Non-critical nodes (those with no downstream dependents) that have
        a ``steps`` input get their step count reduced to save compute.

        Args:
            canvas: The canvas to optimise in place.
        """
        # Determine which nodes have dependents.
        has_dependents: set[str] = set()
        for conn in canvas.list_connections():
            has_dependents.add(conn.from_node)

        for node in canvas.list_nodes():
            if node.id in has_dependents:
                continue  # critical node -- skip
            inputs = node.inputs
            if "steps" in inputs and isinstance(inputs["steps"], (int, float)):
                original = inputs["steps"]
                if isinstance(original, int):
                    reduced = max(1, int(original * _OPTIMIZE_STEP_REDUCTION))
                else:
                    reduced = round(original * _OPTIMIZE_STEP_REDUCTION, 1)
                # ``list_nodes`` returns deep copies, so mutating
                # ``inputs`` would not persist.  Write back via the
                # canvas API instead.
                canvas.update_node_inputs(node.id, {"steps": reduced})

    def __repr__(self) -> str:
        return "AutoDirector(templates={})".format(self._registry.count())
