"""Gradio-based visual interface for TorchaVerse.

This module builds a multi-tab Gradio Blocks interface that exposes all
framework capabilities through an interactive web UI:

* **Multimodal Chat** -- a unified chat that accepts text, image, and
  audio inputs and returns text or image outputs.
* **Image Studio** -- a text-to-image / image-to-image canvas with
  parameter controls (width, height, steps, guidance, seed).
* **Video Studio** -- text-to-video generation with frame and FPS
  controls.
* **RAG Manager** -- upload documents, ingest them, and query the
  knowledge base.
* **Agent Playground** -- run single or multi-agent flows and visualise
  the reasoning chain.
* **Workflow Orchestrator** -- a simplified node-based editor (similar
  to ComfyUI) for chaining generation steps.

The interface reuses :class:`PipelineService` from the API server so
that both surfaces share the same Pipeline/Node back-end (no HTTP
round-trip required for local use).
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional, Tuple

from infrastructure.config_center import ConfigCenter
from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

# Reuse the PipelineService from the service layer so the Web UI and the
# REST API share the same Pipeline/Node back-end.
from serving.service import PipelineService

try:
    import gradio as gr
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "gradio is required for the Web UI. "
        "Install it with: pip install gradio"
    ) from _exc

__all__ = ["create_interface", "launch", "WebUI"]

logger = get_logger("web_ui")


# ===========================================================================
# Pipeline service holder (lazy singleton)
# ===========================================================================
_service: Optional[PipelineService] = None


def _get_service() -> PipelineService:
    """Return a lazily-created :class:`PipelineService` singleton."""
    global _service
    if _service is None:
        _service = PipelineService()
    return _service


def reset_service() -> None:
    """Reset the cached service (useful for testing)."""
    global _service
    _service = None


# ===========================================================================
# Conversion helpers
# ===========================================================================
def _audio_to_numpy(audio: Any) -> Tuple[Any, int]:
    """Convert an audio object to a ``(numpy_array, sample_rate)`` tuple.

    Accepts either a real audio object exposing ``numpy`` / ``waveform``
    / ``sample_rate`` attributes or a placeholder dict returned by the
    node system (in which case an empty waveform is returned).
    """
    import numpy as np

    waveform = getattr(audio, "numpy", None)
    if waveform is None:
        waveform = getattr(audio, "waveform", None)
    if waveform is None:
        return np.zeros(1024, dtype="float32"), 22050
    waveform = np.asarray(waveform)
    if waveform.ndim == 2:
        waveform = waveform[0]
    sample_rate = getattr(audio, "sample_rate", 22050)
    return waveform, int(sample_rate)


def _video_to_frames(video: Any) -> List[Any]:
    """Convert a video object to a list of PIL images.

    Accepts either a real video object exposing ``frames`` / ``fps``
    attributes (a tensor or array of shape ``[T, C, H, W]`` or
    ``[T, H, W, C]``) or a placeholder dict returned by the node system
    (in which case an empty list is returned).
    """
    from PIL import Image as PILImage
    import numpy as np

    frames = getattr(video, "frames", None)
    if frames is None:
        return []
    frames_np = np.asarray(frames)
    if frames_np.ndim == 5:
        frames_np = frames_np[0]
    # Normalise to [T, H, W, C] uint8.
    if frames_np.ndim == 4 and frames_np.shape[-1] not in (1, 3, 4):
        frames_np = np.transpose(frames_np, (0, 2, 3, 1))
    frames_np = (np.clip(frames_np, 0, 1) * 255).astype("uint8") \
        if frames_np.dtype.kind == "f" else frames_np.astype("uint8")
    return [PILImage.fromarray(f) for f in frames_np]


def _to_pil_image(image: Any) -> Any:
    """Return a PIL image for display, or ``None`` for placeholder data."""
    from PIL import Image as PILImage

    if isinstance(image, PILImage.Image):
        return image
    return None


# ===========================================================================
# Tab: Multimodal Chat
# ===========================================================================
def _multimodal_chat(
    message: str,
    image: Optional[Any],
    audio: Optional[Tuple[int, Any]],
    history: List[List[str]],
    model: str,
    max_tokens: int,
) -> Tuple[List[List[str]], str]:
    """Handle a multimodal chat turn.

    Args:
        message: The user's text message.
        image: Optional uploaded image (PIL or None).
        audio: Optional uploaded audio ``(sample_rate, numpy_array)`` tuple.
        history: Chat history as a list of ``[user, assistant]`` pairs.
        model: Model name.
        max_tokens: Maximum tokens for the response.

    Returns:
        A tuple ``(updated_history, cleared_input)``.
    """
    service = _get_service()
    question = (message or "").strip() or "Describe the input in detail."

    try:
        if image is not None:
            result = service._run(
                "multimodal_chat",
                "image_understand",
                "img",
                {
                    "image": image,
                    "question": question,
                    "max_new_tokens": int(max_tokens or 128),
                },
                config={"default_multimodal_model": model},
            )
        else:
            result = service._run(
                "multimodal_chat",
                "text_chat",
                "chat",
                {
                    "prompt": question,
                    "model": model or "default",
                    "max_tokens": int(max_tokens or 256),
                },
                config={"default_text_model": model or "default"},
            )
        if "error" in result:
            response = f"[Error] {result['error']}"
        else:
            response = str(result.get("text", ""))
    except Exception as exc:
        response = f"[Error] {exc}"

    history.append([message, response])
    return history, ""


def _build_multimodal_chat_tab() -> None:
    """Build the multimodal chat tab."""
    gr.Markdown("## Multimodal Unified Chat\nSend text, images, or audio and get a text response.")

    with gr.Row():
        chatbot = gr.Chatbot(label="Conversation", height=450)
        with gr.Column(scale=1):
            image_input = gr.Image(label="Image Input", type="pil")
            audio_input = gr.Audio(label="Audio Input", sources=["upload", "microphone"])

    with gr.Row():
        msg_input = gr.Textbox(
            label="Message",
            placeholder="Type your message or question...",
            scale=4,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear", scale=1)

    with gr.Row():
        model_input = gr.Textbox(label="Model", value="default", scale=1)
        max_tokens_slider = gr.Slider(
            32, 2048, value=256, step=32, label="Max Tokens", scale=1
        )

    state = gr.State([])

    send_btn.click(
        _multimodal_chat,
        inputs=[msg_input, image_input, audio_input, state, model_input, max_tokens_slider],
        outputs=[chatbot, msg_input],
    )
    msg_input.submit(
        _multimodal_chat,
        inputs=[msg_input, image_input, audio_input, state, model_input, max_tokens_slider],
        outputs=[chatbot, msg_input],
    )
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])


# ===========================================================================
# Tab: Image Studio
# ===========================================================================
def _generate_image(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    model: str,
) -> Any:
    """Generate an image and return it for display."""
    service = _get_service()
    seed_val = int(seed) if seed >= 0 else None
    try:
        result = service.image_txt2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            steps=int(steps),
            guidance_scale=float(guidance),
            seed=seed_val,
            model=model,
        )
        if "error" in result:
            raise gr.Error(f"Image generation failed: {result['error']}")
        image = result.get("image", result)
        pil = _to_pil_image(image)
        if pil is None:
            raise gr.Error(
                "Image node returned placeholder data (no real backend)."
            )
        return pil
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Image generation failed: {exc}")


def _build_image_studio_tab() -> None:
    """Build the image generation studio tab."""
    gr.Markdown("## Image Studio\nGenerate images from text prompts with fine-grained control.")

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(label="Prompt", lines=3, placeholder="A beautiful sunset over mountains...")
            neg_prompt_input = gr.Textbox(label="Negative Prompt", lines=2, placeholder="blurry, low quality...")
            model_input = gr.Textbox(label="Model", value="default")

            with gr.Row():
                width_slider = gr.Slider(256, 1024, value=512, step=64, label="Width")
                height_slider = gr.Slider(256, 1024, value=512, step=64, label="Height")

            with gr.Row():
                steps_slider = gr.Slider(1, 100, value=30, step=1, label="Steps")
                guidance_slider = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")

            seed_input = gr.Number(label="Seed (-1 = random)", value=-1)
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", height=512)

    generate_btn.click(
        _generate_image,
        inputs=[
            prompt_input, neg_prompt_input, width_slider, height_slider,
            steps_slider, guidance_slider, seed_input, model_input,
        ],
        outputs=[output_image],
    )


# ===========================================================================
# Tab: Video Studio
# ===========================================================================
def _generate_video(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    steps: int,
    guidance: float,
    seed: int,
    model: str,
) -> Any:
    """Generate a video and return a gallery of frames."""
    service = _get_service()
    seed_val = int(seed) if seed >= 0 else None
    try:
        result = service.video_txt2vid(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            fps=int(fps),
            steps=int(steps),
            guidance_scale=float(guidance),
            seed=seed_val,
            model=model,
        )
        if "error" in result:
            raise gr.Error(f"Video generation failed: {result['error']}")
        video = result.get("video", result)
        frames = _video_to_frames(video)
        if not frames:
            raise gr.Error(
                "Video node returned placeholder data (no real backend)."
            )
        return frames
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Video generation failed: {exc}")


def _build_video_studio_tab() -> None:
    """Build the video generation studio tab."""
    gr.Markdown("## Video Studio\nGenerate videos from text prompts. Frames are shown as a gallery.")

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(label="Prompt", lines=3, placeholder="A cat playing with a ball of yarn...")
            neg_prompt_input = gr.Textbox(label="Negative Prompt", lines=2)
            model_input = gr.Textbox(label="Model", value="default")

            with gr.Row():
                width_slider = gr.Slider(256, 768, value=512, step=64, label="Width")
                height_slider = gr.Slider(256, 768, value=512, step=64, label="Height")

            with gr.Row():
                frames_slider = gr.Slider(4, 64, value=16, step=4, label="Num Frames")
                fps_slider = gr.Slider(1, 30, value=8, step=1, label="FPS")

            with gr.Row():
                steps_slider = gr.Slider(1, 100, value=30, step=1, label="Steps")
                guidance_slider = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance")

            seed_input = gr.Number(label="Seed (-1 = random)", value=-1)
            generate_btn = gr.Button("Generate Video", variant="primary")

        with gr.Column(scale=1):
            output_gallery = gr.Gallery(label="Generated Frames", height=512, columns=4)

    generate_btn.click(
        _generate_video,
        inputs=[
            prompt_input, neg_prompt_input, width_slider, height_slider,
            frames_slider, fps_slider, steps_slider, guidance_slider,
            seed_input, model_input,
        ],
        outputs=[output_gallery],
    )


# ===========================================================================
# Tab: RAG Manager
# ===========================================================================
def _rag_ingest(
    files: List[Any],
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """Ingest uploaded files into the RAG index.

    Each uploaded file is read as text (best-effort UTF-8, falling
    back to latin-1) and forwarded to the ``rag_ingest`` L4 node
    against the default ``"webui"`` index.  The returned status
    message reports how many documents and chunks were embedded.
    """
    service = _get_service()
    if not files:
        return "[Info] No files uploaded."
    documents: List[Dict[str, Any]] = []
    for i, f in enumerate(files):
        path = getattr(f, "name", None) or (f if isinstance(f, str) else None)
        if not path:
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except Exception as exc:  # noqa: BLE001
            try:
                with open(path, "r", encoding="latin-1", errors="replace") as fh:
                    text = fh.read()
            except Exception as exc2:  # noqa: BLE001
                logger.warning("Failed to read %s: %s / %s", path, exc, exc2)
                continue
        if not text.strip():
            continue
        doc_id = f"webui_{i:04d}_{int(time.time())}"
        documents.append({"doc_id": doc_id, "text": text})

    if not documents:
        return "[Info] No readable text content in the uploaded files."

    result = service._run(
        "rag_ingest",
        "rag_ingest",
        "ingest",
        {
            "index_name": "webui",
            "documents": documents,
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
        },
    )
    if "error" in result:
        return f"[Error] {result['error']}"
    docs = int(result.get("documents", 0))
    vecs = int(result.get("vectors", 0))
    return f"[OK] Ingested {docs} documents -> {vecs} vectors into index 'webui'."


def _rag_query(question: str, top_k: int, rerank: bool) -> Tuple[str, str]:
    """Query the RAG engine.

    Backed by the ``rag_query`` L4 node (embedding + top-k retrieval)
    followed by a ``text_chat`` L4 node for answer synthesis.  The
    second tuple element renders the hit list as a Markdown list of
    source citations.
    """
    service = _get_service()
    question = (question or "").strip()
    if not question:
        return "[Error] Question is required.", ""

    retrieval = service._run(
        "rag_query",
        "rag_query",
        "retrieval",
        {"index_name": "webui", "query": question, "top_k": int(top_k)},
    )
    if "error" in retrieval:
        return f"[Error] {retrieval['error']}", ""
    hits = retrieval.get("hits", [])
    context = retrieval.get("context", "")

    if not context:
        return (
            "(no relevant context found in the 'webui' index -- "
            "upload and ingest some documents first)",
            "",
        )

    user_prompt = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    answer = service._run(
        "rag_query_synth",
        "text_chat",
        "answer",
        {
            "prompt": user_prompt,
            "model": "default",
            "max_tokens": 256,
        },
    )
    if "error" in answer:
        return f"[Error] {answer['error']}", ""
    text = str(answer.get("text", ""))

    sources_lines: List[str] = ["**Sources:**"]
    for i, h in enumerate(hits[: int(top_k)], start=1):
        score = float(h.get("score", 0.0))
        doc_id = h.get("doc_id", "?")
        chunk_id = h.get("chunk_id", "?")
        sources_lines.append(
            f"{i}. `{doc_id}` (chunk `{chunk_id}`, score={score:.3f})"
        )
    sources_md = "\n".join(sources_lines) if hits else "(no sources)"
    return text, sources_md


def _rag_clear() -> str:
    """Clear the RAG index.

    Backed by the ``rag_delete`` L4 node with ``drop_index=True``
    against the default ``"webui"`` index.
    """
    service = _get_service()
    result = service._run(
        "rag_clear",
        "rag_delete",
        "drop",
        {"index_name": "webui", "drop_index": True},
    )
    if "error" in result:
        return f"[Error] {result['error']}"
    if result.get("dropped"):
        return "[OK] RAG index 'webui' was dropped."
    return "[Info] RAG index 'webui' did not exist (nothing to clear)."


def _build_rag_tab() -> None:
    """Build the RAG document management tab."""
    gr.Markdown("## RAG Document Manager\nUpload documents, build a knowledge base, and query it.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Document Ingestion")
            file_upload = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".txt", ".md", ".text"],
            )
            with gr.Row():
                chunk_size_slider = gr.Slider(128, 2048, value=512, step=64, label="Chunk Size")
                chunk_overlap_slider = gr.Slider(0, 256, value=64, step=16, label="Chunk Overlap")
            ingest_btn = gr.Button("Ingest Documents", variant="primary")
            clear_btn = gr.Button("Clear Index", variant="stop")
            ingest_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### Query")
            question_input = gr.Textbox(label="Question", lines=3, placeholder="What would you like to know?")
            with gr.Row():
                top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Top-K")
                rerank_checkbox = gr.Checkbox(label="Rerank", value=False)
            query_btn = gr.Button("Ask", variant="primary")
            answer_output = gr.Textbox(label="Answer", lines=8, interactive=False)
            sources_output = gr.Markdown(label="Sources")

    ingest_btn.click(
        _rag_ingest,
        inputs=[file_upload, chunk_size_slider, chunk_overlap_slider],
        outputs=[ingest_status],
    )
    clear_btn.click(_rag_clear, outputs=[ingest_status])
    query_btn.click(
        _rag_query,
        inputs=[question_input, top_k_slider, rerank_checkbox],
        outputs=[answer_output, sources_output],
    )


# ===========================================================================
# Tab: Agent Playground
# ===========================================================================
def _agent_run(
    task: str,
    flow: str,
    max_steps: int,
    stream: bool,
) -> Tuple[str, str]:
    """Run an agent and return the output and reasoning trace.

    Backed by the ``agent_run`` L4 node (ReAct loop with
    tool-calling).  The first returned value is the agent's final
    answer, the second is a Markdown rendering of the per-step
    ``thought / action / observation`` transcript.
    """
    service = _get_service()
    task = (task or "").strip()
    if not task:
        return "[Error] Task is required.", ""
    max_steps_i = int(max_steps)
    result = service._run(
        "agent_run",
        "agent_run",
        "agent",
        {"query": task, "max_steps": max_steps_i, "temperature": 0.0},
    )
    if "error" in result:
        output = f"[Error] {result['error']}"
        trace_text = ""
    else:
        output = str(result.get("final_answer", ""))
        if not result.get("ok", False):
            output = (
                f"{output}\n\n[Warning] agent did not converge "
                f"in {int(result.get('iterations', 0))} step(s)."
            )
        steps = result.get("steps", [])
        trace_lines: List[str] = []
        for i, s in enumerate(steps, start=1):
            trace_lines.append(f"**Step {i}**")
            if s.get("thought"):
                trace_lines.append(f"- **Thought:** {s['thought']}")
            if s.get("action"):
                trace_lines.append(f"- **Action:** {s['action']}")
            if s.get("observation") is not None:
                trace_lines.append(
                    f"- **Observation:** {s['observation']}"
                )
        trace_text = "\n".join(trace_lines) if trace_lines else "(no steps)"
    return output, trace_text


def _format_step(step: Any) -> str:
    """Format a single agent step for display."""
    lines = [f"**Step {step.step_number}**"]
    if step.thought:
        lines.append(f"**Thought:** {step.thought}")
    if step.action:
        lines.append(f"**Action:** {step.action}")
    if step.action_input:
        lines.append(f"**Input:** {step.action_input}")
    if step.observation:
        lines.append(f"**Observation:** {step.observation}")
    return "\n".join(lines)


def _build_agent_tab() -> None:
    """Build the agent playground tab."""
    gr.Markdown("## Agent Playground\nRun autonomous agents and visualise their reasoning chain.")

    with gr.Row():
        with gr.Column(scale=1):
            task_input = gr.Textbox(
                label="Task",
                lines=4,
                placeholder="Describe a task for the agent to accomplish...",
            )
            flow_select = gr.Dropdown(
                choices=["none", "sequential", "parallel", "hierarchical", "debate"],
                value="none",
                label="Flow Topology",
            )
            with gr.Row():
                max_steps_slider = gr.Slider(1, 50, value=10, step=1, label="Max Steps")
                stream_checkbox = gr.Checkbox(label="Stream Steps", value=True)
            run_btn = gr.Button("Run Agent", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Final Output", lines=6, interactive=False)
            trace_md = gr.Markdown(label="Reasoning Trace")

    run_btn.click(
        _agent_run,
        inputs=[task_input, flow_select, max_steps_slider, stream_checkbox],
        outputs=[output_text, trace_md],
    )


# ===========================================================================
# Tab: Workflow Orchestrator (simplified ComfyUI-style)
# ===========================================================================
def _execute_workflow(nodes_json: str) -> str:
    """Execute a workflow defined as a JSON node graph.

    Each node has ``id``, ``type``, ``params``, and ``inputs`` (list of
    upstream node ids).  Supported node types: ``text_prompt``,
    ``text_generate``, ``image_generate``, ``audio_synthesize``,
    ``merge``, ``output``.

    Args:
        nodes_json: A JSON string describing the workflow nodes.

    Returns:
        A textual summary of the execution results.
    """
    import json as _json

    try:
        nodes = _json.loads(nodes_json)
    except _json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}"

    if not isinstance(nodes, list):
        return "Workflow must be a JSON list of nodes."

    # Build a lookup and topologically sort.
    node_map: Dict[str, Dict[str, Any]] = {n["id"]: n for n in nodes}
    results: Dict[str, Any] = {}
    visited: set = set()

    def _resolve(node_id: str) -> Any:
        """Recursively resolve a node's output."""
        if node_id in results:
            return results[node_id]
        if node_id in visited:
            return f"[cycle detected at {node_id}]"
        visited.add(node_id)

        node = node_map.get(node_id)
        if node is None:
            return f"[unknown node {node_id}]"

        # Resolve inputs.
        input_values = [_resolve(i) for i in node.get("inputs", [])]
        params = node.get("params", {})
        ntype = node.get("type", "")

        output: Any = ""

        if ntype == "text_prompt":
            output = params.get("text", "")

        elif ntype == "text_generate":
            svc = _get_service()
            prompt = input_values[0] if input_values else params.get("prompt", "")
            res = svc.text_completion(
                prompt=str(prompt),
                model=params.get("model", "default"),
                max_tokens=int(params.get("max_tokens", 128)),
            )
            output = res.get("text", str(res)) if "error" not in res else f"[error] {res['error']}"

        elif ntype == "image_generate":
            svc = _get_service()
            prompt = input_values[0] if input_values else params.get("prompt", "")
            res = svc.image_txt2img(
                prompt=str(prompt),
                width=int(params.get("width", 512)),
                height=int(params.get("height", 512)),
                steps=int(params.get("steps", 20)),
                model=params.get("model", "default"),
            )
            output = "[image generated]" if "error" not in res else f"[error] {res['error']}"

        elif ntype == "audio_synthesize":
            svc = _get_service()
            text = input_values[0] if input_values else params.get("text", "")
            res = svc.audio_tts(text=str(text))
            output = "[audio generated]" if "error" not in res else f"[error] {res['error']}"

        elif ntype == "merge":
            output = "\n---\n".join(str(v) for v in input_values)

        elif ntype == "output":
            output = input_values[0] if input_values else ""

        else:
            output = f"[unknown node type: {ntype}]"

        results[node_id] = output
        return output

    # Execute all output nodes.
    summaries: List[str] = []
    for node in nodes:
        if node.get("type") == "output":
            val = _resolve(node["id"])
            summaries.append(f"**Output ({node['id']}):**\n{val}")

    if not summaries:
        summaries.append("No output nodes found in the workflow.")

    return "\n\n".join(summaries)


_DEFAULT_WORKFLOW = """[
  {
    "id": "n1",
    "type": "text_prompt",
    "params": {"text": "Write a short poem about the ocean."},
    "inputs": []
  },
  {
    "id": "n2",
    "type": "text_generate",
    "params": {"model": "default", "max_tokens": 128},
    "inputs": ["n1"]
  },
  {
    "id": "n3",
    "type": "output",
    "params": {},
    "inputs": ["n2"]
  }
]"""


def _build_workflow_tab() -> None:
    """Build the workflow orchestrator tab."""
    gr.Markdown(
        "## Workflow Orchestrator\n"
        "Define a node graph as JSON and execute it. "
        "This is a simplified, text-based version of a ComfyUI-style editor.\n\n"
        "**Node types:** `text_prompt`, `text_generate`, `image_generate`, "
        "`audio_synthesize`, `merge`, `output`."
    )

    with gr.Row():
        with gr.Column(scale=1):
            nodes_editor = gr.Code(
                label="Workflow JSON",
                value=_DEFAULT_WORKFLOW,
                language="json",
                lines=20,
            )
            run_btn = gr.Button("Execute Workflow", variant="primary")

        with gr.Column(scale=1):
            output_md = gr.Markdown(label="Execution Results")

    run_btn.click(_execute_workflow, inputs=[nodes_editor], outputs=[output_md])


# ===========================================================================
# Main interface builder
# ===========================================================================
class WebUI:
    """Container for the Gradio Blocks interface.

    Attributes:
        demo: The underlying :class:`gradio.Blocks` instance.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.demo: gr.Blocks = create_interface(**kwargs)

    def launch(self, **kwargs: Any) -> Any:
        """Launch the Gradio interface.

        Forwards all keyword arguments to :meth:`gradio.Blocks.launch`.
        """
        # Pass stored theme to launch (Gradio 6.0 moved theme here).
        theme = getattr(self.demo, "_torcha_theme", None)
        if theme is not None and "theme" not in kwargs:
            kwargs["theme"] = theme
        return self.demo.launch(**kwargs)


def create_interface(
    title: str = "TorchaVerse",
    theme: Optional[str] = None,
) -> gr.Blocks:
    """Create the full Gradio Blocks interface.

    Args:
        title: The title displayed in the header.
        theme: Optional Gradio theme name (e.g. ``"soft"``).

    Returns:
        A configured :class:`gradio.Blocks` instance with all tabs.
    """
    selected_theme: Any = gr.themes.Soft()
    if theme == "default":
        selected_theme = gr.themes.Default()
    elif theme == "monochrome":
        selected_theme = gr.themes.Monochrome()

    custom_css = """
    .gradio-container { max-width: 1200px !important; }
    """

    with gr.Blocks(title=title, css=custom_css) as demo:
        # Store theme for launch() (Gradio 6.0 moved theme to launch).
        demo._torcha_theme = selected_theme  # type: ignore[attr-defined]
        gr.Markdown(
            f"# {title}\n"
            "A pure PyTorch all-modal generative AI framework. "
            "Select a tab below to begin."
        )

        with gr.Tabs():
            with gr.TabItem("Multimodal Chat"):
                _build_multimodal_chat_tab()

            with gr.TabItem("Image Studio"):
                _build_image_studio_tab()

            with gr.TabItem("Video Studio"):
                _build_video_studio_tab()

            with gr.TabItem("RAG Manager"):
                _build_rag_tab()

            with gr.TabItem("Agent Playground"):
                _build_agent_tab()

            with gr.TabItem("Workflow Orchestrator"):
                _build_workflow_tab()

        # Footer.
        gr.Markdown(
            "---\n"
            f"*TorchaVerse v0.3.1* | "
            f"Device: {DeviceManager().get_device()} | "
            "[Documentation](https://github.com/torcha-verse)"
        )

    return demo


def launch(
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    **kwargs: Any,
) -> Any:
    """Create and launch the TorchaVerse Web UI.

    Args:
        host: The host to bind.
        port: The port to bind.
        share: Whether to create a public share link.
        **kwargs: Additional arguments forwarded to
            :meth:`gradio.Blocks.launch`.

    Returns:
        The result of :meth:`gradio.Blocks.launch`.
    """
    logger.info("Launching TorchaVerse Web UI on %s:%d", host, port)
    demo = create_interface()
    # Pass stored theme to launch (Gradio 6.0 moved theme here).
    theme = getattr(demo, "_torcha_theme", None)
    if theme is not None and "theme" not in kwargs:
        kwargs["theme"] = theme
    return demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        **kwargs,
    )


if __name__ == "__main__":
    launch()
