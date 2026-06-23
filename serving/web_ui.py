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

The interface calls the underlying engines directly (no HTTP round-trip
required), keeping latency low for local use.
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from engines.agent_engine import AgentEngine
from engines.audio_engine import AudioEngine, AudioTensor
from engines.image_engine import ImageEngine
from engines.multimodal_engine import MultiModalEngine
from engines.rag_engine import RAGEngine
from engines.text_engine import Message, TextEngine
from engines.video_engine import VideoEngine, VideoTensor
from infrastructure.config_manager import ConfigManager
from infrastructure.device_manager import DeviceManager
from infrastructure.logger import get_logger

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
# Engine holder (lazy singletons)
# ===========================================================================
class _EngineHolder:
    """Lazy singleton container for all engines used by the Web UI."""

    _text: Optional[TextEngine] = None
    _image: Optional[ImageEngine] = None
    _audio: Optional[AudioEngine] = None
    _video: Optional[VideoEngine] = None
    _multimodal: Optional[MultiModalEngine] = None
    _rag: Optional[RAGEngine] = None
    _agent: Optional[AgentEngine] = None

    @classmethod
    def text(cls, model: str = "default") -> TextEngine:
        if cls._text is None:
            cls._text = TextEngine(model)
        return cls._text

    @classmethod
    def image(cls, model: str = "default") -> ImageEngine:
        if cls._image is None:
            cls._image = ImageEngine(model)
        return cls._image

    @classmethod
    def audio(cls) -> AudioEngine:
        if cls._audio is None:
            cls._audio = AudioEngine()
        return cls._audio

    @classmethod
    def video(cls, model: str = "default") -> VideoEngine:
        if cls._video is None:
            cls._video = VideoEngine(model)
        return cls._video

    @classmethod
    def multimodal(cls) -> MultiModalEngine:
        if cls._multimodal is None:
            cls._multimodal = MultiModalEngine()
        return cls._multimodal

    @classmethod
    def rag(cls) -> RAGEngine:
        if cls._rag is None:
            cls._rag = RAGEngine()
        return cls._rag

    @classmethod
    def agent(cls) -> AgentEngine:
        if cls._agent is None:
            cls._agent = AgentEngine()
        return cls._agent

    @classmethod
    def reset(cls) -> None:
        """Reset all cached engines."""
        cls._text = None
        cls._image = None
        cls._audio = None
        cls._video = None
        cls._multimodal = None
        cls._rag = None
        cls._agent = None


# ===========================================================================
# Conversion helpers
# ===========================================================================
def _audio_tensor_to_numpy(audio: AudioTensor) -> Tuple[Any, int]:
    """Convert an :class:`AudioTensor` to a ``(numpy_array, sample_rate)`` tuple."""
    import numpy as np

    waveform = audio.numpy()
    if waveform.ndim == 2:
        waveform = waveform[0]
    return waveform, audio.sample_rate


def _video_tensor_to_frames(video: VideoTensor) -> List[Any]:
    """Convert a :class:`VideoTensor` to a list of PIL images."""
    from PIL import Image as PILImage
    import numpy as np

    frames = video.frames
    if frames.dim() == 5:
        frames = frames[0]
    frames_np = (frames.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(
        "uint8"
    )
    return [PILImage.fromarray(f) for f in frames_np]


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
    engine = _EngineHolder.multimodal()

    audio_tensor = None
    if audio is not None:
        import numpy as np

        sr, data = audio
        waveform = torch.from_numpy(data).float()
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        audio_tensor = AudioTensor(waveform=waveform, sample_rate=sr)

    try:
        response = engine.understand(
            image=image,
            audio=audio_tensor,
            text=message,
            question=message,
            max_tokens=max_tokens,
        )
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
    engine = _EngineHolder.image(model)
    seed_val = int(seed) if seed >= 0 else None
    try:
        image = engine.txt2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            steps=int(steps),
            guidance_scale=float(guidance),
            seed=seed_val,
        )
        return image
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
    engine = _EngineHolder.video(model)
    seed_val = int(seed) if seed >= 0 else None
    try:
        video = engine.txt2video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            fps=int(fps),
            steps=int(steps),
            guidance_scale=float(guidance),
            seed=seed_val,
        )
        frames = _video_tensor_to_frames(video)
        return frames
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
    """Ingest uploaded files into the RAG index."""
    engine = _EngineHolder.rag()

    if not files:
        return "No files uploaded."

    # Collect file paths.
    paths = []
    raw_texts = []
    for f in files:
        if isinstance(f, str):
            paths.append(f)
        elif hasattr(f, "name"):
            paths.append(f.name)
        else:
            raw_texts.append(str(f))

    try:
        if paths:
            engine.ingest(paths)
        for text in raw_texts:
            engine.ingest(text)
    except Exception as exc:
        return f"Error: {exc}"

    return f"Successfully ingested {len(files)} file(s). Total chunks: {engine.index_size}"


def _rag_query(question: str, top_k: int, rerank: bool) -> Tuple[str, str]:
    """Query the RAG engine."""
    engine = _EngineHolder.rag()

    if engine.index_size == 0:
        return "Index is empty. Please ingest documents first.", ""

    try:
        answer, sources = engine.query(question, top_k=int(top_k), rerank=rerank)
    except Exception as exc:
        return f"Error: {exc}", ""

    # Format sources.
    source_lines = []
    for i, chunk in enumerate(sources.chunks, 1):
        excerpt = chunk.text[:150].replace("\n", " ")
        source_lines.append(
            f"**[{i}]** Score: {chunk.score:.3f} | "
            f"Source: {chunk.metadata.get('source', 'unknown')}\n"
            f"> {excerpt}..."
        )
    sources_text = "\n\n".join(source_lines) if source_lines else "No sources retrieved."

    return answer.text, sources_text


def _rag_clear() -> str:
    """Clear the RAG index."""
    engine = _EngineHolder.rag()
    engine.clear_index()
    return "Index cleared."


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
    """Run an agent and return the output and reasoning trace."""
    engine = _EngineHolder.agent()

    trace_lines: List[str] = []

    if flow != "none":
        engine.create_agent(role="manager", max_steps=max_steps)
        engine.create_agent(role="worker", max_steps=max_steps)
        orchestrator = engine.create_flow(
            agents=["manager", "worker"],
            topology=flow,
        )
        result = engine.execute(orchestrator, task)
    else:
        if stream:
            for step in engine.stream(task, max_steps=max_steps):
                trace_lines.append(_format_step(step))
            result = engine.run(task, max_steps=max_steps)
        else:
            result = engine.run(task, max_steps=max_steps)

    # Build trace if not already streamed.
    if not trace_lines:
        for step in result.steps:
            trace_lines.append(_format_step(step))

    trace_text = "\n\n---\n\n".join(trace_lines) if trace_lines else "No steps recorded."
    return result.output, trace_text


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
            engine = _EngineHolder.text(params.get("model", "default"))
            prompt = input_values[0] if input_values else params.get("prompt", "")
            output = engine.generate(
                prompt,
                max_tokens=int(params.get("max_tokens", 128)),
            )

        elif ntype == "image_generate":
            engine = _EngineHolder.image(params.get("model", "default"))
            prompt = input_values[0] if input_values else params.get("prompt", "")
            output = engine.txt2img(
                prompt=prompt,
                width=int(params.get("width", 512)),
                height=int(params.get("height", 512)),
                steps=int(params.get("steps", 20)),
            )

        elif ntype == "audio_synthesize":
            engine = _EngineHolder.audio()
            text = input_values[0] if input_values else params.get("text", "")
            output = engine.synthesize(text)

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
            f"*TorchaVerse v0.1.0* | "
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
