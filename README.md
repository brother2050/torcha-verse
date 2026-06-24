# TorchaVerse

A pure PyTorch all-modal generative AI framework covering text, image, audio, video, digital human, multimodal fusion, RAG, and agents.

**Version: 0.3.1** — Clean single-codebase with canvas type system, digital human nodes, plugin system, and paper integration.

## Quick Start

```bash
pip install -e .
```

### Pipeline-based Generation

```python
from pipeline.composer import PipelineBuilder

p = (PipelineBuilder("cinematic_shot")
    .node("image_txt2img", id="shot1", prompt="A cat playing piano",
          width=1024, height=576, steps=30)
    .node("image_upscale", id="shot1_up", scale=2)
    .connect("shot1", "shot1_up", output_key="image", input_key="image")
    .node("subtitle_burn", id="sub1")
    .connect("shot1_up", "sub1", output_key="image", input_key="video")
    .build())

result = p.run()
```

### Digital Human Pipeline

```python
from pipeline.composer import PipelineBuilder

p = (PipelineBuilder("digital_human")
    .node("dh_voice_clone", id="voice",
          reference_audio="sample.wav", text="Hello world",
          language="en", method="cosyvoice")
    .node("dh_talking_head", id="head",
          portrait_image="avatar.png", method="sadtalker")
    .connect("voice", "head", output_key="audio", input_key="audio")
    .node("dh_face_enhance", id="enhance", strength=0.7)
    .connect("head", "enhance", output_key="video", input_key="video")
    .build())

result = p.run()
```

### Canvas + AutoDirector

```python
from canvas.autodirector import AutoDirector
from pipeline.templates import TemplateRegistry

director = AutoDirector(TemplateRegistry())
canvas = director.generate("a 3-minute anime short about a girl and her robot")
pipeline = canvas.to_pipeline()
result = pipeline.run()
```

### Plugin System

```python
from plugins import PluginManager

mgr = PluginManager()
available = mgr.list_available()
mgr.load("my-custom-nodes")
```

### Paper Integration

```python
from papers import PaperRegistry

reg = PaperRegistry()
papers = reg.list()
# Install a paper's models
reg.get("musetalk")
```

## Architecture (v0.3.1)

```
torcha_verse/
├── config/               # Configuration center (4-tier: System/Project/User/Run)
├── infrastructure/        # L1: ConfigCenter, AuditLogger, ResourceBudget, SourceFetcher
├── assets/                # L2: AssetStore, ModelAsset, CharacterAsset, OutfitAsset
├── core/                  # L3: ModuleBus, Sampler, MemoryPool, PagedKVCache
├── nodes/                 # L4: 29 nodes (text/image/video/audio/subtitle/
│                          #     consistency/export/digital_human) + TypeSystem
├── pipeline/              # L5: DAG, PipelineBuilder, 12 templates, PromptStudio
├── consistency/           # Consistency framework (Character/Outfit/Scene/Depth)
├── canvas/                # Canvas (type-safe connections, versioning, sharing)
├── security/              # 4-gate defense (sanitizer, sandbox, filter, audit)
├── performance/           # Optimizer, Quantizer, BenchmarkSuite
├── plugins/               # Plugin system (3-layer: entry-point/directory/code)
├── papers/                # Paper integration (registry, adapters, YAML configs)
├── models/                # Pure PyTorch model implementations
├── rag/                   # RAG subsystem
├── agents/                # Agent subsystem
├── tools/                 # Built-in tools
├── training/              # Training & fine-tuning
├── serving/               # Application layer (API, CLI, Web UI)
├── scripts/               # Tooling (check_hardcoding.py)
├── evaluation/            # Evaluation
└── examples/              # Example code
```

## Key Features (v0.3.1)

| Feature | Description |
|---------|-------------|
| **6-Layer Architecture** | L1 Infrastructure → L2 Assets → L3 Core → L4 Nodes → L5 Pipeline → L6 Canvas |
| **ModuleBus** | Unified name resolver replacing scattered singletons |
| **29 Nodes** | Text, image, video, audio, subtitle, consistency, export, digital human |
| **Canvas Type System** | 19 port types with compatibility matrix, 7-point connection validation |
| **Digital Human** | 6 nodes: lip sync, talking head, portrait animate, full body, face enhance, voice clone |
| **Consistency Framework** | Character/Outfit/Scene/Depth four-suite with weighted conditions |
| **Plugin System** | 3-layer loading: entry-point, directory scan, programmatic registration |
| **Paper Integration** | Registry with 5 papers, reference_impl linking to Sutskever-30/labmlai/Karpathy/lucidrains |
| **Security** | 4-gate defense: input sanitizer, sandbox, output filter, audit/SBOM |
| **Performance** | SDPA, torch.compile, CUDA Graph, quantization (INT4/INT8/NF4) |

## Testing

```bash
# Run all tests (301 tests)
python -m pytest tests/ -v

# Run only E2E tests
python -m pytest tests/test_e2e_*.py tests/test_integration_combo.py -v

# Run plugin tests
python -m pytest tests/test_plugins.py -v

# Run paper integration tests
python -m pytest tests/test_papers.py -v
```

## License

Apache-2.0
