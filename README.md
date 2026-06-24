# TorchaVerse

A pure PyTorch all-modal generative AI framework covering text, image, audio, video, multimodal fusion, RAG, and agents.

**Version: 0.3.0-alpha** — Architecture redesign with 6-layer + ModuleBus, asset-driven consistency, canvas-based pipeline composition, and defense-in-depth security.

## Quick Start

```bash
pip install -e .
```

### Pipeline-based Generation (v0.3.0)

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

### Canvas + AutoDirector

```python
from canvas.autodirector import AutoDirector
from pipeline.templates import TemplateRegistry

director = AutoDirector(TemplateRegistry())
canvas = director.generate("a 3-minute anime short about a girl and her robot")
# Edit canvas nodes visually, then run
pipeline = canvas.to_pipeline()
result = pipeline.run()
```

### Consistency Framework

```python
from consistency import ConsistencyProfile, ConsistencyManager
from consistency.pipeline import ConsistencyPipeline

mgr = ConsistencyManager()
profile = mgr.create_profile("cinematic",
    character_weight=0.8, outfit_weight=0.7, scene_weight=0.6)
cp = ConsistencyPipeline(profile=profile, character=my_character_asset)
result = cp.generate("sakura walking to school at dawn", width=1024, height=1024)
```

### ModuleBus (replaces scattered singletons)

```python
from core.module_bus import ModuleBus, register_module

bus = ModuleBus()

@register_module("model.text", "my-llama")
def create_llama():
    return MyModel(...)

model = bus.resolve("model.text", "my-llama")
```

### Legacy Engine API (backward compatible)

```python
from engines.text_engine import TextEngine
engine = TextEngine.from_config("llama-8b")
result = engine.generate("Hello, world!", max_tokens=100)
```

## Architecture (v0.3.0)

```
torcha_verse/
├── config/               # Configuration center (4-tier: System/Project/User/Run)
│   ├── _defaults/         # System-level defaults (immutable)
│   └── *.yaml             # Project-level configs
├── infrastructure/        # L1: Infrastructure (ConfigCenter, AuditLogger,
│                          #     ResourceBudget, SourceFetcher, DeviceManager)
├── assets/                # L2: Asset layer (AssetStore, ModelAsset,
│                          #     CharacterAsset, OutfitAsset, SceneAsset, DepthAsset)
├── core/                  # L3: Abstraction (ModuleBus, Sampler, MemoryPool,
│                          #     PagedKVCache, RuntimeScheduler)
├── nodes/                 # L4: Node system (23 nodes: text/image/video/
│                          #     audio/subtitle/consistency/export)
├── pipeline/              # L5: Pipeline (DAG, PipelineBuilder, 12 templates,
│                          #     PromptStudio)
├── consistency/           # Consistency framework (Character/Outfit/Scene/Depth
│                          #     engines, ConsistencyPipeline, ScoreCalculator)
├── canvas/                # Canvas system (Canvas, Versioning, Sharing,
│                          #     AutoDirector, CommunityRegistry)
├── security/              # Security (InputSanitizer, Sandbox, OutputFilter,
│                          #     Audit/SBOM)
├── performance/            # Performance (Optimizer, Quantizer, BenchmarkSuite)
├── engines/               # Legacy capability layer (backward compatible)
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

## Key Features (v0.3.0)

| Feature | Description |
|---------|-------------|
| **6-Layer Architecture** | L1 Infrastructure → L2 Assets → L3 Core → L4 Nodes → L5 Pipeline → L6 Canvas |
| **ModuleBus** | Unified name resolver replacing 7 scattered singletons |
| **AssetStore** | Content-addressed storage with Hot/Warm/Cold tiers, version tracking |
| **Consistency Framework** | Character/Outfit/Scene/Depth four-suite with weighted conditions |
| **Canvas** | 12 built-in templates + full custom mode, versioning, sharing |
| **AutoDirector** | Topic-to-canvas intelligent pipeline generation |
| **Subtitle System** | 5 generation methods (ASR/LLM/align/human/translate), burn, export |
| **Multi-Source Download** | Local/HuggingFace/ModelScope/Modelers with resume + license audit |
| **ResourceBudget** | Hard constraint budget tracking preventing OOM |
| **RuntimeScheduler** | Unified CPU/async/GPU stream scheduling with DAG dependencies |
| **Security** | 4-gate defense: input sanitizer, sandbox, output filter, audit/SBOM |
| **Performance** | SDPA, torch.compile, CUDA Graph, quantization (INT4/INT8/NF4) |

## Testing

```bash
# Run v0.3.0 test suite
python -m pytest tests/test_v03_*.py -v

# Run legacy tests
python -m pytest tests/test_*.py -v --ignore=tests/test_v03_*.py
```

## License

Apache-2.0
