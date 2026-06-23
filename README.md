# TorchaVerse

A pure PyTorch all-modal generative AI framework covering text, image, audio, video, multimodal fusion, RAG, and agents.

## Quick Start

```bash
pip install -e .
```

### Text Generation
```python
from engines.text_engine import TextEngine

engine = TextEngine.from_config("llama-8b")
result = engine.generate("Hello, world!", max_tokens=100)
print(result)
```

### Image Generation
```python
from engines.image_engine import ImageEngine

engine = ImageEngine.from_config("sd15")
image = engine.txt2img("A serene mountain landscape at sunset")
image.save("output.png")
```

### RAG Query
```python
from engines.rag_engine import RAGEngine

engine = RAGEngine()
engine.ingest(["./docs/"])
answer, sources = engine.query("What is the architecture?")
```

### Agent Task
```python
from engines.agent_engine import AgentEngine

engine = AgentEngine()
result = engine.run("Search for the latest AI news and summarize")
```

## Architecture

```
torcha_verse/
├── config/           # Configuration center
├── infrastructure/   # Infrastructure layer
├── core/             # Core layer
├── models/           # Pure PyTorch model implementations
├── engines/          # Capability layer
├── rag/              # RAG subsystem
├── agents/           # Agent subsystem
├── tools/            # Built-in tools
├── training/         # Training & fine-tuning
├── serving/          # Application layer (API, CLI, Web UI)
├── evaluation/       # Evaluation
└── examples/         # Example code
```

## License

Apache-2.0
