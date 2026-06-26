# Model Loading Protocol (v0.8.0)

> **Status**: v0.8.0 design contract.  Backed by
> `models/base.py::ModelMixin` and `core/checkpoint_loader.py`.

This document describes how torcha-verse v0.8.0 loads real upstream
model weights.  It is the single source of truth for the
`from_pretrained` / `save_pretrained` contract that every model
implementation in the project must follow.

---

## 1. Why a new contract?

The pre-v0.8.0 codebase shipped 39 "smoke-test" model
implementations, all initialised via `from_random()` (i.e. random
weights that produce noise at forward time).  The v0.8.0 release
introduces the **diffusers-compatible ModelMixin** protocol so
that:

1. Real upstream checkpoints (HunyuanDiT, HunyuanVideo, FLUX,
   StableDiffusion, etc.) can be loaded with a single line of
   code.
2. New model authors have a single concrete base class to extend
   (`models.base.ModelMixin`).
3. Loading is reproducible across CPU / GPU and across
   `torch_dtype` choices.

The API surface is intentionally a strict subset of the
`diffusers.ModelMixin` contract so that future migration to
`diffusers` is non-breaking.

---

## 2. The ModelMixin contract

```python
import torch
from models.base import ModelMixin

class MyModel(ModelMixin):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.linear = torch.nn.Linear(64, 64)

# 1. Save a random model to disk.
m = MyModel()
m.save_pretrained("/tmp/mymodel")

# 2. Load it back with a different dtype.
m2 = MyModel.from_pretrained(
    "/tmp/mymodel",
    torch_dtype=torch.float16,
    device="cpu",
    strict=False,
)
```

The supported kwargs of `from_pretrained` are:

| Kwargs | Type | Default | Meaning |
|---|---|---|---|
| `subfolder` | `str \| None` | `None` | Look inside `<dir>/<subfolder>`. |
| `torch_dtype` | `torch.dtype \| None` | `None` | Cast every float tensor on load. |
| `device` | `str \| torch.device` | `"cpu"` | Pin every parameter to a single device. |
| `device_map` | `str \| dict` | `None` | `"cpu"` / `"cuda"` / `{"layer.0": "cuda:0"}`. |
| `variant` | `str \| None` | `None` | Pick `<name>.<variant>.safetensors` siblings. |
| `key_renames` | `Mapping[str, str]` | `None` | Apply checkpoint key migration. |
| `strict` | `bool` | `False` | Raise on missing/extra keys (diffusers default). |
| `config` | `dict \| None` | `None` | Override the on-disk `config.json`. |

`save_pretrained` supports `safe_serialization` (default `True`).
When the `safetensors` package is unavailable the call falls
back to `torch.save` (the file extension is preserved).

---

## 3. File layout conventions

The directory written by `save_pretrained` contains:

```
<save_directory>/
├── <class_name_lowercase>.safetensors   # the state-dict
└── config.json                          # (optional) config snapshot
```

Override the file extension via the `_default_file_extension`
class attribute (defaults to `.safetensors`):

```python
class MyModel(ModelMixin):
    _default_file_extension = ".bin"
```

`from_pretrained` will look for these files (in order) when
`pretrained_model_name_or_path` is a directory:

1. `<class_name_lowercase><_default_file_extension>` (e.g.
   `mymodel.safetensors`)
2. `model<_default_file_extension>`
3. `diffusion_pytorch_model<_default_file_extension>`

When `variant` is set, each candidate name is tried both with
and without the variant suffix (`<name>.<variant>.safetensors`
siblings).

Sharded layouts (HuggingFace-style) are supported transparently:
when `<file>.safetensors.index.json` exists next to the requested
file, the loader stitches every shard referenced in the index.

---

## 4. Upstream key-rename tables

Real upstream checkpoints (HunyuanDiT, HunyuanVideo, FLUX) use
their own naming conventions.  torcha-verse ships a small
collection of declarative rename tables in
`core/checkpoint_loader.py`:

```python
from core.checkpoint_loader import HUNYUAN_DIT_KEY_MAP, load_hunyuan_dit
```

`HUNYUAN_DIT_KEY_MAP` is a 50-entry `{upstream: local}` rewrite
table.  Entries with a `{i}` placeholder are expanded per-block at
load time (the user controls `num_blocks` when calling
`load_hunyuan_dit`).

Adding a new upstream is a two-step process:

1. Add a new `<NAME>_KEY_MAP` constant in `core/checkpoint_loader.py`.
2. Add a `load_<name>(...)` helper that calls
   `ModelMixin.from_pretrained` with the map as `key_renames`.

CI will fail if the new map contains stale keys (see
`tests/test_from_pretrained_smoke.py`).

---

## 5. Diffusers compatibility matrix

The v0.8.0 surface area is a deliberate subset of
`diffusers.ModelMixin`.  The following diffusers features are
**not** yet supported and are deferred to v0.9.0:

- HuggingFace Hub `from_pretrained("org/repo")` (requires the
  `huggingface_hub` optional dep).
- Tied weight detection / key auto-mapping.
- Automatic sharding with `accelerate`.
- `variant="fp16"` resolving a single fp32 checkpoint to its
  fp16 sibling (the loader does try both names but does not
  auto-convert).

The v0.8.0 contract is enough to load any checkpoint that ships
a `*.safetensors` file plus (optionally) a `config.json`
sidecar.  Models that ship a `*.bin` only are loaded via the
`torch.load` path (still goes through the same call shape).

---

## 6. Low-level helpers

The two public helpers below are re-exported from `models.base`
and also live in `core.checkpoint_loader`:

```python
from core.checkpoint_loader import (
    load_safetensors,
    save_safetensors,
    transform_checkpoint_dict_key,
    load_state_dict_with_renames,
)
```

* `load_safetensors(path, device="cpu", dtype=None) -> dict`
* `save_safetensors(state_dict, path) -> None`
* `transform_checkpoint_dict_key(state_dict, key_map) -> dict`
* `load_state_dict_with_renames(model, state_dict, key_map, strict=False) -> (missing, unexpected)`

These helpers are the building blocks the ModelMixin uses
internally.  Direct use is supported but should be the exception
(only when callers need finer control than `from_pretrained`
exposes).

---

## 7. End-to-end example: loading a real HunyuanDiT checkpoint

```python
import torch
from core.checkpoint_loader import load_hunyuan_dit

model = load_hunyuan_dit(
    "/path/to/hunyuan-dit-v1.2",
    torch_dtype=torch.float16,
    device="cuda:0",
    num_blocks=20,
    strict=False,
)
# ``model`` is now a fully-initialised, eval-mode
# ``models.image.dit.HunyuanDiT`` ready to be plugged into a
# ``call_diffusion_loop_backend`` invocation.
```

The full e2e pipeline then becomes:

```python
from nodes._helpers import call_diffusion_loop_backend
import torch

latents = torch.randn(1, 4, 64, 64)
text_embeds = model.encode_text("a serene mountain landscape")
out = call_diffusion_loop_backend(
    bus=None,
    name="hunyuan_dit",
    model=model,
    latents=latents,
    text_embeds=text_embeds,
    num_inference_steps=30,
    guidance_scale=4.5,
    sampler="flow_match_euler",
    shift=7.0,
)
# ``out["latents"]`` is the denoised latent tensor.
```

---

## 8. CI guards (v0.8.0)

The following CI guards protect the contract:

| Test | What it checks |
|---|---|
| `tests/test_from_pretrained_smoke.py` | round-trip save/load on every ModelMixin subclass, dtype cast, device_map, key_renames |
| `tests/test_placeholder_registry.py` | every `pass` / `NotImplementedError` is registered |
| `tests/test_hunyuan_dit_smoke.py` (v0.8.5) | end-to-end load from a synthetic upstream-style checkpoint |

The smoke test runs on every commit (~2 s per `from_pretrained`
round-trip, so the total cost is sub-1 s on a stock dev box).

---

## 9. Migration notes from v0.6.x

The pre-v0.8 `BaseModel` had `save(path)` / `load(path, strict=True)`
methods.  These are **deprecated** but still work (they delegate
to `save_pretrained` / `from_pretrained` respectively).  New code
should use the v0.8.0 surface.

The `_default_file_extension` change from `.pt` (legacy) to
`.safetensors` (current) is a one-line override on any class
that needs to keep the old on-disk format.

Pickled checkpoints (`.bin` / `.pt`) are no longer trusted by
default; `strict=False` plus a one-line `torch.load` is the
recommended way to load them.
