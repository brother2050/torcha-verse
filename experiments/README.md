# experiments/

Lightweight A/B testing framework for TorchaVerse.

## Why

When you ship a prompt change, sampler tweak, or LoRA blend to a
fraction of production traffic, you need (a) deterministic per-user
arm assignment and (b) per-arm metric rollups. `experiments/` is a
dependency-free in-process implementation of that workflow.

## Components

- `Experiment` — dataclass: name, list of `Variant`, primary metric,
  start / end timestamps.
- `Variant` — dataclass: name, weight, per-arm config blob.
- `bucket_assign(user_id, variants)` — hash-modulo assignment,
  Optimizely-style.  Same user always lands in the same arm for a
  fixed experiment configuration.
- `ExperimentRunner` — wraps an `Experiment`, exposes `pick`,
  `record`, `summary`, and `stop`.  Keeps an in-memory
  `{variant: {metric: [values]}}` table.

## Quick start

```python
from experiments import (
    Experiment, Variant, ExperimentRunner,
)

exp = Experiment(
    name="prompt_v2",
    variants=[
        Variant("control", weight=1.0, config={"prompt": "old"}),
        Variant("treatment", weight=1.0, config={"prompt": "new"}),
    ],
    primary_metric="ctr",
)
runner = ExperimentRunner(exp)
runner.record(user_id="abc-123", metric_name="ctr", value=0.17)
print(runner.summary())
```

## Determinism notes

`bucket_assign` uses `md5(user_id)[:4]` reduced modulo the total
weight — stable across processes and platforms, so a rollout is
reproducible from logs alone.  Adding a new variant mid-experiment
will reshuffle existing users unless you also bump the hash salt.

## Not in scope

- No remote config / feature flag service.
- No Bayesian stopping rules — call `runner.stop()` yourself.
- No thread safety; one runner per process.
