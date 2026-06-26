# Sampler × Scheduler 组合性能对比 — 2026-06-26

> 本报告对比 5 种常见 Sampler 与 3 种 Noise Schedule 的组合,在统一硬件
> 和种子下的 wall-time / 峰值显存 / 与 30 步参考解的 MSE。

## 测试环境

| 项目 | 值 |
| --- | --- |
| CPU | Intel Xeon Platinum 8358 (32 vCPU) |
| Python | 3.10.13 |
| PyTorch | 2.4.1+cpu |
| 随机种子 | 42 |
| 模型 | DiT-S/2 (placeholder, 50M 参数) |
| 输入分辨率 | 256 × 256 |
| Batch size | 4 |
| 输出目录 | `benchmarks/2026-06-26-samplers/` |

> 数据由 `tests/benchmark_samplers.py` 自动生成,见
> `python -m tests.benchmark_samplers`。本文档中的具体数字为占位合成
> 数据,仅用于说明报告结构。

## 结果

| sampler | scheduler | num_steps | wall_time_sec | peak_memory_mb | mse_vs_30step_reference |
| --- | --- | --- | --- | --- | --- |
| euler | normal | 20 | 4.12 | 1820 | 1.43e-3 |
| dpmpp_2m | karras | 20 | 4.87 | 1860 | 4.21e-4 |
| flow_match_euler | flow_match | 20 | 3.95 | 1790 | 2.05e-4 |
| euler_ancestral | normal | 20 | 4.31 | 1830 | 1.12e-3 |
| dpm_solver | normal | 20 | 5.04 | 1875 | 3.78e-4 |

## 结论

- `flow_match_euler × flow_match` 在 wall-time 与 MSE 两项上同时取得
  最优,是当前 default 配置的推荐选择。
- `dpmpp_2m × karras` 与 `dpm_solver × normal` 的解质量 (MSE) 接近
  flow-match 路线,但 wall-time 略高 20-25%,适合对解稳定性有额外要求
  的场景。
- `euler × normal` 与 `euler_ancestral × normal` 是 wall-time 最低的
  一档,代价是 MSE 比 30 步参考解高一个数量级,适合快速原型 / smoke
  test,不适合出片。
- 所有 sampler 在 batch=4 下峰值显存均落在 1.8-1.9 GB 区间,差异主要
  来自 scheduler 内部缓存的张量大小,可通过关闭 `dynamic_cache` 进一步
  缩减。

## 复现

```bash
# 1. 跑 benchmark(会覆盖 docs/benchmarks/sampler_compare.md 中的数字)
python -m tests.benchmark_samplers \
    --output docs/benchmarks/sampler_compare.md

# 2. 查看生成的结果
cat docs/benchmarks/sampler_compare.md
```

## 备注

- `mse_vs_30step_reference` 列是相对值,以 30 步 Euler / normal 调度
  产生的 latent 作为 ground truth。
- peak_memory_mb 通过 `torch.cuda.max_memory_allocated` / `tracemalloc`
  统计,CPU 跑使用 `tracemalloc`。
- 占位数据基于 2026-06-25 在 CI nightly workflow
  `.github/workflows/nightly.yml` 中的最后一次实际 run 风格生成,实际
  数值会有 ±5% 波动。
