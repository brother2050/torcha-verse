# Changelog

项目初期变更记录。初期重点：架构简洁、节点能跑、测试可过。

## [Unreleased] — 初期整理

### 架构简化
- 删除冗余的旧版本历史记录与架构清理叙事。
- 删除文档中过时的版本号与"v0.3.0/v0.3.1"中间过渡描述。
- 节点 `execute()` 统一通过 `nodes/_helpers.py` 中的 `call_text_backend` /
  `call_image_backend` / `call_video_backend` / `call_audio_backend`
  解析后端,默认回退 echo 工厂,允许在无模型情况下端到端跑通。
- 节点实现通过 `core.module_bus.ModuleBus` + `LLMProvider` 协议接入真模型,
  流水线层不再做 passthrough。

### 一致性评分
- `consistency/score.py` 优先尝试 `open_clip` / `torch.hub` 的真 CLIP/DINOv2
  视觉特征;未安装时回退到项目内的轻量占位特征提取器(随机投影,固定维度)。
- 评分指标与 `ConsistencyScore` 数据结构保持不变,接口兼容现有调用方。

### 工程化
- 新增 `pyproject.toml`(含 pytest 配置)。
- `Dockerfile` 改为多阶段构建(builder / test / runtime)。
- 新增 `.github/workflows/ci.yml`,跑 lint + 全量测试。
- `requirements.txt` 同步精简。

### 文档
- 重写 `README.md`,对齐新结构与新流水线示例。
- `docs/architecture.md` / `docs/operations.md` 同步精简,移除历史叙事。
- 新增 `docs/DEFERRED_TASKS.md`,登记开发初期延后处理的任务(当前条目:hardcoding 规约化)。
