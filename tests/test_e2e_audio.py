"""端到端连通测试 -- 音频节点(audio_tts / audio_music)。

覆盖单一功能正常路径:构建 Pipeline -> 执行 -> 验证输出键。
所有节点返回占位数据,无需 GPU。
"""
from __future__ import annotations

from pipeline.composer import PipelineBuilder


# ---------------------------------------------------------------------------
# audio_tts
# ---------------------------------------------------------------------------
def test_audio_tts_e2e(pipeline_ctx):
    """构建 audio_tts Pipeline -> 执行 -> 验证输出含 audio 与 sample_rate。"""
    pipeline = (
        PipelineBuilder("audio_tts_e2e")
        .node(
            "audio_tts",
            id="tts",
            text="欢迎来到数字人世界。",
            voice="zh-female-1",
            language="zh",
            speed=1.0,
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["tts"]

    assert "audio" in out, "audio_tts 输出应包含 'audio' 键"
    assert "sample_rate" in out, "audio_tts 输出应包含 'sample_rate' 键"
    assert isinstance(out["sample_rate"], int) and out["sample_rate"] > 0


# ---------------------------------------------------------------------------
# audio_music
# ---------------------------------------------------------------------------
def test_audio_music_e2e(pipeline_ctx):
    """构建 audio_music Pipeline -> 执行 -> 验证输出含 audio。"""
    pipeline = (
        PipelineBuilder("audio_music_e2e")
        .node(
            "audio_music",
            id="music",
            prompt="轻快的电子流行乐,120 BPM",
            duration=30.0,
        )
        .build()
    )

    results = pipeline.run(pipeline_ctx)
    out = results["music"]

    assert "audio" in out, "audio_music 输出应包含 'audio' 键"
