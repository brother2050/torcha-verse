"""v0.10.5: 乱码处理 (half-character truncation + diagnostics).

覆盖:

* :class:`ByteTokenizer` decode 行为: 越界 id 产生完整 U+FFFD 序列,
  不再截断周围多字节 UTF-8 字符
* :func:`sanitise_generation`: 过滤 control chars + 截断长 FFFD 串
* :func:`quality_metrics` / :func:`garble_assessment`:
  printable_ratio / fffd_ratio 阈值
* :class:`PipelineService` text_completion / text_chat
  返回的 dict 含 ``garble_level`` + ``quality_warning`` 字段
* 端到端: 用户看到的不再有截断的半个字符, 而是"整齐的"
  U+FFFD 替换 + 黄色警告
"""
from __future__ import annotations

import unittest

from models.providers.tiny_transformer import ByteTokenizer
from models.providers._text_sanitiser import (
    sanitise_generation,
    quality_metrics,
    garble_assessment,
    _MAX_FFFD_RUN,
)


# ---------------------------------------------------------------------------
# 1. ByteTokenizer.decode: 越界 id 完整 U+FFFD 替代 skip
# ---------------------------------------------------------------------------
class TestByteTokenizerDecodeVulnIds(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = ByteTokenizer()

    def test_in_range_id_decodes_to_byte(self) -> None:
        # id=ord('A')+3 -> byte 'A'
        out = self.tok.decode([ord("A") + 3])
        assert out == "A", f"got {out!r}"

    def test_out_of_range_id_emits_full_fffd(self) -> None:
        # id=300 is outside the byte vocab; previously it was
        # *skipped* which could leave a partial UTF-8 sequence.
        out = self.tok.decode([300])
        assert out == "\ufffd", f"got {out!r}"

    def test_cjk_around_vuln_id_not_truncated(self) -> None:
        # The regression: "你好" (3 bytes each) interleaved with
        # an out-of-range id used to lose the second and third
        # bytes of the CJK character, producing a single FFFD
        # followed by garbage.  The fix emits a full FFFD for
        # the out-of-range id, preserving the surrounding CJK.
        cjk_bytes = "你好".encode("utf-8")
        ids = [1] + [b + 3 for b in cjk_bytes] + [300] + [2]
        out = self.tok.decode(ids)
        # Must contain "你好" intact + a single FFFD between BOS
        # and EOS special tokens.
        assert "你好" in out, f"CJK truncated: {out!r}"
        assert "\ufffd" in out, f"no replacement char: {out!r}"
        # Exactly one FFFD run (the out-of-range id).
        runs = out.split("\ufffd")
        # 1 FFFD => 2 splits
        assert len(runs) == 2, f"expected 1 FFFD, got {len(runs)-1}: {out!r}"

    def test_decode_is_idempotent(self) -> None:
        # Sanity: decode(decode(x)) == decode(x) for the in-vocab path.
        ids = [ord(c) + 3 for c in "hello"]
        out1 = self.tok.decode(ids)
        # Re-encode the *decoded* string and decode again.
        ids2 = [ord(c) + 3 for c in out1]
        out2 = self.tok.decode(ids2)
        assert out1 == out2, f"decode not idempotent: {out1!r} vs {out2!r}"

    def test_decode_empty(self) -> None:
        assert self.tok.decode([]) == ""

    def test_decode_only_special_tokens(self) -> None:
        # All-PAD/BOS/EOS/MASK -> empty.
        out = self.tok.decode([0, 1, 2, 259])
        assert out == "", f"expected empty, got {out!r}"


# ---------------------------------------------------------------------------
# 2. sanitise_generation
# ---------------------------------------------------------------------------
class TestSanitiseGeneration(unittest.TestCase):
    def test_empty(self) -> None:
        assert sanitise_generation("") == ""

    def test_strips_control_chars(self) -> None:
        # ``\x00`` (NUL) and ``\x7f`` (DEL) are removed; ``\n\t``
        # and ``\r`` are preserved.
        out = sanitise_generation("a\x00b\x01c\nd\te\x7ff")
        assert out == "abc\nd\tef", f"got {out!r}"

    def test_preserves_tab_newline_cr(self) -> None:
        out = sanitise_generation("a\nb\tc\rd")
        assert out == "a\nb\tc\rd", f"got {out!r}"

    def test_truncates_long_fffd_run(self) -> None:
        # 8+ FFFDs in a row -> truncate at the start of the run.
        out = sanitise_generation("hello" + "\ufffd" * 10 + "world")
        assert out == "hello", f"got {out!r}"

    def test_keeps_short_fffd_runs(self) -> None:
        # 7 FFFDs (below _MAX_FFFD_RUN) are preserved.
        out = sanitise_generation("hi" + "\ufffd" * 7 + "end")
        assert "\ufffd" in out
        assert "hi" in out and "end" in out

    def test_normalises_crlf(self) -> None:
        out = sanitise_generation("a\r\nb")
        assert out == "a\nb", f"got {out!r}"

    def test_strips_trailing_whitespace(self) -> None:
        out = sanitise_generation("hello   \n\n")
        assert out == "hello", f"got {out!r}"

    def test_idempotent(self) -> None:
        raw = "a\x00b\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffdc"
        once = sanitise_generation(raw)
        twice = sanitise_generation(once)
        assert once == twice, f"not idempotent: {once!r} vs {twice!r}"


# ---------------------------------------------------------------------------
# 3. quality_metrics
# ---------------------------------------------------------------------------
class TestQualityMetrics(unittest.TestCase):
    def test_empty_text(self) -> None:
        m = quality_metrics("")
        assert m["length"] == 0.0
        assert m["printable_ratio"] == 0.0
        assert m["fffd_ratio"] == 0.0
        assert m["control_ratio"] == 0.0

    def test_all_printable(self) -> None:
        m = quality_metrics("Hello, world!")
        assert m["printable_ratio"] == 1.0
        assert m["fffd_ratio"] == 0.0
        assert m["control_ratio"] == 0.0

    def test_all_fffd(self) -> None:
        m = quality_metrics("\ufffd" * 5)
        assert m["fffd_ratio"] == 1.0
        assert m["printable_ratio"] == 0.0

    def test_mixed(self) -> None:
        # "hi" + 3x FFFD + "ok" => 7 chars total
        m = quality_metrics("hi\ufffd\ufffd\ufffdok")
        assert m["length"] == 7.0
        # 4 printable (h, i, o, k)
        assert abs(m["printable_ratio"] - 4 / 7) < 1e-9
        assert abs(m["fffd_ratio"] - 3 / 7) < 1e-9

    def test_control_chars_counted(self) -> None:
        # 1 printable + 2 control + 1 printable = 4 chars
        m = quality_metrics("a\x00\x01b")
        assert m["control_ratio"] == 0.5
        assert m["printable_ratio"] == 0.5


# ---------------------------------------------------------------------------
# 4. garble_assessment
# ---------------------------------------------------------------------------
class TestGarbleAssessment(unittest.TestCase):
    def test_clean_text_is_ok(self) -> None:
        level, reason = garble_assessment("Hello, this is a normal response.")
        assert level == "ok", f"expected ok, got {level}: {reason}"
        assert reason == ""

    def test_garbled_heuristic(self) -> None:
        # Mostly FFFD + non-printable
        bad = "a" + "\ufffd" * 50 + "\x00\x01\x02"
        level, reason = garble_assessment(bad)
        assert level == "garbled", f"expected garbled, got {level}"
        assert "micro-transformer" in reason or "fffd" in reason.lower()

    def test_fffd_only_warn(self) -> None:
        # Below the garbled threshold (printable_ratio still ok).
        bad = "hello world " + "\ufffd" * 5
        level, reason = garble_assessment(bad)
        # printable is still >= 0.5 so it is ``ok`` here
        # (10 printable + 5 fffd out of 15 = 0.66 printable, 0.33 fffd)
        # fffd >= 0.2 -> warn
        assert level in ("warn", "ok"), f"got {level}"

    def test_garble_reason_mentions_real_checkpoint(self) -> None:
        # When fully garbled, the reason should hint at the
        # workaround (loading a real checkpoint).
        bad = "\ufffd" * 100
        _, reason = garble_assessment(bad)
        assert "checkpoint" in reason.lower() or "loading" in reason.lower()


# ---------------------------------------------------------------------------
# 5. PipelineService 端到端: quality_warning 字段
# ---------------------------------------------------------------------------
class TestPipelineServiceGarbleWarning(unittest.TestCase):
    def setUp(self) -> None:
        from nodes._helpers._backends import reset_default_backends
        reset_default_backends()

    def tearDown(self) -> None:
        from nodes._helpers._backends import reset_default_backends
        reset_default_backends()

    def test_text_completion_contains_garble_level(self) -> None:
        from serving.service._service import PipelineService
        svc = PipelineService()
        r = svc.text_completion(
            prompt="how are you?",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_tokens=10,
        )
        assert "garble_level" in r, (
            "v0.10.5: text_completion result must include "
            "'garble_level'"
        )
        assert r["garble_level"] in ("ok", "warn", "garbled"), (
            f"unexpected garble_level: {r['garble_level']!r}"
        )
        # Random-weight micro-transformer should be at least 'warn'.
        assert r["garble_level"] in ("warn", "garbled"), (
            f"expected warn/garbled for random-weight model, got "
            f"{r['garble_level']!r}"
        )
        assert "quality_warning" in r, (
            "v0.10.5: random-weight model should trigger quality_warning"
        )

    def test_text_chat_contains_garble_level(self) -> None:
        from serving.service._service import PipelineService
        svc = PipelineService()
        r = svc.text_chat(
            prompt="how are you?",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_tokens=10,
        )
        assert "garble_level" in r
        assert r["garble_level"] in ("ok", "warn", "garbled")

    def test_text_completion_text_has_no_control_chars(self) -> None:
        # The model layer applies sanitise_generation; the
        # returned ``text`` should not contain any of the
        # stripped control bytes.
        from serving.service._service import PipelineService
        svc = PipelineService()
        r = svc.text_completion(
            prompt="hello",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_tokens=10,
        )
        text = r.get("text", "")
        # Only \n \t \r are allowed.
        for ch in text:
            cp = ord(ch)
            if cp < 0x20 and ch not in "\n\t\r":
                assert False, (
                    f"control char {cp:#x} leaked into text: {text!r}"
                )
            if cp == 0x7F:
                assert False, f"DEL char in text: {text!r}"


if __name__ == "__main__":
    unittest.main()
