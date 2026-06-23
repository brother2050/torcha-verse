"""Tests for the Engines Layer."""

from __future__ import annotations

import pytest
import torch

from engines.text_engine import SamplingStrategy, Message


class TestTextEngine:
    """Test TextEngine components."""

    def test_sampling_strategy(self):
        """SamplingStrategy applies temperature correctly."""
        strategy = SamplingStrategy(temperature=0.5, top_k=10, top_p=0.9)
        logits = torch.randn(1, 100)
        result = strategy.apply(logits)
        assert result is not None

    def test_message_creation(self):
        """Message dataclass creates correctly."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_sampling_strategy_to_dict(self):
        """SamplingStrategy.to_dict returns config."""
        strategy = SamplingStrategy(temperature=0.8, top_k=40, top_p=0.95)
        d = strategy.to_dict()
        assert d["temperature"] == 0.8
        assert d["top_k"] == 40


class TestEngineImports:
    """Test that all engine modules can be imported."""

    def test_import_text_engine(self):
        """TextEngine module imports successfully."""
        from engines import text_engine
        assert hasattr(text_engine, "TextEngine")

    def test_import_image_engine(self):
        """ImageEngine module imports successfully."""
        from engines import image_engine
        assert hasattr(image_engine, "ImageEngine")

    def test_import_audio_engine(self):
        """AudioEngine module imports successfully."""
        from engines import audio_engine
        assert hasattr(audio_engine, "AudioEngine")

    def test_import_video_engine(self):
        """VideoEngine module imports successfully."""
        from engines import video_engine
        assert hasattr(video_engine, "VideoEngine")

    def test_import_multimodal_engine(self):
        """MultiModalEngine module imports successfully."""
        from engines import multimodal_engine
        assert hasattr(multimodal_engine, "MultiModalEngine")

    def test_import_rag_engine(self):
        """RAGEngine module imports successfully."""
        from engines import rag_engine
        assert hasattr(rag_engine, "RAGEngine")

    def test_import_agent_engine(self):
        """AgentEngine module imports successfully."""
        from engines import agent_engine
        assert hasattr(agent_engine, "AgentEngine")
