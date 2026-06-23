"""Tests for the Agent subsystem."""

from __future__ import annotations

import pytest
import torch

from agents.base_agent import BaseAgent, Step, Result, Memory, ShortTermMemory
from agents.react_agent import ReActAgent
from agents.tool_call_agent import ToolCallAgent
from agents.flows.sequential import SequentialFlow
from agents.flows.hierarchical import HierarchicalFlow
from agents.flows.debate import DebateFlow
from agents.flows import FlowOrchestrator


class TestBaseAgent:
    """Test BaseAgent and memory."""

    def test_short_term_memory(self):
        """ShortTermMemory stores and retrieves messages."""
        mem = ShortTermMemory(max_messages=3)
        mem.add(role="user", content="msg1")
        mem.add(role="user", content="msg2")
        assert len(mem.get_messages()) == 2
        mem.add(role="user", content="msg3")
        mem.add(role="user", content="msg4")  # should evict msg1
        assert len(mem.get_messages()) == 3

    def test_step_dataclass(self):
        """Step dataclass creates correctly."""
        step = Step(thought="thinking", action="search", observation="result")
        assert step.thought == "thinking"
        assert step.action == "search"

    def test_result_dataclass(self):
        """Result dataclass creates correctly."""
        result = Result(output="done", steps=[], metadata={})
        assert result.output == "done"


class TestReActAgent:
    """Test ReActAgent."""

    def test_instantiation(self):
        """ReActAgent can be instantiated with role and model."""
        agent = ReActAgent(role="assistant", model=None, tools=[])
        assert agent is not None

    def test_format_tools(self):
        """format_tools returns formatted string."""
        agent = ReActAgent(role="assistant", model=None, tools=["calc", "search"])
        formatted = agent.format_tools(["calc", "search"])
        assert isinstance(formatted, str)
        assert "calc" in formatted


class TestToolCallAgent:
    """Test ToolCallAgent."""

    def test_instantiation(self):
        """ToolCallAgent can be instantiated."""
        agent = ToolCallAgent(role="assistant", model=None, tools=[])
        assert agent is not None


class TestSequentialFlow:
    """Test SequentialFlow."""

    def test_instantiation(self):
        """SequentialFlow can be instantiated with agents."""
        agent = ReActAgent(role="worker", model=None, tools=[])
        flow = SequentialFlow(agents=[agent])
        assert flow is not None


class TestHierarchicalFlow:
    """Test HierarchicalFlow."""

    def test_instantiation(self):
        """HierarchicalFlow can be instantiated with manager and workers."""
        manager = ReActAgent(role="manager", model=None, tools=[])
        worker = ReActAgent(role="worker", model=None, tools=[])
        flow = HierarchicalFlow(manager=manager, workers=[worker])
        assert flow is not None


class TestDebateFlow:
    """Test DebateFlow."""

    def test_instantiation(self):
        """DebateFlow can be instantiated with 2+ agents."""
        a1 = ReActAgent(role="debater1", model=None, tools=[])
        a2 = ReActAgent(role="debater2", model=None, tools=[])
        flow = DebateFlow(agents=[a1, a2])
        assert flow is not None


class TestFlowOrchestrator:
    """Test FlowOrchestrator."""

    def test_create_sequential(self):
        """FlowOrchestrator creates sequential flow."""
        agent = ReActAgent(role="worker", model=None, tools=[])
        orchestrator = FlowOrchestrator()
        flow = orchestrator.create_flow(agents=[agent], topology="sequential")
        assert flow is not None

    def test_create_hierarchical(self):
        """FlowOrchestrator creates hierarchical flow."""
        manager = ReActAgent(role="manager", model=None, tools=[])
        worker = ReActAgent(role="worker", model=None, tools=[])
        orchestrator = FlowOrchestrator()
        flow = orchestrator.create_flow(agents=[manager, worker], topology="hierarchical")
        assert flow is not None

    def test_create_debate(self):
        """FlowOrchestrator creates debate flow."""
        a1 = ReActAgent(role="debater1", model=None, tools=[])
        a2 = ReActAgent(role="debater2", model=None, tools=[])
        orchestrator = FlowOrchestrator()
        flow = orchestrator.create_flow(agents=[a1, a2], topology="debate")
        assert flow is not None
