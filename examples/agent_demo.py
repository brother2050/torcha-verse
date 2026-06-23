"""Agent demo.

Demonstrates ReAct agent with tool calling and multi-agent sequential flow.

Run with::

    python examples/agent_demo.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tools.calculator import CalculatorTool
from tools.python_executor import PythonExecutorTool
from tools.file_ops import FileOpsTool
from core.tool_registry import ToolRegistry


def main() -> None:
    print("=" * 60)
    print("TorchaVerse — Agent Demo")
    print("=" * 60)

    # --- 1. Register tools ---
    print("\n[1] Registering tools...")
    registry = ToolRegistry()
    calc = CalculatorTool()
    py_exec = PythonExecutorTool()
    file_ops = FileOpsTool()

    registry.register_tool(
        name="calculator",
        func=calc.execute,
        description="Evaluate mathematical expressions safely",
        parameter_schema={
            "expression": {"type": "string", "description": "Math expression"},
        },
    )
    registry.register_tool(
        name="python_executor",
        func=py_exec.execute,
        description="Execute Python code and return output",
        parameter_schema={
            "code": {"type": "string", "description": "Python code to execute"},
        },
    )
    registry.register_tool(
        name="file_ops",
        func=file_ops.execute,
        description="Read, write, and manage files",
        parameter_schema={
            "operation": {"type": "string"},
            "path": {"type": "string"},
        },
    )
    print(f"    Registered {len(registry.get_tool_descriptions())} tools")
    for desc in registry.get_tool_descriptions():
        print(f"      - {desc['name']}: {desc['description']}")

    # --- 2. Execute calculator ---
    print("\n[2] Testing CalculatorTool...")
    result = registry.execute_tool("calculator", {"expression": "2 + 3 * 4"})
    print(f"    2 + 3 * 4 = {result}")

    result = registry.execute_tool("calculator", {"expression": "sqrt(144) + 10"})
    print(f"    sqrt(144) + 10 = {result}")

    result = registry.execute_tool("calculator", {"expression": "sin(3.14159 / 2)"})
    print(f"    sin(pi/2) = {result}")

    # --- 3. Execute Python code ---
    print("\n[3] Testing PythonExecutorTool...")
    code = "print('Hello from TorchaVerse!'); x = sum(range(100)); print(f'Sum: {x}')"
    result = registry.execute_tool("python_executor", {"code": code})
    print(f"    Output: {result}")

    # --- 4. File operations ---
    print("\n[4] Testing FileOpsTool...")
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "outputs")
    os.makedirs(test_dir, exist_ok=True)
    test_file = os.path.join(test_dir, "agent_test.txt")

    registry.execute_tool("file_ops", {"operation": "write", "path": test_file, "content": "Hello from Agent!"})
    result = registry.execute_tool("file_ops", {"operation": "read", "path": test_file})
    print(f"    File content: {result}")

    # --- 5. Simulate ReAct loop ---
    print("\n[5] Simulating ReAct loop...")
    task = "Calculate the factorial of 10 and then compute its square root"

    steps = [
        {"thought": "I need to calculate 10! first", "action": "calculator", "params": {"expression": "1*2*3*4*5*6*7*8*9*10"}},
        {"thought": "10! = 3628800, now compute sqrt", "action": "calculator", "params": {"expression": "sqrt(3628800)"}},
    ]

    for i, step in enumerate(steps):
        print(f"\n    Step {i+1}:")
        print(f"      Thought: {step['thought']}")
        print(f"      Action:  {step['action']}({step['params']})")
        result = registry.execute_tool(step["action"], step["params"])
        print(f"      Result:  {result}")

    print(f"\n    Final Answer: sqrt(10!) = {registry.execute_tool('calculator', {'expression': 'sqrt(3628800)'})}")

    # --- 6. Multi-agent flow ---
    print("\n[6] Multi-agent Sequential Flow simulation...")
    agents = ["Researcher", "Writer", "Reviewer"]
    task = "Write a summary about TorchaVerse"
    current_output = task

    for agent_name in agents:
        print(f"\n    [{agent_name}]")
        print(f"      Input:  {current_output[:60]}...")
        # Simulate agent processing.
        current_output = f"[{agent_name} processed] {current_output}"
        print(f"      Output: {current_output[:60]}...")

    print(f"\n    Final output: {current_output[:80]}...")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
