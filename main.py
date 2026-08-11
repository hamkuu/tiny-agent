from agent import TinyAgent
from llm import LLM
from memory import Memory
from planning import NativeReAct
from toolbox import add, multiply, subtract
from tools import NativeTools

# Gemma 4 E4B (with native thinking and tool calling)
llm = LLM(model="gemma4:e4b", think=True)

# Register tools
tools = NativeTools(requires_approval=[])
tools.add_tool("add", add, "add(a: str, b: str)")
tools.add_tool("subtract", subtract, "subtract(a: str, b: str)")
tools.add_tool("multiply", multiply, "multiply(a: str, b: str)")

# Memory
memory = Memory()

# ReAct
react = NativeReAct(max_steps=10)

# Create agent
agent = TinyAgent(llm=llm, tools=tools, memory=memory, planner=react)

# Multi-step task with reasoning
res = agent.run("What is (4.6 + 6.685) x 4, and then subtract 3.14 from the result?")
print(res)
