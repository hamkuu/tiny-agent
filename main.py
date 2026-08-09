from agent import TinyAgent
from llm import LLM
from memory import Memory
from planning import ReAct
from toolbox import multiply, add, subtract
from tools import NativeTools, Tools

# Gemma 3 12B (no native thinking or tool calling)
llm = LLM(model="gemma3:12b")

# Tools
tools = Tools(requires_approval=[])
tools.add_tool("add", add, "add(a: str, b: str)")
tools.add_tool("subtract", subtract, "subtract(a: str, b: str)")
tools.add_tool("multiply", multiply, "multiply(a: str, b: str)")

# Memory
memory = Memory()

# ReAct
react = ReAct(max_steps=10)

# Create agent
agent = TinyAgent(llm=llm, tools=tools, memory=memory, planner=react)

# Multistep task with reasoning
res = agent.run("What is (4.6 + 6.685) x 4, and then subtract 3.14 from the result?")
print(res)
print(agent.trajectory.runs)
