# Gemma 4 E4B (with native thinking and tool calling)
import datetime

from agent import TinyAgent
from llm import LLM
from memory import Memory
from planning import NativeReAct
from toolbox import add, multiply, subtract
from tools import NativeTools

llm = LLM(model="gemma4:e4b", think=True)

# Math specialist
math_tools = NativeTools()
math_tools.add_tool("add", add)
math_tools.add_tool("subtract", subtract)
math_tools.add_tool("multiply", multiply)

# Math Agent
math_agent = TinyAgent(
    llm=llm, tools=math_tools, memory=Memory(), planner=NativeReAct()
)


def today() -> str:
    """Return today's date (YYYY-MM-DD)."""
    return datetime.date.today().isoformat()


def days_between(a: str, b: str) -> int:
    """Days between two ISO dates."""
    return (datetime.date.fromisoformat(b) - datetime.date.fromisoformat(a)).days


# Date specialist
date_tools = NativeTools()
date_tools.add_tool("today", today)
date_tools.add_tool("days_between", days_between)

# Date Agent
date_agent = TinyAgent(
    llm=llm, tools=date_tools, memory=Memory(), planner=NativeReAct()
)


def ask_math_agent(question: str) -> str:
    """Delegate to the math specialist."""
    return math_agent.run(question)


def ask_date_agent(question: str) -> str:
    """Ask queries to a sub-Agent that handles ISO dates."""
    return date_agent.run(question)


# Tools - Add the Math Agent and Date Agent as tools!
tools = NativeTools()
tools.add_tool("ask_math_agent", ask_math_agent)
tools.add_tool("ask_date_agent", ask_date_agent)

# Orchestrator Agent
orchestrator_agent = TinyAgent(
    llm=llm, tools=tools, memory=Memory(), planner=NativeReAct()
)

res = orchestrator_agent.run("If I save €4 per day until 2030, how much will I have?")
print(res)

for index, step in enumerate(orchestrator_agent.trajectory.runs[0]["steps"]):
    print(f"-- Step {index + 1} ---")
    if step.action:
        print(f"Tool: {step.action}")
        print(f"Observation: {step.observation}\n")
    else:
        print(f"Answer: {step.answer}\n")
