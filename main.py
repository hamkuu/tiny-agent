from agent import TinyAgent
from llm import LLM
from memory import Memory
from toolbox import multiply
from tools import NativeTools

llm = LLM(model="gemma4:e4b")
memory = Memory()

tools = NativeTools(requires_approval=[])
tools.add_tool("multiply", multiply)

agent = TinyAgent(llm=llm, memory=memory, tools=tools)

res = agent.run("What is 5.1 times 7.3?")
print(res)
print(agent.memory.get_messages())
