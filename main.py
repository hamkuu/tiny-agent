from agent import TinyAgent
from llm import LLM
from memory import Memory
from toolbox import multiply
from tools import Tools, NativeTools

llm = LLM(model="gemma3:12b")
memory = Memory()

tools = Tools(requires_approval=[])
tools.add_tool(
    name="multiply",
    func=multiply,
    description="Multiplies two numbers: multiply(a: str, b: str)",
)

agent = TinyAgent(llm=llm, memory=memory, tools=tools)

res = agent.run("What is 5.1 times 7.3?")
print(res)
print(agent.memory.get_messages())
