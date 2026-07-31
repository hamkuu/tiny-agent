from agent import TinyAgent
from llm import LLM
from memory import Memory

llm = LLM(model="gemma4:e4b")
memory = Memory()

agent = TinyAgent(llm=llm, memory=memory)

res = agent.run("My name is Hiro Shaw")
print(res)

res = agent.run("What is my name?")
print(res)

print(agent.trajectory.runs)
