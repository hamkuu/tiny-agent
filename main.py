from agent import TinyAgent
from llm import LLM

llm = LLM(model="gemma4:e4b")

agent = TinyAgent(llm=llm)
response = agent.run("What is 2 + 2?")
print(response)
print(agent.trajectory.runs)
