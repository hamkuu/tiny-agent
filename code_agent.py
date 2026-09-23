from agent import TinyAgent
from display import Display
from llm import LLM
from memory import Memory
from planning import NativeReAct
from toolbox import execute_python, list_files, read_file, write_file
from tools import NativeTools

# Gemma 4 E4B (with native thinking and tool calling)
llm = LLM(model="gemma4:e4b", think=True)

tools = NativeTools(requires_approval=["write_file", "execute_python"])
tools.add_tool("read_file", read_file)
tools.add_tool("list_files", list_files)
tools.add_tool("write_file", write_file)
tools.add_tool("execute_python", execute_python)

memory = Memory()

react = NativeReAct(max_steps=10)

display = Display()

agent = TinyAgent(llm=llm, tools=tools, memory=Memory(), planner=react, display=display)

NAME = """
██████ ██ ███  ██ ██  ██   ▄████▄  ▄████  ██████ ███  ██ ██████
  ██   ██ ██ ▀▄██ ▀██▀    ██▄▄██ ██  ▄▄▄ ██▄▄   ██ ▀▄██  ██
  ██   ██ ██   ██   ██     ██  ██  ▀███▀  ██▄▄▄▄ ██   ██  ██
"""  # ANSI Compact


def main():

    print(f"\n{NAME}\n")

    while True:
        try:
            query = input().strip()
            print("\033[0m")
        except (KeyboardInterrupt, EOFError):
            print("\033[0m", end="")
            break
        if not query or query.lower() in ("exit", "quit"):
            break
        try:
            agent.run(query)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}\n")


if __name__ == "__main__":
    main()
