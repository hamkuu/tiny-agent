from common import Response
from display import Display
from llm import LLM
from toolbox import execute_python, list_files, read_file, write_file
from tools import NativeTools

# Gemma 4 E4B (with native thinking and tool calling)
llm = LLM(model="gemma4:e4b", think=True)

tools = NativeTools(requires_approval=["write_file", "execute_python"])
tools.add_tool("read_file", read_file)
tools.add_tool("list_files", list_files)
tools.add_tool("write_file", write_file)
tools.add_tool("execute_python", execute_python)

response = Response(
    content="I executed Python.",
    reasoning="Let's execute some python!",
    tool_call={"tool": "execute_python", "kwargs": {"code": "print('Hello World!')"}},
)

# Display the Response
display = Display()
display("thinking", response)
display("response", response)
display("tool_call", response)
display("observation", "Hello World!")
