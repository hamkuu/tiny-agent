import inspect
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path


def add(a: str, b: str) -> float:
    return float(a) + float(b)


def subtract(a: str, b: str) -> float:
    return float(a) - float(b)


def multiply(a: str, b: str) -> float:
    return float(a) * float(b)


# Convert specific types to string descriptions
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool_to_schema(function: Callable) -> dict:
    """Convert a Python function to an OpenAI-style tool schema."""
    signature = inspect.signature(function)

    # Extract metadata
    properties, required = {}, []
    for name, parameter in signature.parameters.items():
        properties[name] = {"type": TYPE_MAP.get(parameter.annotation, "string")}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    # Fill schema
    schema = {
        "type": "function",
        "function": {
            "name": function.__name__,
            "description": inspect.getdoc(function),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

    return schema


def read_file(path: str) -> str:
    """Read a file's contents."""
    target = Path(path)
    if not target.exists():
        return f"Error: '{path}' not found."
    return target.read_text(encoding="utf-8")


def list_files(directory: str = ".") -> str:
    """List files in a directory."""
    target = Path(directory)
    if not target.is_dir():
        return f"Error: '{directory}' is not a directory."
    entries = sorted(target.iterdir())
    lines = [f"{p.name}/" if p.is_dir() else p.name for p in entries]
    return "\n".join(lines) or "(empty)"


def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Written to '{path}'."


def execute_python(code: str) -> str:
    """Execute Python code and return output."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (30s limit)."

    if result.returncode != 0:
        return (
            f"Exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        ).strip()
    return result.stdout.strip() or "(no output)"
