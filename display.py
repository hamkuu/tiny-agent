from common import Response

BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[32m"  # THOUGHT
RED = "\033[31m"  # ACTION
YELLOW = "\033[33m"  # OBSERVATION
PURPLE = "\033[35m"  # ANSWER


class Display:
    """Chat interface with styling."""

    def __call__(self, event: str, data: str | Response | None = None) -> None:

        # "Thinking" line
        if event == "thinking":
            print(f"  Thinking...\n{RESET}")

        # THOUGHT
        elif event == "response":
            print(f"{BOLD}{GREEN}{'▒▒ THOUGHT ▒▒':<13}{RESET}")
            print(f"{data.reasoning}{RESET}\n")

            # ANSWER
            if data.content:
                print(f"{BOLD}{PURPLE}{'▒▒ ANSWER ▒▒':<13}{RESET}")
                print(f"{data.content}{RESET}\n")

        # ACTION
        elif event == "tool_call" and data:
            tool = data.tool_call["tool"]
            kwargs = data.tool_call["kwargs"]
            print(f"{BOLD}{RED}{'▒▒ ACTION ▒▒':<13}{RESET}")
            print(f"{tool}({kwargs}){RESET}\n")

        # OBSERVATION
        elif event == "observation":
            print(f"{BOLD}{YELLOW}{'▒▒ OBSERVATION ▒▒':<13}{RESET}")
            print(f"{data}{RESET}\n")
            print(f"{'─' * 40}STEP{'─' * 40}{RESET}\n")
