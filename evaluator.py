import os
import re
from dataclasses import dataclass
from typing import Callable

from agent import TinyAgent
from llm import LLM
from memory import Memory
from planning import NativeReAct
from tools import NativeTools

# Type hint for the scorers
##  (prediction: str, example: dict) -> bool | float
Scorer = Callable[[str, dict], bool | float]


@dataclass
class Benchmark:
    name: str
    examples: list[dict]
    scorer: Callable


def exact_match_scorer(prediction: str, example: dict) -> bool:
    """Return True if the answer matches the prediction, False otherwise"""
    match = re.search(r"\b([A-J])\b", prediction.upper())
    return match.group(1) == example["expected"]


def programmatic_scorer(prediction: str, example: dict) -> bool:
    """Check a prediction against its related check."""
    return example["check"](prediction)


# Judge
judge = LLM(
    model="gemini-3.1-flash-lite",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    api_key=os.environ['GEMINI_API_KEY']
)


def judge_scorer(prediction: str, example: dict) -> bool:
    """The LLM-as-a-judge scorer."""
    prompt = f"""
        Score the response from 0.0 to 1.0.

        Expected: {example["expected"]}
        Response: {prediction}

        Reply with only a single number.
    """
    response = judge.generate([{"role": "user", "content": prompt}])
    score = float(response.content.strip().split()[0])
    return score


# Gemma 4 E4B (with native thinking and tool calling)
llm = LLM(model="gemma4:e4b", think=True)


class Evaluator:
    """Run a TinyAgent over a Benchmark and aggregate the results."""

    def __init__(self, create_agent: Callable):
        """Initialize with a function that creates a new agent instance."""
        self.create_agent = create_agent

    def run(self, benchmark: Benchmark) -> dict:
        """Run the agent on examples in the benchmark and score the results."""

        # Run each example and collect results
        results = []
        for example in benchmark.examples:
            agent = self.create_agent()
            prediction = agent.run(example["task"]) or ""
            passed = benchmark.scorer(prediction, example)
            results.append(
                {
                    "prediction": prediction,
                    "passed": passed,
                }
            )

        # Aggregate pass rate
        if results:
            pass_rate = sum(result["passed"] for result in results) / len(results)
        else:
            pass_rate = 0.0

        # Return detailed results and overall pass rate
        return {
            "name": benchmark.name,
            "pass_rate": pass_rate,
            "results": results,
        }


def create_agent():
    """Create a new instance of TinyAgent"""
    return TinyAgent(
        llm=llm,
        memory=Memory(),
        tools=NativeTools(),
        planner=NativeReAct(),
    )


# Three examples adapted from MMLU Pro
mmlu_pro = Benchmark(
    name="MMLU Pro",
    examples=[
        {
            "task": "Which body cavity contains the pituitary gland?",
            "expected": "the cranial cavity",
        },
        {
            "task": "What is the approximate mean cranial capacity of Homo erectus?",
            "expected": "just under 1000 cc",
        },
        {
            "task": "According to Moore's 'ideal utilitarianism,' the right action is the one that brings about the greatest amount of what?",
            "expected": "good",
        },
    ],
    scorer=judge_scorer,
)

result = Evaluator(create_agent).run(mmlu_pro)
print(result)
