import re

from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Generate, TaskState, solver

from sycophancy_eval.prompts import PRESSURE_TEMPLATES, SYSTEM_PROMPT

MAX_TOKENS = 300

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# A truncated response can open a think block and never close it.
UNCLOSED_THINK = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    text = THINK_BLOCK.sub("", text or "")
    text = UNCLOSED_THINK.sub("", text)
    return text.strip()


def is_empty(text: str) -> bool:
    return strip_think(text) == ""


@solver
def two_turn_pressure(pressure_type: str):
    template = PRESSURE_TEMPLATES[pressure_type]

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        wrong_answer = state.metadata["wrong_answer"]
        state.metadata["pressure_type"] = pressure_type
        state.metadata["empty_response"] = False

        state.messages.insert(0, ChatMessageSystem(content=SYSTEM_PROMPT))

        state = await generate(state, max_tokens=MAX_TOKENS)
        initial = strip_think(state.output.completion)
        state.metadata["initial_response"] = initial
        if not initial:
            state.metadata["empty_response"] = True

        state.messages.append(ChatMessageUser(content=template.format(wrong_answer=wrong_answer)))

        state = await generate(state, max_tokens=MAX_TOKENS)
        final = strip_think(state.output.completion)
        state.metadata["final_response"] = final
        if not final:
            state.metadata["empty_response"] = True

        return state

    return solve
