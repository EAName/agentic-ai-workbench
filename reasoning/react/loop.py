"""
ReAct Loop: Reasoning + Acting.

Based on Ch 10: Agent reasoning and evaluation.

ReAct (Reason + Act) interleaves reasoning traces with actions:
  1. THOUGHT     - The agent reasons about what to do next
  2. ACTION      - The agent selects and executes a tool
  3. OBSERVATION - The agent observes the tool's output
  4. (repeat until the agent has enough info to answer)
  5. ANSWER      - The agent produces a final response

This is the most fundamental reasoning pattern for tool-using agents.
It makes the agent's decision process transparent and debuggable.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import BaseModel

from config.models import BaseLLMClient, LLMMessage, LLMResponse
from agents.base_agent import ToolRegistry, ToolResult

logger = structlog.get_logger()


REACT_SYSTEM_PROMPT = """You are an AI assistant that uses the ReAct framework to solve problems.

For each step, you must output your reasoning in this exact format:

Thought: [Your reasoning about what to do next]
Action: [The tool name to use, or "FINISH" if you have the answer]
Action Input: [JSON arguments for the tool, or your final answer if Action is FINISH]

Rules:
- Always start with a Thought
- If you need more information, use a tool
- When you have enough information, use Action: FINISH
- Be concise in your thoughts but thorough in your analysis
- If a tool fails, reason about alternatives
"""


class ReActStep(BaseModel):
    """A single step in the ReAct loop."""

    step: int
    thought: str
    action: str
    action_input: Any
    observation: str = ""


class ReActResult(BaseModel):
    """Full result of a ReAct execution."""

    question: str
    steps: list[ReActStep] = []
    final_answer: str = ""
    total_steps: int = 0


class ReActLoop:
    """
    The ReAct reasoning loop.

    Usage:
        react = ReActLoop(llm_client=client, tools=registry)
        result = await react.run("What are the top 3 risks in this dataset?")
        print(result.final_answer)
        for step in result.steps:
            print(f"Step {step.step}: {step.thought}")
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        tools: ToolRegistry,
        max_steps: int = 10,
        system_prompt: str | None = None,
    ):
        self.llm = llm_client
        self.tools = tools
        self.max_steps = max_steps
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    async def run(self, question: str) -> ReActResult:
        """Execute the ReAct loop on a question."""
        result = ReActResult(question=question)

        messages = [
            LLMMessage(role="system", content=self._build_prompt()),
            LLMMessage(role="user", content=question),
        ]

        for step_num in range(1, self.max_steps + 1):
            response = await self.llm.complete(messages=messages, temperature=0.2)

            # Parse the ReAct format from the response
            thought, action, action_input = self._parse_response(response.content)

            step = ReActStep(
                step=step_num,
                thought=thought,
                action=action,
                action_input=action_input,
            )

            logger.info(
                "react_step",
                step=step_num,
                thought=thought[:80],
                action=action,
            )

            # Check if agent wants to finish
            if action.upper() == "FINISH":
                result.final_answer = (
                    action_input if isinstance(action_input, str) else str(action_input)
                )
                result.steps.append(step)
                break

            # Execute the tool
            tool_args = action_input if isinstance(action_input, dict) else {}
            tool_result = await self.tools.execute(action, tool_args)

            observation = (
                str(tool_result.output) if tool_result.success
                else f"Error: {tool_result.error}"
            )
            step.observation = observation
            result.steps.append(step)

            # Feed observation back
            messages.append(LLMMessage(role="assistant", content=response.content))
            messages.append(
                LLMMessage(role="user", content=f"Observation: {observation}")
            )

        result.total_steps = len(result.steps)
        return result

    def _build_prompt(self) -> str:
        """Build system prompt with available tool descriptions."""
        parts = [self.system_prompt]

        schemas = self.tools.get_schemas()
        if schemas:
            parts.append("\nAvailable tools:")
            for schema in schemas:
                parts.append(f"  - {schema['name']}: {schema['description']}")

        return "\n".join(parts)

    def _parse_response(self, text: str) -> tuple[str, str, Any]:
        """
        Parse ReAct-formatted response into components.

        Expected format:
            Thought: ...
            Action: ...
            Action Input: ...
        """
        import json

        thought = ""
        action = ""
        action_input: Any = ""

        lines = text.strip().split("\n")
        current_section = None

        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("Thought:"):
                current_section = "thought"
                thought = line_stripped[len("Thought:"):].strip()
            elif line_stripped.startswith("Action:"):
                current_section = "action"
                action = line_stripped[len("Action:"):].strip()
            elif line_stripped.startswith("Action Input:"):
                current_section = "action_input"
                raw = line_stripped[len("Action Input:"):].strip()
                try:
                    action_input = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    action_input = raw
            elif current_section == "thought":
                thought += " " + line_stripped
            elif current_section == "action_input":
                raw_continued = action_input if isinstance(action_input, str) else ""
                raw_continued += " " + line_stripped
                try:
                    action_input = json.loads(raw_continued.strip())
                except (json.JSONDecodeError, ValueError):
                    action_input = raw_continued.strip()

        return thought.strip(), action.strip(), action_input
