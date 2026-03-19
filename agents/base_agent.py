"""
Base Agent: The core perception-reasoning-action loop.

Based on Ch 1 (Agent fundamentals), Ch 2 (LLM integration), Ch 3 (Assistants),
and Ch 5 (Empowering agents with actions).

Every agent in this system inherits from BaseAgent. The loop:
  1. PERCEIVE  - Receive input + retrieve relevant memory
  2. REASON    - Think through the problem (via LLM)
  3. ACT       - Execute tools or produce output
  4. OBSERVE   - Capture results and update memory
  5. EVALUATE  - Score the output quality
  6. FEEDBACK  - Store learnings for future improvement
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Callable
from datetime import datetime, timezone

import structlog
from pydantic import BaseModel, Field

from config.models import create_llm_client, BaseLLMClient, LLMMessage, LLMResponse

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Agent Profile (Ch 7, 9: Profiles and Personas)
# ---------------------------------------------------------------------------

class AgentProfile(BaseModel):
    """Defines WHO the agent is. Loaded from YAML."""

    name: str
    role: str
    goal: str
    backstory: str = ""
    constraints: list[str] = []
    tools: list[str] = []
    temperature: float = 0.7
    max_iterations: int = 10
    model: str | None = None
    provider: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentProfile:
        """Load agent profile from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_system_prompt(self) -> str:
        """Convert profile into a system prompt for the LLM."""
        parts = [
            f"You are {self.name}, a {self.role}.",
            f"\nYour primary goal: {self.goal}",
        ]
        if self.backstory:
            parts.append(f"\nBackground: {self.backstory}")
        if self.constraints:
            parts.append("\nConstraints you must follow:")
            for c in self.constraints:
                parts.append(f"  - {c}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool Registry (Ch 5: Empowering agents with actions)
# ---------------------------------------------------------------------------

class ToolResult(BaseModel):
    """Result from executing a tool."""

    tool_name: str
    success: bool
    output: Any
    error: str | None = None
    execution_time_ms: float = 0


class ToolRegistry:
    """Registry of tools available to agents."""

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """Register a tool with its schema."""
        self._tools[name] = func
        self._schemas[name] = {
            "name": name,
            "description": description,
            "input_schema": parameters,
        }

    def get_tool(self, name: str) -> Callable | None:
        return self._tools.get(name)

    def get_schemas(self, tool_names: list[str] | None = None) -> list[dict]:
        """Get tool schemas for LLM function calling."""
        if tool_names:
            return [self._schemas[n] for n in tool_names if n in self._schemas]
        return list(self._schemas.values())

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a registered tool."""
        import time

        func = self._tools.get(name)
        if not func:
            return ToolResult(
                tool_name=name,
                success=False,
                output=None,
                error=f"Tool '{name}' not found",
            )

        start = time.monotonic()
        try:
            import asyncio

            if asyncio.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)

            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_name=name,
                success=True,
                output=result,
                execution_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("tool_execution_failed", tool=name, error=str(e))
            return ToolResult(
                tool_name=name,
                success=False,
                output=None,
                error=str(e),
                execution_time_ms=elapsed,
            )


# Global tool registry
tool_registry = ToolRegistry()


# ---------------------------------------------------------------------------
# Agent Trace (for debugging and evaluation)
# ---------------------------------------------------------------------------

class AgentStep(BaseModel):
    """A single step in the agent's reasoning trace."""

    step_number: int
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    thought: str = ""
    action: str = ""
    action_input: dict[str, Any] = {}
    observation: str = ""
    tool_result: ToolResult | None = None


class AgentTrace(BaseModel):
    """Full trace of an agent's execution for debugging and evaluation."""

    agent_name: str
    task: str
    steps: list[AgentStep] = []
    final_output: str = ""
    total_tokens: int = 0
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str = ""

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Base Agent (the core loop)
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    The foundational agent class. Implements the core loop:
    perceive -> reason -> act -> observe -> evaluate.

    Subclass this to build specialized agents.
    """

    def __init__(
        self,
        profile: AgentProfile,
        llm_client: BaseLLMClient | None = None,
        tools: ToolRegistry | None = None,
    ):
        self.profile = profile
        self.llm = llm_client or create_llm_client(
            provider=profile.provider,
            model=profile.model,
        )
        self.tools = tools or tool_registry
        self.conversation_history: list[LLMMessage] = []
        self.trace: AgentTrace | None = None

    async def run(self, task: str) -> str:
        """
        Execute the full agent loop on a task.

        This is the main entry point. It runs the ReAct-style loop:
        think, decide on an action, execute it, observe results, repeat.
        """
        self.trace = AgentTrace(agent_name=self.profile.name, task=task)

        # Initialize conversation with system prompt + task
        self.conversation_history = [
            LLMMessage(role="system", content=self._build_system_prompt()),
            LLMMessage(role="user", content=task),
        ]

        logger.info(
            "agent_started",
            agent=self.profile.name,
            task=task[:100],
        )

        final_output = ""
        for iteration in range(self.profile.max_iterations):
            # Get tool schemas for function calling
            tool_schemas = self.tools.get_schemas(self.profile.tools or None)

            # REASON: Ask LLM what to do
            response = await self.llm.complete(
                messages=self.conversation_history,
                tools=tool_schemas if tool_schemas else None,
                temperature=self.profile.temperature,
            )

            self.trace.total_tokens += response.usage.get("input_tokens", 0)
            self.trace.total_tokens += response.usage.get("output_tokens", 0)

            # If LLM wants to use a tool: ACT
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    step = AgentStep(
                        step_number=iteration + 1,
                        thought=response.content,
                        action=tool_call["name"],
                        action_input=tool_call["arguments"],
                    )

                    # Execute the tool
                    result = await self.tools.execute(
                        tool_call["name"], tool_call["arguments"]
                    )
                    step.tool_result = result
                    step.observation = str(result.output) if result.success else f"Error: {result.error}"

                    self.trace.add_step(step)

                    # Feed observation back to LLM
                    self.conversation_history.append(
                        LLMMessage(role="assistant", content=response.content)
                    )
                    self.conversation_history.append(
                        LLMMessage(
                            role="user",
                            content=f"Tool result for {tool_call['name']}: {step.observation}",
                        )
                    )

                    logger.info(
                        "agent_tool_call",
                        agent=self.profile.name,
                        tool=tool_call["name"],
                        success=result.success,
                        iteration=iteration + 1,
                    )
            else:
                # No tool call: LLM is providing final answer
                final_output = response.content
                step = AgentStep(
                    step_number=iteration + 1,
                    thought=response.content,
                    observation="Final answer produced",
                )
                self.trace.add_step(step)
                break

        self.trace.final_output = final_output
        self.trace.completed_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "agent_completed",
            agent=self.profile.name,
            iterations=len(self.trace.steps),
            tokens=self.trace.total_tokens,
        )

        return final_output

    def _build_system_prompt(self) -> str:
        """Assemble the full system prompt from profile + tool descriptions."""
        parts = [self.profile.to_system_prompt()]

        # Add available tools description
        if self.profile.tools:
            schemas = self.tools.get_schemas(self.profile.tools)
            if schemas:
                parts.append("\nYou have access to the following tools:")
                for schema in schemas:
                    parts.append(
                        f"  - {schema['name']}: {schema['description']}"
                    )
                parts.append(
                    "\nUse tools when you need external information or to "
                    "take actions. When you have enough information to answer, "
                    "provide your final response directly."
                )

        return "\n".join(parts)
