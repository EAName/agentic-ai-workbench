"""
Sequential and Stepwise Planners.

Based on Ch 11: Agent planning and feedback.

Planning enables agents to decompose complex tasks into ordered steps,
execute them sequentially, handle failures, and adapt the plan mid-flight.

Sequential Planner: Generates a full plan upfront, then executes step by step.
Stepwise Planner: Generates one step at a time based on current state (adaptive).
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from pydantic import BaseModel

from config.models import BaseLLMClient, LLMMessage
from agents.base_agent import ToolRegistry

logger = structlog.get_logger()


class PlanStep(BaseModel):
    """A single step in a plan."""

    step_number: int
    description: str
    tool: str = ""
    tool_input: dict[str, Any] = {}
    expected_output: str = ""
    status: str = "pending"  # pending, running, completed, failed, skipped
    actual_output: str = ""
    error: str = ""


class Plan(BaseModel):
    """A complete execution plan."""

    goal: str
    steps: list[PlanStep] = []
    current_step: int = 0
    status: str = "created"  # created, running, completed, failed

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == "completed")
        return completed / len(self.steps) * 100

    @property
    def completed_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == "completed"]


PLAN_GENERATION_PROMPT = """You are a planning agent. Given a goal and available tools, create a step-by-step plan.

Available tools:
{tools}

Goal: {goal}

Create a plan with clear, executable steps. Each step should specify:
- What to do (description)
- Which tool to use (if any)
- What input the tool needs
- What output to expect

Respond with JSON only (no markdown fences):
{{
  "steps": [
    {{
      "step_number": 1,
      "description": "...",
      "tool": "tool_name or empty string",
      "tool_input": {{}},
      "expected_output": "..."
    }}
  ]
}}
"""


class SequentialPlanner:
    """
    Generates a complete plan upfront, then executes it step by step.

    Best for: well-defined tasks where the full path is knowable in advance.
    Example: data pipeline execution, report generation, deployment checklists.

    Usage:
        planner = SequentialPlanner(llm_client, tool_registry)
        plan = await planner.create_plan("Analyze the Q3 sales data and generate a report")
        result = await planner.execute_plan(plan)
    """

    def __init__(self, llm_client: BaseLLMClient, tools: ToolRegistry):
        self.llm = llm_client
        self.tools = tools

    async def create_plan(self, goal: str) -> Plan:
        """Generate a plan for achieving the goal."""
        schemas = self.tools.get_schemas()
        tools_text = "\n".join(
            f"  - {s['name']}: {s['description']}" for s in schemas
        )

        prompt = PLAN_GENERATION_PROMPT.format(tools=tools_text, goal=goal)

        response = await self.llm.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.3,
        )

        plan = Plan(goal=goal)

        try:
            clean = response.content.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
                clean = clean.strip()

            data = json.loads(clean)
            for step_data in data.get("steps", []):
                plan.steps.append(PlanStep(**step_data))
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("plan_parse_failed", error=str(e))
            # Create a single-step fallback plan
            plan.steps.append(
                PlanStep(
                    step_number=1,
                    description=f"Directly address: {goal}",
                    expected_output="Task result",
                )
            )

        logger.info("plan_created", goal=goal[:80], steps=len(plan.steps))
        return plan

    async def execute_plan(self, plan: Plan) -> Plan:
        """Execute a plan step by step."""
        plan.status = "running"

        for step in plan.steps:
            plan.current_step = step.step_number
            step.status = "running"

            logger.info(
                "executing_step",
                step=step.step_number,
                description=step.description[:60],
                tool=step.tool,
            )

            if step.tool and step.tool.strip():
                # Execute tool
                result = await self.tools.execute(step.tool, step.tool_input)
                if result.success:
                    step.actual_output = str(result.output)
                    step.status = "completed"
                else:
                    step.error = result.error or "Unknown error"
                    step.status = "failed"
                    logger.error(
                        "step_failed",
                        step=step.step_number,
                        error=step.error,
                    )
                    # Attempt recovery
                    recovered = await self._recover_step(plan, step)
                    if not recovered:
                        plan.status = "failed"
                        return plan
            else:
                # No tool; use LLM to complete the step
                context = self._build_context(plan, step)
                response = await self.llm.complete(
                    messages=[LLMMessage(role="user", content=context)],
                    temperature=0.3,
                )
                step.actual_output = response.content
                step.status = "completed"

        plan.status = "completed"
        logger.info("plan_completed", goal=plan.goal[:80], steps=len(plan.steps))
        return plan

    async def _recover_step(self, plan: Plan, failed_step: PlanStep) -> bool:
        """Attempt to recover from a failed step by re-planning."""
        prompt = (
            f"Step {failed_step.step_number} failed: {failed_step.description}\n"
            f"Error: {failed_step.error}\n\n"
            f"Suggest an alternative approach or indicate if the plan should abort.\n"
            f"Respond with JSON: {{\"action\": \"retry|skip|abort\", \"alternative\": \"...\"}}"
        )

        response = await self.llm.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.2,
        )

        try:
            data = json.loads(response.content.strip())
            action = data.get("action", "abort")
            if action == "skip":
                failed_step.status = "skipped"
                return True
            elif action == "retry":
                # Could implement retry logic here
                return False
        except json.JSONDecodeError:
            pass

        return False

    def _build_context(self, plan: Plan, current_step: PlanStep) -> str:
        """Build context from completed steps for the current step."""
        parts = [f"Goal: {plan.goal}\n"]
        parts.append("Completed steps:")
        for step in plan.completed_steps:
            parts.append(
                f"  Step {step.step_number}: {step.description}\n"
                f"  Result: {step.actual_output[:200]}"
            )
        parts.append(
            f"\nNow execute step {current_step.step_number}: {current_step.description}"
        )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Stepwise Planner (adaptive, one step at a time)
# ---------------------------------------------------------------------------

STEPWISE_PROMPT = """You are an adaptive planning agent. Given the current state, decide the NEXT single step.

Goal: {goal}

Completed steps:
{history}

Current state:
{state}

Available tools:
{tools}

Decide the next action. Respond with JSON (no markdown fences):
{{
  "thought": "Why I am choosing this next step",
  "description": "What to do",
  "tool": "tool_name or empty string",
  "tool_input": {{}},
  "is_final": false
}}

Set is_final to true if the goal has been achieved.
"""


class StepwisePlanner:
    """
    Generates one step at a time based on current state.

    Best for: exploratory tasks, research, debugging, anything where
    the next step depends on what you just learned.

    Usage:
        planner = StepwisePlanner(llm_client, tool_registry)
        plan = await planner.run("Investigate why the pipeline is failing")
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        tools: ToolRegistry,
        max_steps: int = 15,
    ):
        self.llm = llm_client
        self.tools = tools
        self.max_steps = max_steps

    async def run(self, goal: str) -> Plan:
        """Execute the stepwise planning loop."""
        plan = Plan(goal=goal, status="running")
        state: dict[str, Any] = {}

        schemas = self.tools.get_schemas()
        tools_text = "\n".join(
            f"  - {s['name']}: {s['description']}" for s in schemas
        )

        for step_num in range(1, self.max_steps + 1):
            history_text = "\n".join(
                f"  {s.step_number}. {s.description} -> {s.actual_output[:100]}"
                for s in plan.completed_steps
            ) or "  (none yet)"

            prompt = STEPWISE_PROMPT.format(
                goal=goal,
                history=history_text,
                state=json.dumps(state, default=str)[:500],
                tools=tools_text,
            )

            response = await self.llm.complete(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.3,
            )

            try:
                clean = response.content.strip()
                if clean.startswith("```"):
                    clean = clean.split("```")[1]
                    if clean.startswith("json"):
                        clean = clean[4:]
                    clean = clean.strip()
                data = json.loads(clean)
            except json.JSONDecodeError:
                logger.warning("stepwise_parse_failed", step=step_num)
                break

            step = PlanStep(
                step_number=step_num,
                description=data.get("description", ""),
                tool=data.get("tool", ""),
                tool_input=data.get("tool_input", {}),
            )

            # Check if goal is achieved
            if data.get("is_final", False):
                step.status = "completed"
                step.actual_output = "Goal achieved"
                plan.steps.append(step)
                break

            # Execute the step
            if step.tool and step.tool.strip():
                result = await self.tools.execute(step.tool, step.tool_input)
                step.actual_output = str(result.output) if result.success else f"Error: {result.error}"
                step.status = "completed" if result.success else "failed"
                state[f"step_{step_num}_result"] = step.actual_output[:200]
            else:
                step.actual_output = data.get("description", "")
                step.status = "completed"

            plan.steps.append(step)

        plan.status = "completed"
        return plan
