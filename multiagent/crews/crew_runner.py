"""
Multi-Agent Crew Runner.

Based on Ch 4: Exploring multi-agent systems.

Multi-agent systems enable agents with different specializations to
collaborate on complex tasks. This module provides:
  - Crew definition (which agents, what roles, communication rules)
  - Task routing (directing sub-tasks to the right agent)
  - Result aggregation (combining outputs from multiple agents)
  - Communication protocol (how agents share information)

Patterns supported:
  - Sequential: Agent A -> Agent B -> Agent C (pipeline)
  - Hierarchical: Manager agent delegates to worker agents
  - Collaborative: Agents discuss and critique each other's work
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel

from agents.base_agent import BaseAgent, AgentProfile, ToolRegistry
from config.models import create_llm_client, BaseLLMClient, LLMMessage

logger = structlog.get_logger()


class CrewTask(BaseModel):
    """A task assigned to an agent within a crew."""

    task_id: str
    description: str
    assigned_agent: str
    dependencies: list[str] = []  # task_ids that must complete first
    context: str = ""
    output: str = ""
    status: str = "pending"


class CrewDefinition(BaseModel):
    """Definition of a multi-agent crew."""

    name: str
    description: str
    strategy: str = "sequential"  # sequential, hierarchical, collaborative
    agents: list[str] = []  # profile file paths
    tasks: list[CrewTask] = []

    @classmethod
    def from_yaml(cls, path: str | Path) -> CrewDefinition:
        with open(path) as f:
            data = yaml.safe_load(f)
        tasks = [CrewTask(**t) for t in data.pop("tasks", [])]
        return cls(**data, tasks=tasks)


class CrewResult(BaseModel):
    """Result from a crew execution."""

    crew_name: str
    tasks: list[CrewTask]
    final_output: str = ""
    total_tokens: int = 0


class CrewRunner:
    """
    Orchestrates multi-agent crews.

    Usage:
        runner = CrewRunner()
        runner.add_agent("researcher", AgentProfile.from_yaml("research_analyst.yaml"))
        runner.add_agent("reviewer", AgentProfile.from_yaml("code_reviewer.yaml"))

        result = await runner.run_sequential(
            goal="Research and review the FL SAM2 architecture",
            tasks=[
                {"agent": "researcher", "task": "Research FL SAM2 LoRA approaches"},
                {"agent": "reviewer", "task": "Review the research for accuracy"},
            ]
        )
    """

    def __init__(self, tools: ToolRegistry | None = None):
        self.agents: dict[str, BaseAgent] = {}
        self.tools = tools

    def add_agent(
        self,
        name: str,
        profile: AgentProfile,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        """Register an agent with the crew."""
        client = llm_client or create_llm_client(
            provider=profile.provider, model=profile.model
        )
        self.agents[name] = BaseAgent(
            profile=profile, llm_client=client, tools=self.tools
        )
        logger.info("crew_agent_added", name=name, role=profile.role)

    async def run_sequential(
        self,
        goal: str,
        tasks: list[dict[str, str]],
    ) -> CrewResult:
        """
        Run tasks sequentially, passing each output as context to the next.

        Args:
            goal: The overall crew objective.
            tasks: List of dicts with "agent" and "task" keys.
        """
        crew_result = CrewResult(crew_name="sequential_crew", tasks=[])
        accumulated_context = f"Overall goal: {goal}\n\n"

        for i, task_def in enumerate(tasks):
            agent_name = task_def["agent"]
            task_description = task_def["task"]

            agent = self.agents.get(agent_name)
            if not agent:
                logger.error("agent_not_found", name=agent_name)
                continue

            # Inject context from previous agents
            full_task = f"{accumulated_context}Your task: {task_description}"

            logger.info(
                "crew_task_started",
                agent=agent_name,
                task=task_description[:60],
                step=i + 1,
            )

            output = await agent.run(full_task)

            crew_task = CrewTask(
                task_id=f"task_{i+1}",
                description=task_description,
                assigned_agent=agent_name,
                output=output,
                status="completed",
            )
            crew_result.tasks.append(crew_task)

            # Accumulate context for next agent
            accumulated_context += (
                f"--- Output from {agent_name} ---\n{output}\n\n"
            )

            if agent.trace:
                crew_result.total_tokens += agent.trace.total_tokens

        crew_result.final_output = crew_result.tasks[-1].output if crew_result.tasks else ""
        return crew_result

    async def run_hierarchical(
        self,
        goal: str,
        manager_agent: str,
        worker_agents: list[str],
    ) -> CrewResult:
        """
        Manager agent decomposes the goal and delegates to workers.

        The manager decides which worker handles each sub-task and
        synthesizes all worker outputs into a final deliverable.
        """
        manager = self.agents.get(manager_agent)
        if not manager:
            raise ValueError(f"Manager agent '{manager_agent}' not found")

        # Step 1: Manager decomposes the goal
        decomposition_prompt = (
            f"You are managing a team of agents: {', '.join(worker_agents)}.\n"
            f"Goal: {goal}\n\n"
            f"Decompose this goal into sub-tasks, one per agent. "
            f"Respond with a list of assignments:\n"
            f"Agent: [name] -> Task: [description]\n"
        )

        plan_output = await manager.run(decomposition_prompt)

        # Step 2: Execute worker tasks in parallel
        import asyncio

        worker_tasks = []
        for worker_name in worker_agents:
            worker = self.agents.get(worker_name)
            if worker:
                worker_task = (
                    f"Context from manager:\n{plan_output}\n\n"
                    f"You are {worker_name}. Complete your assigned portion of: {goal}"
                )
                worker_tasks.append((worker_name, worker.run(worker_task)))

        worker_results = {}
        for name, coro in worker_tasks:
            worker_results[name] = await coro

        # Step 3: Manager synthesizes
        synthesis_prompt = (
            f"Goal: {goal}\n\n"
            f"Worker outputs:\n"
        )
        for name, output in worker_results.items():
            synthesis_prompt += f"\n--- {name} ---\n{output}\n"
        synthesis_prompt += "\nSynthesize these into a cohesive final deliverable."

        final_output = await manager.run(synthesis_prompt)

        crew_result = CrewResult(
            crew_name="hierarchical_crew",
            tasks=[
                CrewTask(
                    task_id=f"worker_{name}",
                    description=f"Worker task for {name}",
                    assigned_agent=name,
                    output=output,
                    status="completed",
                )
                for name, output in worker_results.items()
            ],
            final_output=final_output,
        )

        return crew_result
