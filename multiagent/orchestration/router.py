"""
Task Router for Multi-Agent Orchestration.

Based on Ch 4: Exploring multi-agent systems.

Routes incoming tasks to the most appropriate agent based on the task
content, agent capabilities, and current agent availability.

Routing strategies:
  - Keyword matching: Route based on task keywords vs agent profiles
  - LLM classification: Use an LLM to classify which agent should handle it
  - Round-robin: Distribute tasks evenly across agents
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from pydantic import BaseModel

from config.models import BaseLLMClient, LLMMessage
from agents.base_agent import AgentProfile

logger = structlog.get_logger()


class RoutingDecision(BaseModel):
    """Result of a routing decision."""

    task: str
    assigned_agent: str
    confidence: float
    reasoning: str = ""


ROUTING_PROMPT = """You are a task router. Given a task and a list of available agents,
decide which agent should handle the task.

Available agents:
{agents}

Task: {task}

Choose the BEST agent for this task. Respond with JSON only (no markdown fences):
{{
  "assigned_agent": "agent_name",
  "confidence": 0.0 to 1.0,
  "reasoning": "Why this agent is the best fit"
}}
"""


class TaskRouter:
    """
    Routes tasks to the most appropriate agent.

    Usage:
        router = TaskRouter(llm_client)
        router.register_agent("researcher", research_profile)
        router.register_agent("engineer", engineer_profile)

        decision = await router.route("Analyze the Q3 earnings data")
        # decision.assigned_agent -> "researcher"
    """

    def __init__(self, llm_client: BaseLLMClient | None = None):
        self.llm = llm_client
        self.agents: dict[str, AgentProfile] = {}

    def register_agent(self, name: str, profile: AgentProfile) -> None:
        """Register an agent as available for routing."""
        self.agents[name] = profile

    async def route(self, task: str) -> RoutingDecision:
        """Route a task to the best agent."""
        if not self.agents:
            raise ValueError("No agents registered for routing")

        if len(self.agents) == 1:
            name = next(iter(self.agents))
            return RoutingDecision(
                task=task,
                assigned_agent=name,
                confidence=1.0,
                reasoning="Only one agent available",
            )

        # Use LLM-based routing if client available
        if self.llm:
            return await self._llm_route(task)

        # Fallback to keyword matching
        return self._keyword_route(task)

    async def _llm_route(self, task: str) -> RoutingDecision:
        """Use LLM to decide which agent handles the task."""
        agents_text = "\n".join(
            f"  - {name}: {profile.role}. Goal: {profile.goal[:100]}"
            for name, profile in self.agents.items()
        )

        prompt = ROUTING_PROMPT.format(agents=agents_text, task=task)
        response = await self.llm.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.1,
        )

        try:
            clean = response.content.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
                clean = clean.strip()
            data = json.loads(clean)

            agent_name = data.get("assigned_agent", "")
            if agent_name not in self.agents:
                # Fallback to first agent
                agent_name = next(iter(self.agents))

            return RoutingDecision(
                task=task,
                assigned_agent=agent_name,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, StopIteration):
            return self._keyword_route(task)

    def _keyword_route(self, task: str) -> RoutingDecision:
        """Simple keyword-based routing fallback."""
        task_lower = task.lower()

        # Score each agent based on keyword overlap
        scores: dict[str, int] = {}
        for name, profile in self.agents.items():
            score = 0
            profile_text = f"{profile.role} {profile.goal} {profile.backstory}".lower()
            for word in task_lower.split():
                if len(word) > 3 and word in profile_text:
                    score += 1
            scores[name] = score

        best = max(scores, key=scores.get)  # type: ignore
        max_score = scores[best]
        total = sum(scores.values()) or 1

        return RoutingDecision(
            task=task,
            assigned_agent=best,
            confidence=max_score / total if total > 0 else 0.5,
            reasoning=f"Keyword match score: {max_score}",
        )
