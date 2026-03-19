"""
Feedback Loop: Self-Improving Agents.

Based on Ch 11: Agent planning and feedback.

Feedback loops allow agents to improve over time by:
  1. Collecting feedback on outputs (automated or human)
  2. Storing feedback alongside the original task and output
  3. Retrieving relevant past feedback when handling similar tasks
  4. Adjusting behavior based on accumulated learnings

This creates a virtuous cycle: better outputs -> better feedback -> even better outputs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel

from config.models import BaseLLMClient, LLMMessage

logger = structlog.get_logger()


class FeedbackEntry(BaseModel):
    """A single feedback record."""

    timestamp: str
    task: str
    output: str
    score: float  # 0-10
    feedback_text: str
    source: str = "automated"  # "automated", "human", "evaluation"
    tags: list[str] = []
    improvements: list[str] = []


class FeedbackStore:
    """
    Persistent feedback storage.

    Stores feedback as JSONL (one JSON object per line) for simplicity
    and append-only performance. For production, replace with a database.
    """

    def __init__(self, store_path: str = "./data/conversations/feedback.jsonl"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, entry: FeedbackEntry) -> None:
        """Append a feedback entry."""
        with open(self.store_path, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    def get_recent(self, n: int = 20) -> list[FeedbackEntry]:
        """Get the N most recent feedback entries."""
        entries = self._load_all()
        return entries[-n:]

    def get_by_tags(self, tags: list[str]) -> list[FeedbackEntry]:
        """Get entries matching any of the given tags."""
        entries = self._load_all()
        return [e for e in entries if any(t in e.tags for t in tags)]

    def get_low_scores(self, threshold: float = 5.0) -> list[FeedbackEntry]:
        """Get entries with scores below threshold (for improvement targeting)."""
        entries = self._load_all()
        return [e for e in entries if e.score < threshold]

    def average_score(self, last_n: int = 50) -> float:
        """Calculate average score over recent entries."""
        entries = self.get_recent(last_n)
        if not entries:
            return 0.0
        return sum(e.score for e in entries) / len(entries)

    def _load_all(self) -> list[FeedbackEntry]:
        """Load all entries from the store."""
        if not self.store_path.exists():
            return []
        entries = []
        with open(self.store_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(FeedbackEntry.model_validate_json(line))
                    except Exception:
                        continue
        return entries


SELF_CRITIQUE_PROMPT = """You are a quality reviewer. Evaluate this agent output.

## Task
{task}

## Agent Output
{output}

Score the output 1-10 and provide specific, actionable improvement suggestions.

Respond with JSON only (no markdown fences):
{{
  "score": N,
  "feedback": "Overall assessment",
  "improvements": [
    "Specific improvement 1",
    "Specific improvement 2"
  ],
  "tags": ["relevant", "category", "tags"]
}}
"""


class FeedbackLoop:
    """
    Complete feedback loop for agent self-improvement.

    Usage:
        feedback = FeedbackLoop(llm_client)

        # After each agent run:
        entry = await feedback.auto_critique(task, output)
        feedback.store.add(entry)

        # When building prompts, inject past learnings:
        learnings = feedback.get_learnings(tags=["research"])
        # -> Append to system prompt as "past lessons learned"

        # Monitor trends:
        avg = feedback.store.average_score()
        low = feedback.store.get_low_scores()
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        store_path: str = "./data/conversations/feedback.jsonl",
    ):
        self.llm = llm_client
        self.store = FeedbackStore(store_path)

    async def auto_critique(self, task: str, output: str) -> FeedbackEntry:
        """Use LLM to automatically critique an agent output."""
        prompt = SELF_CRITIQUE_PROMPT.format(task=task, output=output)

        response = await self.llm.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.2,
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
            data = {"score": 5, "feedback": "Parse error", "improvements": [], "tags": []}

        entry = FeedbackEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            task=task[:500],
            output=output[:1000],
            score=float(data.get("score", 5)),
            feedback_text=data.get("feedback", ""),
            source="automated",
            tags=data.get("tags", []),
            improvements=data.get("improvements", []),
        )

        return entry

    def add_human_feedback(
        self,
        task: str,
        output: str,
        score: float,
        feedback_text: str,
        tags: list[str] | None = None,
    ) -> FeedbackEntry:
        """Record human feedback for an agent output."""
        entry = FeedbackEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            task=task[:500],
            output=output[:1000],
            score=score,
            feedback_text=feedback_text,
            source="human",
            tags=tags or [],
        )
        self.store.add(entry)
        return entry

    def get_learnings(
        self,
        tags: list[str] | None = None,
        last_n: int = 10,
    ) -> str:
        """
        Extract accumulated learnings for prompt injection.

        Returns a formatted string of past improvements that can be
        appended to an agent's system prompt.
        """
        if tags:
            entries = self.store.get_by_tags(tags)[-last_n:]
        else:
            entries = self.store.get_recent(last_n)

        if not entries:
            return ""

        # Extract unique improvements
        all_improvements: list[str] = []
        for entry in entries:
            for imp in entry.improvements:
                if imp not in all_improvements:
                    all_improvements.append(imp)

        if not all_improvements:
            return ""

        parts = ["Based on past experience, keep these lessons in mind:"]
        for imp in all_improvements[-10:]:  # Cap at 10 most recent
            parts.append(f"  - {imp}")

        return "\n".join(parts)
