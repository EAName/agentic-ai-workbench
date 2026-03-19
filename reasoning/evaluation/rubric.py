"""
Evaluation Rubric Engine.

Based on Ch 10: Agent reasoning and evaluation.

Evaluation rubrics systematically score agent outputs against defined
criteria. This is essential for:
  - Measuring agent quality over time
  - Comparing different agent configurations
  - Automated prompt testing and selection
  - Detecting regression when changing models or prompts

Rubrics can be LLM-graded (using another LLM as judge) or
rule-based (pattern matching, length checks, etc.).
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from pydantic import BaseModel

from config.models import BaseLLMClient, LLMMessage

logger = structlog.get_logger()


class RubricCriterion(BaseModel):
    """A single evaluation criterion."""

    name: str
    description: str
    weight: float = 1.0
    max_score: int = 10


class EvaluationScore(BaseModel):
    """Score for a single criterion."""

    criterion: str
    score: float
    max_score: int
    explanation: str = ""


class EvaluationResult(BaseModel):
    """Complete evaluation result."""

    scores: list[EvaluationScore]
    total_score: float
    max_possible: float
    percentage: float
    summary: str = ""

    @property
    def passed(self) -> bool:
        """Check if evaluation passed (>= 70%)."""
        return self.percentage >= 70.0


LLM_JUDGE_PROMPT = """You are an expert evaluator. Score the following output against each criterion.

## Task
{task}

## Output to Evaluate
{output}

## Criteria
{criteria}

For each criterion, provide a score and brief explanation.
Respond with JSON only (no markdown fences):
{{
  "scores": [
    {{"criterion": "name", "score": N, "explanation": "..."}},
    ...
  ],
  "summary": "Overall assessment in 1-2 sentences"
}}
"""


class RubricEvaluator:
    """
    Evaluate agent outputs against a rubric.

    Usage:
        evaluator = RubricEvaluator(llm_client)
        evaluator.add_criterion("accuracy", "Factual correctness", weight=2.0)
        evaluator.add_criterion("completeness", "Covers all aspects")
        evaluator.add_criterion("clarity", "Clear and well-structured")

        result = await evaluator.evaluate(
            task="Summarize the Q3 earnings report",
            output=agent_output,
        )
        print(f"Score: {result.percentage:.0f}%")
        if not result.passed:
            print("Agent output needs improvement")
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.criteria: list[RubricCriterion] = []

    def add_criterion(
        self,
        name: str,
        description: str,
        weight: float = 1.0,
        max_score: int = 10,
    ) -> None:
        """Add an evaluation criterion to the rubric."""
        self.criteria.append(
            RubricCriterion(
                name=name,
                description=description,
                weight=weight,
                max_score=max_score,
            )
        )

    async def evaluate(self, task: str, output: str) -> EvaluationResult:
        """Evaluate an output against all criteria using LLM-as-judge."""
        criteria_text = "\n".join(
            f"- {c.name} (weight: {c.weight}x, max: {c.max_score}): {c.description}"
            for c in self.criteria
        )

        prompt = LLM_JUDGE_PROMPT.format(
            task=task,
            output=output,
            criteria=criteria_text,
        )

        response = await self.llm.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.1,
        )

        return self._parse_evaluation(response.content)

    def _parse_evaluation(self, text: str) -> EvaluationResult:
        """Parse LLM judge response into structured scores."""
        try:
            clean = text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
                clean = clean.strip()

            data = json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            logger.warning("evaluation_parse_failed", raw=text[:200])
            # Return default scores
            return EvaluationResult(
                scores=[
                    EvaluationScore(
                        criterion=c.name, score=5, max_score=c.max_score
                    )
                    for c in self.criteria
                ],
                total_score=5.0 * len(self.criteria),
                max_possible=sum(c.max_score * c.weight for c in self.criteria),
                percentage=50.0,
                summary="Evaluation parsing failed; default scores assigned.",
            )

        scores = []
        total_weighted = 0.0
        max_possible = 0.0

        for score_data in data.get("scores", []):
            criterion_name = score_data.get("criterion", "")
            # Find matching criterion for weight
            criterion = next(
                (c for c in self.criteria if c.name == criterion_name), None
            )
            weight = criterion.weight if criterion else 1.0
            max_score = criterion.max_score if criterion else 10

            score_val = min(float(score_data.get("score", 0)), max_score)
            scores.append(
                EvaluationScore(
                    criterion=criterion_name,
                    score=score_val,
                    max_score=max_score,
                    explanation=score_data.get("explanation", ""),
                )
            )
            total_weighted += score_val * weight
            max_possible += max_score * weight

        percentage = (total_weighted / max_possible * 100) if max_possible > 0 else 0

        return EvaluationResult(
            scores=scores,
            total_score=total_weighted,
            max_possible=max_possible,
            percentage=percentage,
            summary=data.get("summary", ""),
        )


# ---------------------------------------------------------------------------
# Preset rubrics for common use cases
# ---------------------------------------------------------------------------

def research_rubric(llm_client: BaseLLMClient) -> RubricEvaluator:
    """Pre-built rubric for evaluating research outputs."""
    evaluator = RubricEvaluator(llm_client)
    evaluator.add_criterion("accuracy", "Claims are factually correct and well-sourced", weight=2.0)
    evaluator.add_criterion("completeness", "All key aspects of the topic are covered", weight=1.5)
    evaluator.add_criterion("structure", "Output is well-organized with clear sections", weight=1.0)
    evaluator.add_criterion("actionability", "Provides specific, actionable recommendations", weight=1.5)
    evaluator.add_criterion("conciseness", "No unnecessary filler; every sentence adds value", weight=1.0)
    return evaluator


def code_review_rubric(llm_client: BaseLLMClient) -> RubricEvaluator:
    """Pre-built rubric for evaluating code review outputs."""
    evaluator = RubricEvaluator(llm_client)
    evaluator.add_criterion("bug_detection", "Identifies actual bugs and logical errors", weight=2.0)
    evaluator.add_criterion("security", "Catches security vulnerabilities and exposure risks", weight=2.0)
    evaluator.add_criterion("suggestions", "Provides specific, correct code fixes", weight=1.5)
    evaluator.add_criterion("prioritization", "Issues are properly categorized by severity", weight=1.0)
    evaluator.add_criterion("false_positives", "Low rate of false alarms (higher = better)", weight=1.0)
    return evaluator
