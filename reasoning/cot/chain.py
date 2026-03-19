"""
Chain of Thought (CoT) and Tree of Thought (ToT) reasoning.

Based on Ch 10: Agent reasoning and evaluation.

CoT: Forces the LLM to show its work step by step before answering.
     Dramatically improves accuracy on complex reasoning tasks.

ToT: Explores multiple reasoning paths in parallel, then selects
     the best one. Like CoT but with branching and backtracking.

Self-Consistency: Run CoT multiple times, take the majority answer.
"""

from __future__ import annotations

import json
import asyncio
from typing import Any

import structlog
from pydantic import BaseModel

from config.models import BaseLLMClient, LLMMessage

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Chain of Thought
# ---------------------------------------------------------------------------

COT_SYSTEM_PROMPT = """You are an expert analyst. When solving problems:

1. Break the problem into clear steps
2. Work through each step explicitly, showing your reasoning
3. State any assumptions you are making
4. Check your work before giving a final answer
5. Format your response as:

## Step-by-Step Reasoning
[Your detailed reasoning here]

## Final Answer
[Your concise final answer here]
"""


class CoTResult(BaseModel):
    """Result from Chain of Thought reasoning."""

    question: str
    reasoning: str
    final_answer: str
    model: str = ""


class ChainOfThought:
    """
    Chain of Thought prompting.

    Forces explicit step-by-step reasoning before the final answer.
    Best for: math, logic, multi-step analysis, complex comparisons.

    Usage:
        cot = ChainOfThought(llm_client)
        result = await cot.reason("Compare the risk profiles of these two investments...")
        print(result.reasoning)
        print(result.final_answer)
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        system_prompt: str | None = None,
    ):
        self.llm = llm_client
        self.system_prompt = system_prompt or COT_SYSTEM_PROMPT

    async def reason(
        self,
        question: str,
        context: str = "",
        temperature: float = 0.3,
    ) -> CoTResult:
        """Run Chain of Thought reasoning on a question."""
        user_content = question
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{question}"

        messages = [
            LLMMessage(role="system", content=self.system_prompt),
            LLMMessage(role="user", content=user_content),
        ]

        response = await self.llm.complete(
            messages=messages, temperature=temperature
        )

        reasoning, answer = self._parse_response(response.content)

        return CoTResult(
            question=question,
            reasoning=reasoning,
            final_answer=answer,
            model=response.model,
        )

    def _parse_response(self, text: str) -> tuple[str, str]:
        """Split response into reasoning and final answer."""
        if "## Final Answer" in text:
            parts = text.split("## Final Answer", 1)
            reasoning = parts[0].replace("## Step-by-Step Reasoning", "").strip()
            answer = parts[1].strip()
        elif "Final Answer:" in text:
            parts = text.split("Final Answer:", 1)
            reasoning = parts[0].strip()
            answer = parts[1].strip()
        else:
            # No clear separation; treat the whole thing as reasoning + answer
            reasoning = text
            answer = text.split("\n")[-1].strip()

        return reasoning, answer


# ---------------------------------------------------------------------------
# Self-Consistency (multiple CoT runs, majority vote)
# ---------------------------------------------------------------------------

class SelfConsistencyResult(BaseModel):
    """Result from self-consistency voting."""

    question: str
    individual_answers: list[str]
    consensus_answer: str
    agreement_ratio: float
    all_reasoning: list[str]


class SelfConsistency:
    """
    Self-Consistency: Run CoT multiple times with higher temperature,
    then take the most common answer.

    This reduces variance and catches reasoning errors.
    Best for: factual questions, numerical answers, classification tasks.

    Usage:
        sc = SelfConsistency(llm_client, num_samples=5)
        result = await sc.reason("What is the market cap of...")
        print(f"Answer: {result.consensus_answer} ({result.agreement_ratio:.0%} agreement)")
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        num_samples: int = 5,
        temperature: float = 0.7,
    ):
        self.cot = ChainOfThought(llm_client)
        self.num_samples = num_samples
        self.temperature = temperature

    async def reason(self, question: str, context: str = "") -> SelfConsistencyResult:
        """Run multiple CoT samples and vote on the answer."""
        tasks = [
            self.cot.reason(question, context, temperature=self.temperature)
            for _ in range(self.num_samples)
        ]

        results = await asyncio.gather(*tasks)

        answers = [r.final_answer for r in results]
        reasoning = [r.reasoning for r in results]

        # Find most common answer (simple majority)
        from collections import Counter

        counter = Counter(answers)
        consensus, count = counter.most_common(1)[0]
        agreement = count / len(answers)

        return SelfConsistencyResult(
            question=question,
            individual_answers=answers,
            consensus_answer=consensus,
            agreement_ratio=agreement,
            all_reasoning=reasoning,
        )


# ---------------------------------------------------------------------------
# Tree of Thought
# ---------------------------------------------------------------------------

TOT_EVALUATE_PROMPT = """Evaluate this reasoning path for the given problem.

Problem: {problem}

Reasoning Path:
{reasoning}

Rate this path on a scale of 1-10 for:
- Correctness: Is the logic sound?
- Completeness: Does it address all aspects?
- Clarity: Is the reasoning clear?

Respond with JSON only:
{{"correctness": N, "completeness": N, "clarity": N, "total": N, "should_continue": true/false}}
"""


class ThoughtNode(BaseModel):
    """A single node in the thought tree."""

    thought: str
    score: float = 0.0
    children: list[ThoughtNode] = []
    is_terminal: bool = False


class ToTResult(BaseModel):
    """Result from Tree of Thought exploration."""

    question: str
    best_path: list[str]
    best_score: float
    final_answer: str
    paths_explored: int


class TreeOfThought:
    """
    Tree of Thought: Explore multiple reasoning branches.

    Generates several possible next steps at each point, evaluates
    them, and continues exploring the most promising branches.

    Best for: creative problem-solving, strategy, complex planning.

    Usage:
        tot = TreeOfThought(llm_client, branches=3, depth=3)
        result = await tot.reason("Design an architecture for...")
        print(result.final_answer)
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        branches: int = 3,
        depth: int = 3,
    ):
        self.llm = llm_client
        self.branches = branches
        self.depth = depth

    async def reason(self, question: str) -> ToTResult:
        """Explore the thought tree and return the best path."""
        root_thoughts = await self._generate_thoughts(question, [])

        best_path: list[str] = []
        best_score: float = 0.0
        paths_explored = 0

        for thought in root_thoughts:
            path, score = await self._explore(question, [thought], 1)
            paths_explored += 1
            if score > best_score:
                best_score = score
                best_path = path

        # Generate final answer from best path
        final_answer = await self._synthesize(question, best_path)

        return ToTResult(
            question=question,
            best_path=best_path,
            best_score=best_score,
            final_answer=final_answer,
            paths_explored=paths_explored,
        )

    async def _explore(
        self,
        question: str,
        path: list[str],
        current_depth: int,
    ) -> tuple[list[str], float]:
        """Recursively explore a reasoning path."""
        if current_depth >= self.depth:
            score = await self._evaluate_path(question, path)
            return path, score

        # Check if current path is worth continuing
        score = await self._evaluate_path(question, path)
        if score < 3.0:  # Prune low-scoring paths
            return path, score

        # Generate next thoughts
        next_thoughts = await self._generate_thoughts(question, path)

        best_path = path
        best_score = score

        for thought in next_thoughts:
            new_path = path + [thought]
            result_path, result_score = await self._explore(
                question, new_path, current_depth + 1
            )
            if result_score > best_score:
                best_score = result_score
                best_path = result_path

        return best_path, best_score

    async def _generate_thoughts(
        self, question: str, path: list[str]
    ) -> list[str]:
        """Generate possible next thoughts given the current path."""
        path_text = "\n".join(f"Step {i+1}: {t}" for i, t in enumerate(path))
        prompt = (
            f"Problem: {question}\n\n"
            f"Reasoning so far:\n{path_text}\n\n"
            f"Generate {self.branches} distinct possible next steps. "
            f"Each should explore a different angle or approach.\n"
            f"Format: one step per line, numbered 1-{self.branches}."
        )

        response = await self.llm.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.8,
        )

        # Parse numbered lines
        thoughts = []
        for line in response.content.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # Remove numbering prefix
                text = line.lstrip("0123456789.)- ").strip()
                if text:
                    thoughts.append(text)

        return thoughts[: self.branches]

    async def _evaluate_path(self, question: str, path: list[str]) -> float:
        """Score a reasoning path."""
        path_text = "\n".join(f"Step {i+1}: {t}" for i, t in enumerate(path))
        prompt = TOT_EVALUATE_PROMPT.format(problem=question, reasoning=path_text)

        response = await self.llm.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.1,
        )

        try:
            # Try to parse JSON from response
            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            scores = json.loads(text)
            return float(scores.get("total", 0))
        except (json.JSONDecodeError, ValueError, KeyError):
            return 5.0  # Default middle score if parsing fails

    async def _synthesize(self, question: str, path: list[str]) -> str:
        """Synthesize a final answer from the best reasoning path."""
        path_text = "\n".join(f"Step {i+1}: {t}" for i, t in enumerate(path))
        prompt = (
            f"Problem: {question}\n\n"
            f"Best reasoning path:\n{path_text}\n\n"
            f"Based on this reasoning, provide a clear, concise final answer."
        )

        response = await self.llm.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.3,
        )

        return response.content
