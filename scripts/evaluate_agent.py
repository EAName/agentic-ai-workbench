"""
Agent Evaluation Runner.

Run evaluation rubrics against agent outputs to measure quality.

Usage:
    python scripts/evaluate_agent.py --profile agents/profiles/research_analyst.yaml \
        --task "Analyze the fusion energy market" --rubric research
"""

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.base_agent import BaseAgent, AgentProfile
from reasoning.evaluation.rubric import (
    RubricEvaluator,
    research_rubric,
    code_review_rubric,
)
from planning.feedback.loop import FeedbackLoop
from config.models import create_llm_client

# Import tools
from agents.tools import file_ops  # noqa: F401


RUBRIC_PRESETS = {
    "research": research_rubric,
    "code_review": code_review_rubric,
}


def resolve_profile_path(profile_path: str) -> Path:
    """Resolve profile path from cwd or project root."""
    path = Path(profile_path).expanduser()
    if path.exists():
        return path
    root_candidate = PROJECT_ROOT / profile_path
    if root_candidate.exists():
        return root_candidate
    return path


async def main():
    parser = argparse.ArgumentParser(description="Evaluate agent output quality")
    parser.add_argument("--profile", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument(
        "--rubric",
        type=str,
        default="research",
        choices=list(RUBRIC_PRESETS.keys()),
    )
    parser.add_argument("--feedback", action="store_true", help="Also run feedback loop")
    args = parser.parse_args()

    # Load and run agent
    profile = AgentProfile.from_yaml(resolve_profile_path(args.profile))
    agent = BaseAgent(profile=profile)

    print(f"Running agent: {profile.name}")
    print(f"Task: {args.task}\n")

    output = await agent.run(args.task)

    print("--- Agent Output ---")
    print(output[:500])
    if len(output) > 500:
        print(f"... ({len(output)} total characters)")
    print()

    # Evaluate
    llm = create_llm_client()
    evaluator = RUBRIC_PRESETS[args.rubric](llm)

    print(f"Evaluating with '{args.rubric}' rubric...")
    result = await evaluator.evaluate(task=args.task, output=output)

    print(f"\n--- Evaluation Results ---")
    print(f"Overall: {result.percentage:.1f}% {'PASS' if result.passed else 'FAIL'}")
    print(f"Summary: {result.summary}\n")

    for score in result.scores:
        bar = "#" * int(score.score) + "." * (score.max_score - int(score.score))
        print(f"  {score.criterion:20s} [{bar}] {score.score}/{score.max_score}")
        if score.explanation:
            print(f"    {score.explanation}")

    # Optionally run feedback loop
    if args.feedback:
        print("\n--- Feedback Loop ---")
        feedback = FeedbackLoop(llm)
        entry = await feedback.auto_critique(args.task, output)
        feedback.store.add(entry)

        print(f"Score: {entry.score}/10")
        print(f"Feedback: {entry.feedback_text}")
        if entry.improvements:
            print("Improvements:")
            for imp in entry.improvements:
                print(f"  - {imp}")

        avg = feedback.store.average_score()
        print(f"\nRunning average score: {avg:.1f}/10")


if __name__ == "__main__":
    asyncio.run(main())
