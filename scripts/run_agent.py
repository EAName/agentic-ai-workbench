"""
CLI Agent Runner.

Run any agent profile from the command line.

Usage:
    python scripts/run_agent.py --profile agents/profiles/research_analyst.yaml --task "Analyze fusion energy market"
    python scripts/run_agent.py --profile agents/profiles/data_engineer.yaml --task "Design a pipeline for sensor data"
    python scripts/run_agent.py --profile agents/profiles/research_analyst.yaml --interactive
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.base_agent import BaseAgent, AgentProfile
from config.models import create_llm_client

# Import tools to register them
from agents.tools import file_ops  # noqa: F401


def resolve_profile_path(profile_path: str) -> Path:
    """Resolve profile path from cwd or project root."""
    path = Path(profile_path).expanduser()
    if path.exists():
        return path
    root_candidate = PROJECT_ROOT / profile_path
    if root_candidate.exists():
        return root_candidate
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Run an AI agent")
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Path to agent profile YAML",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task for the agent to execute",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print the full reasoning trace",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save output to file",
    )
    return parser.parse_args()


async def run_single_task(agent: BaseAgent, task: str, show_trace: bool = False):
    """Run a single task and print results."""
    print(f"\n{'='*60}")
    print(f"Agent: {agent.profile.name}")
    print(f"Task: {task}")
    print(f"{'='*60}\n")

    output = await agent.run(task)

    print("\n--- Agent Output ---\n")
    print(output)

    if show_trace and agent.trace:
        print(f"\n--- Reasoning Trace ({len(agent.trace.steps)} steps) ---\n")
        for step in agent.trace.steps:
            print(f"Step {step.step_number}:")
            if step.thought:
                print(f"  Thought: {step.thought[:200]}")
            if step.action:
                print(f"  Action: {step.action}")
            if step.observation:
                print(f"  Observation: {step.observation[:200]}")
            print()

        print(f"Total tokens: {agent.trace.total_tokens}")

    return output


async def run_interactive(agent: BaseAgent):
    """Run in interactive chat mode."""
    print(f"\nChat with {agent.profile.name} ({agent.profile.role})")
    print(f"Type 'quit' to exit, 'trace' to show last trace\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "trace":
            if agent.trace:
                print(json.dumps(agent.trace.to_dict(), indent=2, default=str))
            else:
                print("No trace available yet.")
            continue

        output = await agent.run(user_input)
        print(f"\n{agent.profile.name}: {output}\n")


async def main():
    args = parse_args()

    # Load profile
    profile_path = resolve_profile_path(args.profile)
    profile = AgentProfile.from_yaml(profile_path)

    # Create agent
    agent = BaseAgent(profile=profile)

    if args.interactive:
        await run_interactive(agent)
    elif args.task:
        output = await run_single_task(agent, args.task, show_trace=args.trace)

        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"\nOutput saved to {args.output}")
    else:
        print("Error: Provide --task or --interactive")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
