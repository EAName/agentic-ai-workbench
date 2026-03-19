"""
Streamlit Agent UI.

Based on Ch 7: Assembling and using an agent platform.

A rapid prototyping interface for interacting with agents,
viewing reasoning traces, and testing different configurations.

Run with: streamlit run agent_platform/ui/app.py
"""

import asyncio
from pathlib import Path

import streamlit as st

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.base_agent import BaseAgent, AgentProfile
from agents.tools import file_ops  # noqa: F401


def get_profiles() -> list[Path]:
    """List available agent profiles."""
    profiles_dir = PROJECT_ROOT / "agents/profiles"
    if profiles_dir.exists():
        return sorted(profiles_dir.glob("*.yaml"))
    return []


def main():
    st.set_page_config(page_title="Agentic AI Workbench", layout="wide")
    st.title("Agentic AI Workbench")
    st.caption("Based on AI Agents in Action by Micheal Lanham")

    # Sidebar: Agent selection
    with st.sidebar:
        st.header("Agent Configuration")

        profiles = get_profiles()
        if not profiles:
            st.warning("No agent profiles found in agents/profiles/")
            return

        selected = st.selectbox(
            "Select Agent Profile",
            profiles,
            format_func=lambda p: p.stem.replace("_", " ").title(),
        )

        if selected:
            profile = AgentProfile.from_yaml(selected)
            st.markdown(f"**Role:** {profile.role}")
            st.markdown(f"**Goal:** {profile.goal[:100]}...")
            st.markdown(f"**Model:** {profile.model or 'default'}")
            st.markdown(f"**Temperature:** {profile.temperature}")
            st.markdown(f"**Tools:** {', '.join(profile.tools) or 'None'}")

        st.divider()
        show_trace = st.checkbox("Show reasoning trace", value=True)

    # Main area: Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None

    # Initialize or reset agent when profile changes
    if selected:
        profile = AgentProfile.from_yaml(selected)
        if (
            st.session_state.agent is None
            or st.session_state.agent.profile.name != profile.name
        ):
            st.session_state.agent = BaseAgent(profile=profile)
            st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Enter your task..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                agent = st.session_state.agent
                output = asyncio.run(agent.run(prompt))

            st.markdown(output)
            st.session_state.messages.append(
                {"role": "assistant", "content": output}
            )

            # Show trace
            if show_trace and agent.trace:
                with st.expander(
                    f"Reasoning Trace ({len(agent.trace.steps)} steps, "
                    f"{agent.trace.total_tokens} tokens)"
                ):
                    for step in agent.trace.steps:
                        st.markdown(f"**Step {step.step_number}**")
                        if step.thought:
                            st.text(f"Thought: {step.thought[:300]}")
                        if step.action:
                            st.text(f"Action: {step.action}")
                            if step.action_input:
                                st.json(step.action_input)
                        if step.observation:
                            st.text(f"Observation: {step.observation[:300]}")
                        st.divider()


if __name__ == "__main__":
    main()
