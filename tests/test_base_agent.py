"""Tests for the base agent module."""

import pytest
from agents.base_agent import AgentProfile, BaseAgent, ToolRegistry


class TestAgentProfile:
    def test_from_dict(self, sample_profile_dict):
        profile = AgentProfile(**sample_profile_dict)
        assert profile.name == "Test Agent"
        assert profile.role == "Testing assistant"
        assert len(profile.constraints) == 2

    def test_to_system_prompt(self, sample_profile_dict):
        profile = AgentProfile(**sample_profile_dict)
        prompt = profile.to_system_prompt()
        assert "Test Agent" in prompt
        assert "Testing assistant" in prompt
        assert "Be concise" in prompt

    def test_default_values(self):
        profile = AgentProfile(name="Min", role="Test", goal="Do stuff")
        assert profile.temperature == 0.7
        assert profile.max_iterations == 10
        assert profile.tools == []


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        registry.register(
            name="test_tool",
            func=lambda x: x,
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )
        assert registry.get_tool("test_tool") is not None
        assert registry.get_tool("nonexistent") is None

    def test_get_schemas(self):
        registry = ToolRegistry()
        registry.register("tool_a", lambda: None, "Tool A", {})
        registry.register("tool_b", lambda: None, "Tool B", {})

        all_schemas = registry.get_schemas()
        assert len(all_schemas) == 2

        filtered = registry.get_schemas(["tool_a"])
        assert len(filtered) == 1
        assert filtered[0]["name"] == "tool_a"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        registry = ToolRegistry()
        registry.register("add", lambda a, b: a + b, "Add numbers", {})

        result = await registry.execute("add", {"a": 2, "b": 3})
        assert result.success is True
        assert result.output == 5

    @pytest.mark.asyncio
    async def test_execute_not_found(self):
        registry = ToolRegistry()
        result = await registry.execute("missing", {})
        assert result.success is False
        assert "not found" in result.error


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_run_simple(self, mock_llm, sample_profile_dict):
        mock_llm.add_response("Here is my analysis of the topic.")

        profile = AgentProfile(**sample_profile_dict)
        agent = BaseAgent(profile=profile, llm_client=mock_llm)

        output = await agent.run("Analyze this topic")
        assert output == "Here is my analysis of the topic."
        assert agent.trace is not None
        assert len(agent.trace.steps) == 1
        assert agent.trace.total_tokens > 0
