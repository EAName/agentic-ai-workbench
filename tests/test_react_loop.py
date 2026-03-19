"""Tests for the ReAct reasoning loop."""

import pytest
from reasoning.react.loop import ReActLoop, ReActResult
from agents.base_agent import ToolRegistry


class TestReActParsing:
    @pytest.mark.asyncio
    async def test_react_finish(self, mock_llm):
        mock_llm.add_response(
            "Thought: I know the answer.\n"
            "Action: FINISH\n"
            "Action Input: The answer is 42."
        )

        tools = ToolRegistry()
        react = ReActLoop(llm_client=mock_llm, tools=tools, max_steps=5)
        result = await react.run("What is the answer?")

        assert result.final_answer == "The answer is 42."
        assert len(result.steps) == 1
        assert result.steps[0].action == "FINISH"

    @pytest.mark.asyncio
    async def test_react_with_tool(self, mock_llm):
        # First call: agent wants to use a tool
        mock_llm.add_response(
            'Thought: I need to read the file.\n'
            'Action: read_file\n'
            'Action Input: {"file_path": "test.txt"}'
        )
        # Second call: agent finishes
        mock_llm.add_response(
            "Thought: Now I have the data.\n"
            "Action: FINISH\n"
            "Action Input: The file contains test data."
        )

        tools = ToolRegistry()
        tools.register(
            "read_file",
            lambda file_path: "test content",
            "Read a file",
            {"type": "object", "properties": {"file_path": {"type": "string"}}},
        )

        react = ReActLoop(llm_client=mock_llm, tools=tools, max_steps=5)
        result = await react.run("What is in test.txt?")

        assert result.final_answer == "The file contains test data."
        assert len(result.steps) == 2
        assert result.steps[0].action == "read_file"
