"""
Test fixtures and mock LLM client.

The mock client allows testing agent logic without making real API calls.
"""

import pytest
from typing import Any

from config.models import BaseLLMClient, LLMMessage, LLMResponse


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for testing.

    Configure responses before test:
        mock_llm = MockLLMClient()
        mock_llm.add_response("This is the agent's response")
        mock_llm.add_response("Second call response", tool_calls=[...])
    """

    def __init__(self):
        self.responses: list[LLMResponse] = []
        self.call_history: list[dict[str, Any]] = []
        self._call_index = 0

    def add_response(
        self,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        stop_reason: str = "end_turn",
    ) -> None:
        self.responses.append(
            LLMResponse(
                content=content,
                tool_calls=tool_calls or [],
                usage={"input_tokens": 100, "output_tokens": 50},
                model="mock-model",
                stop_reason=stop_reason,
            )
        )

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.call_history.append(
            {
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        if self._call_index < len(self.responses):
            response = self.responses[self._call_index]
            self._call_index += 1
            return response

        # Default response if none configured
        return LLMResponse(
            content="Default mock response",
            usage={"input_tokens": 10, "output_tokens": 5},
            model="mock-model",
            stop_reason="end_turn",
        )


@pytest.fixture
def mock_llm():
    """Provide a fresh mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def sample_profile_dict():
    """Sample agent profile data."""
    return {
        "name": "Test Agent",
        "role": "Testing assistant",
        "goal": "Help run tests",
        "backstory": "A test agent",
        "constraints": ["Be concise", "Be accurate"],
        "tools": [],
        "temperature": 0.5,
        "max_iterations": 5,
    }
