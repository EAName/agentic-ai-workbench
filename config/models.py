"""
LLM client factory.
Abstracts provider differences so agent code never touches raw API clients.

Supports: Anthropic (Claude), OpenAI (GPT), with a unified interface.
Based on Ch 2: Harnessing the power of large language models.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import structlog
from pydantic import BaseModel

from config.settings import settings

logger = structlog.get_logger()


class LLMMessage(BaseModel):
    """Unified message format across providers."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class LLMResponse(BaseModel):
    """Unified response format."""

    content: str
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, int] = {}
    model: str = ""
    stop_reason: str | None = None


class BaseLLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        ...


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client."""

    def __init__(self, api_key: str, model: str):
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        # Separate system message from conversation
        system_msg = ""
        conv_messages = []
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                conv_messages.append({"role": msg.role, "content": msg.content})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": conv_messages,
            "max_tokens": max_tokens or settings.llm.default_max_tokens,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = tools

        response = await self.client.messages.create(**kwargs)

        # Parse response into unified format
        content_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )

        return LLMResponse(
            content="\n".join(content_parts),
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            model=self.model,
            stop_reason=response.stop_reason,
        )


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT client."""

    def __init__(self, api_key: str, model: str):
        import openai

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        msgs = [{"role": m.role, "content": m.content} for m in messages]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": max_tokens or settings.llm.default_max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                )

        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            model=self.model,
            stop_reason=choice.finish_reason,
        )


def create_llm_client(
    provider: str | None = None,
    model: str | None = None,
) -> BaseLLMClient:
    """Factory function. Returns the appropriate LLM client.

    Args:
        provider: "anthropic" or "openai". Defaults to config.
        model: Model identifier. Defaults to config.

    Returns:
        Configured LLM client ready for use.
    """
    provider = provider or settings.llm.default_provider
    model = model or settings.llm.default_model

    if provider == "anthropic":
        if not settings.llm.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        return AnthropicClient(settings.llm.anthropic_api_key, model)
    elif provider == "openai":
        if not settings.llm.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        return OpenAIClient(settings.llm.openai_api_key, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
