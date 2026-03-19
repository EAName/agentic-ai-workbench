"""
Base tool interface and registration decorator.

Based on Ch 5: Empowering agents with actions.

Tools are the bridge between an agent's reasoning and the real world.
Every tool must declare its name, description, and parameter schema
so the LLM knows when and how to call it.
"""

from __future__ import annotations

from typing import Any, Callable
from functools import wraps

from agents.base_agent import tool_registry


def agent_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> Callable:
    """
    Decorator that registers a function as an agent tool.

    Usage:
        @agent_tool(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        )
        async def web_search(query: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        tool_registry.register(
            name=name,
            func=func,
            description=description,
            parameters=parameters,
        )

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs) if __import__("asyncio").iscoroutinefunction(func) else func(*args, **kwargs)

        return wrapper

    return decorator
