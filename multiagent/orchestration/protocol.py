"""
Agent Communication Protocol.

Based on Ch 4: Exploring multi-agent systems.

Defines how agents communicate with each other in multi-agent systems.
Supports structured messages with type classification so agents can
distinguish between requests, responses, critiques, and handoffs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(Enum):
    REQUEST = "request"       # Agent asks another to do something
    RESPONSE = "response"     # Agent returns a result
    CRITIQUE = "critique"     # Agent critiques another's output
    HANDOFF = "handoff"       # Agent passes control to another
    STATUS = "status"         # Agent reports its current state
    BROADCAST = "broadcast"   # Message to all agents in the crew


class AgentMessage(BaseModel):
    """A structured message between agents."""

    msg_id: str = ""
    from_agent: str
    to_agent: str = ""  # empty = broadcast
    msg_type: MessageType
    content: str
    data: dict[str, Any] = {}
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    parent_msg_id: str = ""  # for threading


class MessageBus:
    """
    Simple in-memory message bus for agent communication.

    In production, replace with Redis pub/sub, RabbitMQ, or similar.

    Usage:
        bus = MessageBus()

        # Agent A sends a request to Agent B
        bus.send(AgentMessage(
            from_agent="researcher",
            to_agent="reviewer",
            msg_type=MessageType.REQUEST,
            content="Please review this analysis",
            data={"analysis": "..."}
        ))

        # Agent B retrieves messages
        messages = bus.receive("reviewer")
    """

    def __init__(self):
        self._messages: list[AgentMessage] = []
        self._counter = 0

    def send(self, message: AgentMessage) -> str:
        """Send a message. Returns the message ID."""
        self._counter += 1
        message.msg_id = f"msg_{self._counter:04d}"
        self._messages.append(message)
        return message.msg_id

    def receive(
        self,
        agent_name: str,
        msg_type: MessageType | None = None,
        unread_only: bool = True,
    ) -> list[AgentMessage]:
        """Get messages for a specific agent."""
        results = []
        for msg in self._messages:
            if msg.to_agent == agent_name or (msg.to_agent == "" and msg.from_agent != agent_name):
                if msg_type is None or msg.msg_type == msg_type:
                    results.append(msg)
        return results

    def get_thread(self, msg_id: str) -> list[AgentMessage]:
        """Get all messages in a thread."""
        thread = []
        for msg in self._messages:
            if msg.msg_id == msg_id or msg.parent_msg_id == msg_id:
                thread.append(msg)
        return sorted(thread, key=lambda m: m.timestamp)

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        self._counter = 0

    @property
    def message_count(self) -> int:
        return len(self._messages)
