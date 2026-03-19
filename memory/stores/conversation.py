"""
Conversation Memory Manager.

Based on Ch 8: Understanding agent memory and knowledge.

Memory types implemented:
  - Conversational: Short-term chat history (sliding window)
  - Semantic: Long-term factual knowledge (via RAG/vector store)
  - Episodic: Past experiences and outcomes (feedback store)
  - Procedural: How-to knowledge (tool usage patterns)

This module handles conversational memory with compression
to keep context windows manageable during long interactions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel

from config.models import BaseLLMClient, LLMMessage

logger = structlog.get_logger()


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    role: str
    content: str
    timestamp: str = ""
    metadata: dict[str, Any] = {}


class ConversationMemory:
    """
    Manages conversation history with a sliding window and compression.

    When the conversation exceeds max_turns, older messages are
    summarized into a compressed context block, preserving the most
    important information while staying within token limits.

    Usage:
        memory = ConversationMemory(llm_client, max_turns=20)
        memory.add("user", "What is federated learning?")
        memory.add("assistant", "Federated learning is...")

        # Get messages for the next LLM call
        messages = await memory.get_messages()

        # Save/load for persistence
        memory.save("./data/conversations/session_001.json")
        memory.load("./data/conversations/session_001.json")
    """

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        max_turns: int = 20,
        compression_threshold: int = 15,
    ):
        self.llm = llm_client
        self.max_turns = max_turns
        self.compression_threshold = compression_threshold
        self.turns: list[ConversationTurn] = []
        self.compressed_context: str = ""
        self.session_id: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def add(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a turn to the conversation."""
        self.turns.append(
            ConversationTurn(
                role=role,
                content=content,
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata=metadata or {},
            )
        )

    async def get_messages(self, system_prompt: str = "") -> list[LLMMessage]:
        """
        Get the message list for the next LLM call.

        If conversation is long, compresses older messages and
        prepends the summary to the system prompt.
        """
        # Compress if needed
        if len(self.turns) > self.compression_threshold and self.llm:
            await self._compress()

        messages = []

        # System prompt with compressed context
        sys_content = system_prompt
        if self.compressed_context:
            sys_content += (
                f"\n\n--- Conversation History Summary ---\n"
                f"{self.compressed_context}\n"
                f"--- End Summary ---"
            )
        if sys_content:
            messages.append(LLMMessage(role="system", content=sys_content))

        # Recent turns (within window)
        recent = self.turns[-self.max_turns:]
        for turn in recent:
            messages.append(LLMMessage(role=turn.role, content=turn.content))

        return messages

    async def _compress(self) -> None:
        """Compress older turns into a summary."""
        if not self.llm:
            return

        # Take the oldest turns beyond the window
        to_compress = self.turns[:-self.compression_threshold]
        if not to_compress:
            return

        conversation_text = "\n".join(
            f"{t.role}: {t.content[:200]}" for t in to_compress
        )

        prompt = (
            "Summarize this conversation history, preserving key facts, "
            "decisions, and context that would be needed to continue the "
            "conversation. Be concise but complete.\n\n"
            f"{conversation_text}"
        )

        response = await self.llm.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.1,
        )

        self.compressed_context = response.content
        # Remove compressed turns
        self.turns = self.turns[-self.compression_threshold:]

        logger.info(
            "memory_compressed",
            turns_compressed=len(to_compress),
            summary_length=len(self.compressed_context),
        )

    def save(self, path: str | Path) -> None:
        """Save conversation to a JSON file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "session_id": self.session_id,
            "compressed_context": self.compressed_context,
            "turns": [t.model_dump() for t in self.turns],
        }

        filepath.write_text(json.dumps(data, indent=2))
        logger.info("conversation_saved", path=str(filepath), turns=len(self.turns))

    def load(self, path: str | Path) -> None:
        """Load conversation from a JSON file."""
        filepath = Path(path)
        data = json.loads(filepath.read_text())

        self.session_id = data.get("session_id", self.session_id)
        self.compressed_context = data.get("compressed_context", "")
        self.turns = [ConversationTurn(**t) for t in data.get("turns", [])]

        logger.info("conversation_loaded", path=str(filepath), turns=len(self.turns))

    def clear(self) -> None:
        """Clear all memory."""
        self.turns.clear()
        self.compressed_context = ""

    @property
    def turn_count(self) -> int:
        return len(self.turns)
