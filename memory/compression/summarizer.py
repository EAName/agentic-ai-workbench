"""
Memory Compression.

Based on Ch 8: Understanding agent memory and knowledge.

As agents accumulate knowledge and conversation history, token usage grows.
Compression techniques reduce memory footprint while preserving the most
important information.

Strategies:
  - Summarization: Compress long text into key points
  - Entity extraction: Pull out named entities and relationships
  - Importance scoring: Keep high-value memories, discard low-value ones
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import BaseModel

from config.models import BaseLLMClient, LLMMessage

logger = structlog.get_logger()


class CompressedMemory(BaseModel):
    """A compressed memory unit."""

    summary: str
    key_entities: list[str] = []
    key_facts: list[str] = []
    importance_score: float = 0.5
    original_token_count: int = 0
    compressed_token_count: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_token_count == 0:
            return 0.0
        return 1 - (self.compressed_token_count / self.original_token_count)


COMPRESS_PROMPT = """Compress the following text into a concise summary that preserves:
1. Key facts and data points
2. Important entities (people, organizations, technologies)
3. Decisions made or actions taken
4. Open questions or unresolved items

Text to compress:
{text}

Respond with JSON only (no markdown fences):
{{
  "summary": "Concise summary preserving key information",
  "key_entities": ["entity1", "entity2"],
  "key_facts": ["fact1", "fact2"],
  "importance_score": 0.0 to 1.0
}}
"""


class MemoryCompressor:
    """
    Compresses text-based memories to save context window space.

    Usage:
        compressor = MemoryCompressor(llm_client)
        compressed = await compressor.compress(long_text)
        print(f"Compressed {compressed.compression_ratio:.0%}")

        # Batch compress conversation turns
        summaries = await compressor.compress_conversation(turns)
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client

    async def compress(self, text: str) -> CompressedMemory:
        """Compress a single text block."""
        import json

        original_tokens = len(text) // 4  # rough estimate

        prompt = COMPRESS_PROMPT.format(text=text[:3000])

        response = await self.llm.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.1,
        )

        try:
            clean = response.content.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
                clean = clean.strip()
            data = json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            data = {
                "summary": response.content[:500],
                "key_entities": [],
                "key_facts": [],
                "importance_score": 0.5,
            }

        summary = data.get("summary", "")
        compressed_tokens = len(summary) // 4

        result = CompressedMemory(
            summary=summary,
            key_entities=data.get("key_entities", []),
            key_facts=data.get("key_facts", []),
            importance_score=data.get("importance_score", 0.5),
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
        )

        logger.info(
            "memory_compressed",
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            ratio=f"{result.compression_ratio:.0%}",
        )

        return result

    async def compress_conversation(
        self,
        turns: list[dict[str, str]],
        max_summary_tokens: int = 500,
    ) -> CompressedMemory:
        """Compress a list of conversation turns."""
        text = "\n".join(
            f"{t.get('role', 'unknown')}: {t.get('content', '')[:200]}"
            for t in turns
        )
        return await self.compress(text)

    async def selective_compress(
        self,
        memories: list[CompressedMemory],
        keep_top_n: int = 10,
    ) -> list[CompressedMemory]:
        """
        Keep only the most important memories.

        Sorts by importance score and returns the top N.
        Useful for long-running agents that accumulate many memories.
        """
        sorted_memories = sorted(
            memories, key=lambda m: m.importance_score, reverse=True
        )
        kept = sorted_memories[:keep_top_n]
        discarded = len(memories) - len(kept)

        if discarded > 0:
            logger.info(
                "memories_pruned",
                kept=len(kept),
                discarded=discarded,
            )

        return kept
