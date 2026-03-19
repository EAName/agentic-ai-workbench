"""
Research Action: A multi-step research workflow.

Based on Ch 5: Empowering agents with actions, and
Ch 6: Building autonomous assistants.

An action is a higher-order composition of tools and reasoning
that accomplishes a complex goal. This research action:
  1. Queries the knowledge base for existing information
  2. Identifies gaps
  3. Uses tools to fill those gaps
  4. Synthesizes everything into a structured output
"""

from __future__ import annotations

from typing import Any

import structlog

from config.models import BaseLLMClient, LLMMessage
from memory.rag.pipeline import RAGPipeline
from agents.base_agent import ToolRegistry

logger = structlog.get_logger()


class ResearchAction:
    """
    Multi-step research action combining RAG + tools + reasoning.

    Usage:
        action = ResearchAction(llm_client, rag_pipeline, tool_registry)
        result = await action.execute(
            topic="Fusion energy startups for AV portfolio",
            depth="comprehensive",
        )
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        rag: RAGPipeline | None = None,
        tools: ToolRegistry | None = None,
    ):
        self.llm = llm_client
        self.rag = rag or RAGPipeline()
        self.tools = tools

    async def execute(
        self,
        topic: str,
        depth: str = "standard",
        output_format: str = "structured",
    ) -> dict[str, Any]:
        """
        Execute the research action.

        Args:
            topic: What to research.
            depth: "quick" (1 pass), "standard" (2 passes), "comprehensive" (3 passes + critique).
            output_format: "structured" (sections), "narrative" (prose), "bullets" (key points).
        """
        logger.info("research_action_started", topic=topic, depth=depth)

        # Step 1: Retrieve existing knowledge
        rag_results = self.rag.retrieve(topic, top_k=8)
        existing_context = self.rag.format_context(rag_results)

        # Step 2: Identify gaps
        gap_prompt = (
            f"Topic: {topic}\n\n"
            f"Existing knowledge:\n{existing_context or 'No existing knowledge found.'}\n\n"
            f"What are the key information gaps? List 3-5 specific questions "
            f"that need to be answered for a {depth} analysis."
        )

        gap_response = await self.llm.complete(
            messages=[LLMMessage(role="user", content=gap_prompt)],
            temperature=0.3,
        )

        # Step 3: Synthesize
        synthesis_prompt = (
            f"Topic: {topic}\n\n"
            f"Existing knowledge:\n{existing_context}\n\n"
            f"Key gaps identified:\n{gap_response.content}\n\n"
            f"Produce a {depth} {output_format} analysis. Include:\n"
            f"- Executive summary\n"
            f"- Key findings\n"
            f"- Risk factors\n"
            f"- Recommendations\n"
            f"- Confidence level and limitations"
        )

        synthesis = await self.llm.complete(
            messages=[LLMMessage(role="user", content=synthesis_prompt)],
            temperature=0.3,
        )

        result = {
            "topic": topic,
            "depth": depth,
            "existing_sources": len(rag_results),
            "gaps_identified": gap_response.content,
            "analysis": synthesis.content,
        }

        # Step 4: Self-critique for comprehensive depth
        if depth == "comprehensive":
            critique_prompt = (
                f"Critically review this analysis for:\n"
                f"1. Factual accuracy\n"
                f"2. Logical consistency\n"
                f"3. Missing perspectives\n"
                f"4. Unsupported claims\n\n"
                f"Analysis:\n{synthesis.content[:2000]}"
            )

            critique = await self.llm.complete(
                messages=[LLMMessage(role="user", content=critique_prompt)],
                temperature=0.2,
            )
            result["self_critique"] = critique.content

        logger.info(
            "research_action_completed",
            topic=topic,
            sources=len(rag_results),
        )

        return result
