"""
FastAPI Agent Server.

Based on Ch 7: Assembling and using an agent platform.

Serves agents via REST API so they can be called from any client,
integrated into workflows, or exposed as microservices.

Endpoints:
  POST /agents/run          - Run an agent on a task
  POST /agents/evaluate     - Evaluate an output with a rubric
  POST /rag/query           - Query the knowledge base
  POST /rag/ingest          - Ingest a document
  GET  /agents/profiles     - List available agent profiles
  GET  /health              - Health check
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(
    title="Agentic AI Workbench",
    description="Production-grade agentic AI platform API",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class AgentRunRequest(BaseModel):
    profile: str  # path to profile YAML
    task: str
    provider: str | None = None
    model: str | None = None


class AgentRunResponse(BaseModel):
    agent_name: str
    output: str
    steps: int
    tokens: int


class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    collection: str = "knowledge_base"


class RAGQueryResponse(BaseModel):
    results: list[dict[str, Any]]
    context: str


class RAGIngestRequest(BaseModel):
    file_path: str
    collection: str = "knowledge_base"


class EvaluateRequest(BaseModel):
    task: str
    output: str
    rubric: str = "research"  # "research" or "code_review"


class EvaluateResponse(BaseModel):
    percentage: float
    passed: bool
    summary: str
    scores: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "agentic-ai-workbench"}


@app.get("/agents/profiles")
async def list_profiles():
    """List all available agent profiles."""
    profiles_dir = PROJECT_ROOT / "agents/profiles"
    profiles = []
    if profiles_dir.exists():
        for f in sorted(profiles_dir.glob("*.yaml")):
            profiles.append({"name": f.stem, "path": str(f)})
    return {"profiles": profiles}


@app.post("/agents/run", response_model=AgentRunResponse)
async def run_agent(request: AgentRunRequest):
    """Run an agent on a task."""
    from agents.base_agent import BaseAgent, AgentProfile
    from agents.tools import file_ops  # noqa: F401
    from agents.tools import web_search  # noqa: F401

    profile_path = Path(request.profile)
    if not profile_path.is_absolute():
        profile_path = PROJECT_ROOT / profile_path
    if not profile_path.exists():
        raise HTTPException(404, f"Profile not found: {request.profile}")

    profile = AgentProfile.from_yaml(profile_path)
    if request.provider:
        profile.provider = request.provider
    if request.model:
        profile.model = request.model

    agent = BaseAgent(profile=profile)
    output = await agent.run(request.task)

    return AgentRunResponse(
        agent_name=profile.name,
        output=output,
        steps=len(agent.trace.steps) if agent.trace else 0,
        tokens=agent.trace.total_tokens if agent.trace else 0,
    )


@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """Query the knowledge base via RAG."""
    from memory.rag.pipeline import RAGPipeline

    rag = RAGPipeline(collection_name=request.collection)
    results = rag.retrieve(request.query, top_k=request.top_k)
    context = rag.format_context(results)

    return RAGQueryResponse(
        results=[
            {
                "content": r.document.content[:300],
                "score": r.score,
                "metadata": r.document.metadata,
            }
            for r in results
        ],
        context=context,
    )


@app.post("/rag/ingest")
async def rag_ingest(request: RAGIngestRequest):
    """Ingest a document into the knowledge base."""
    from memory.rag.pipeline import RAGPipeline

    path = Path(request.file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise HTTPException(404, f"File not found: {request.file_path}")

    rag = RAGPipeline(collection_name=request.collection)
    chunks = rag.ingest_file(path)

    return {"file": str(path), "chunks_indexed": chunks}


@app.post("/agents/evaluate", response_model=EvaluateResponse)
async def evaluate_output(request: EvaluateRequest):
    """Evaluate an agent output with a rubric."""
    from reasoning.evaluation.rubric import research_rubric, code_review_rubric
    from config.models import create_llm_client

    llm = create_llm_client()

    rubric_map = {
        "research": research_rubric,
        "code_review": code_review_rubric,
    }

    factory = rubric_map.get(request.rubric)
    if not factory:
        raise HTTPException(400, f"Unknown rubric: {request.rubric}")

    evaluator = factory(llm)
    result = await evaluator.evaluate(task=request.task, output=request.output)

    return EvaluateResponse(
        percentage=result.percentage,
        passed=result.passed,
        summary=result.summary,
        scores=[
            {
                "criterion": s.criterion,
                "score": s.score,
                "max_score": s.max_score,
                "explanation": s.explanation,
            }
            for s in result.scores
        ],
    )


# ---------------------------------------------------------------------------
# Run with: uvicorn agent_platform.api.server:app --reload
# ---------------------------------------------------------------------------
