# Comprehensive Step-by-Step Guide

## How to Use This Workbench in Your Daily Work

This guide is structured around the seven layers from *AI Agents in Action*,
ordered from foundations to advanced orchestration. Each section tells you
exactly what to do, where the code lives, and how to extend it.

---

## 1. Define Your Agent (5 minutes)

Every agent starts with a profile. The profile is a YAML file that defines
who the agent is, what it can do, and how it should behave.

**Create a new profile:**

```yaml
# agents/profiles/your_agent.yaml
name: "Deal Sourcing Analyst"
role: "Venture capital analyst specializing in AI/ML deal sourcing"
goal: >
  Identify and evaluate early-stage companies in AI infrastructure,
  defense tech, and privacy-enhancing technologies for investment.
backstory: >
  You have deep expertise in AI/ML technical architecture and understand
  both the technical and commercial viability of startups.
constraints:
  - Use the Power Law Alignment Scorecard for evaluation
  - Flag conflict-of-interest concerns explicitly
  - Quantify market opportunity where possible
tools:
  - web_search
  - read_file
  - write_file
temperature: 0.3
max_iterations: 15
provider: anthropic
model: claude-sonnet-4-20250514
```

**What the profile controls:**
- `name/role/goal/backstory`: Become the system prompt
- `constraints`: Hard rules the agent must follow
- `tools`: Which tools the agent can access (registered in agents/tools/)
- `temperature`: Lower = more deterministic, higher = more creative
- `max_iterations`: Safety limit on the ReAct loop

---

## 2. Register Custom Tools (10 minutes)

Tools are Python functions decorated with `@agent_tool`. The decorator
registers them in the global registry so any agent can use them.

**Create a new tool:**

```python
# agents/tools/your_tool.py
from agents.tools.base_tool import agent_tool

@agent_tool(
    name="search_crunchbase",
    description="Search Crunchbase for company funding data",
    parameters={
        "type": "object",
        "properties": {
            "company_name": {"type": "string", "description": "Company to search"}
        },
        "required": ["company_name"]
    }
)
async def search_crunchbase(company_name: str) -> dict:
    # Your implementation here
    async with aiohttp.ClientSession() as session:
        ...
    return {"company": company_name, "funding": "..."}
```

**Then import it in `agents/tools/__init__.py`:**
```python
from agents.tools import your_tool  # noqa: F401
```

**Built-in tools:**
- `read_file` / `write_file` / `list_directory`: File operations
- `database`: Read-only SQL against SQLite
- `code_exec`: Run Python code in a sandboxed subprocess

---

## 3. Build Your Knowledge Base (15 minutes)

RAG gives your agents access to documents beyond their training data.

**Step 1: Add documents**
Drop files into `data/knowledge_base/`:
- PDFs, Word docs, Markdown, Python files, YAML, JSON
- Example: DD memos, company profiles, market maps, technical specs

**Step 2: Index them**
```bash
python scripts/index_knowledge.py
```

**Step 3: Query in code**
```python
from memory.rag.pipeline import RAGPipeline

rag = RAGPipeline()
results = rag.retrieve("federated learning privacy medical imaging", top_k=5)
context = rag.format_context(results)
# Inject `context` into your agent's system prompt
```

**Step 4: Use via API**
```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "fusion energy market size", "top_k": 5}'
```

---

## 4. Choose Your Reasoning Pattern

Different tasks need different reasoning approaches:

| Pattern | Best For | File |
|---------|----------|------|
| ReAct | Tool-using tasks, research, data gathering | reasoning/react/loop.py |
| Chain of Thought | Math, logic, step-by-step analysis | reasoning/cot/chain.py |
| Tree of Thought | Creative strategy, architecture design | reasoning/cot/chain.py |
| Self-Consistency | Factual Q&A, classification, scoring | reasoning/cot/chain.py |

**ReAct (most common):**
The base agent already uses ReAct. The loop is:
Thought -> Action -> Observation -> Thought -> ... -> Final Answer

**Chain of Thought:**
```python
from reasoning.cot.chain import ChainOfThought

cot = ChainOfThought(llm_client)
result = await cot.reason(
    "Compare LeoLabs and Zephyr Fusion on Power Law alignment",
    context=rag_context,
)
print(result.reasoning)
print(result.final_answer)
```

**Self-Consistency (high-stakes decisions):**
```python
from reasoning.cot.chain import SelfConsistency

sc = SelfConsistency(llm_client, num_samples=5)
result = await sc.reason("Should we invest in this company?")
print(f"Answer: {result.consensus_answer}")
print(f"Agreement: {result.agreement_ratio:.0%}")
```

---

## 5. Plan Complex Tasks

For multi-step work, use a planner to decompose and execute.

**Sequential Planner (known steps):**
```python
from planning.sequential.planner import SequentialPlanner

planner = SequentialPlanner(llm_client, tool_registry)
plan = await planner.create_plan("Build a DD memo for Anthology Bio")
result = await planner.execute_plan(plan)

for step in result.steps:
    print(f"Step {step.step_number}: {step.description} [{step.status}]")
```

**Stepwise Planner (exploratory):**
```python
from planning.sequential.planner import StepwisePlanner

planner = StepwisePlanner(llm_client, tool_registry, max_steps=15)
result = await planner.run("Investigate why the data pipeline is failing")
```

---

## 6. Evaluate Agent Outputs

Never ship agent output without evaluation. Use rubrics.

```bash
# CLI evaluation
python scripts/evaluate_agent.py \
  --profile agents/profiles/research_analyst.yaml \
  --task "Analyze the fusion energy market" \
  --rubric research \
  --feedback
```

**Custom rubric:**
```python
from reasoning.evaluation.rubric import RubricEvaluator

evaluator = RubricEvaluator(llm_client)
evaluator.add_criterion("technical_depth", "Demonstrates deep technical understanding", weight=2.0)
evaluator.add_criterion("investment_thesis", "Clear investment thesis with supporting evidence", weight=2.0)
evaluator.add_criterion("risk_assessment", "Identifies and quantifies key risks", weight=1.5)

result = await evaluator.evaluate(task=task, output=agent_output)
print(f"Score: {result.percentage:.0f}% - {'PASS' if result.passed else 'FAIL'}")
```

---

## 7. Set Up Feedback Loops

Feedback loops make agents improve over time.

```python
from planning.feedback.loop import FeedbackLoop

feedback = FeedbackLoop(llm_client)

# After each agent run, auto-critique the output
entry = await feedback.auto_critique(task, output)
feedback.store.add(entry)

# When building prompts, inject past learnings
learnings = feedback.get_learnings(tags=["research", "dd_memo"])
# Append learnings to system prompt
```

**Track trends:**
```python
avg_score = feedback.store.average_score(last_n=50)
low_performers = feedback.store.get_low_scores(threshold=5.0)
```

---

## 8. Orchestrate Multi-Agent Crews

For complex deliverables, use multiple agents.

**Sequential crew (pipeline):**
```python
from multiagent.crews.crew_runner import CrewRunner
from agents.base_agent import AgentProfile

runner = CrewRunner()
runner.add_agent("researcher", AgentProfile.from_yaml("agents/profiles/research_analyst.yaml"))
runner.add_agent("engineer", AgentProfile.from_yaml("agents/profiles/data_engineer.yaml"))
runner.add_agent("reviewer", AgentProfile.from_yaml("agents/profiles/code_reviewer.yaml"))

result = await runner.run_sequential(
    goal="Produce a technical DD memo on a federated learning startup",
    tasks=[
        {"agent": "researcher", "task": "Research the company's technology and market position"},
        {"agent": "engineer", "task": "Evaluate the technical architecture and scalability"},
        {"agent": "reviewer", "task": "Review for accuracy, gaps, and red flags"},
    ],
)
```

**Hierarchical crew (manager + workers):**
```python
result = await runner.run_hierarchical(
    goal="Build a comprehensive market map for space infrastructure",
    manager_agent="researcher",
    worker_agents=["engineer", "reviewer"],
)
```

---

## 9. Build Behavior Trees for Autonomous Workflows

When agents need structured decision-making with fallbacks:

```python
from agents.behavior_tree import (
    BehaviorTree, SequenceNode, SelectorNode,
    ConditionNode, ActionNode, NodeStatus,
)

async def check_data(bb):
    # Check if we already have data
    return NodeStatus.SUCCESS if bb.get("data") else NodeStatus.FAILURE

async def fetch_from_cache(bb):
    # Try cache first
    ...

async def fetch_from_api(bb):
    # Fallback to API
    ...

async def analyze(bb):
    # Run analysis on whatever data we got
    ...

tree = BehaviorTree(
    SequenceNode("pipeline", [
        SelectorNode("get_data", [
            ActionNode("cache", fetch_from_cache),
            ActionNode("api", fetch_from_api),
        ]),
        ActionNode("analyze", analyze),
    ])
)

status = await tree.run({"task": "analyze Q3 data"})
```

---

## 10. Serve Agents via API

```bash
# Start the server
uvicorn agent_platform.api.server:app --reload --port 8000

# Or use Streamlit for prototyping
streamlit run agent_platform/ui/app.py
```

**API endpoints:**
- `POST /agents/run` - Run any agent on a task
- `POST /agents/evaluate` - Score an output
- `POST /rag/query` - Search the knowledge base
- `POST /rag/ingest` - Add a document
- `GET /agents/profiles` - List available agents
- `GET /health` - Health check

---

## Cursor Workflow Cheat Sheet

| What You Want | What to Tell Cursor Agent Mode |
|---------------|-------------------------------|
| New agent | "Create a new agent profile YAML in agents/profiles/ for [purpose]" |
| New tool | "Create a new agent tool in agents/tools/ that [does X]" |
| RAG query | "Use the RAG pipeline to find information about [topic]" |
| Run evaluation | "Evaluate this output using the research rubric" |
| Multi-agent | "Set up a crew with [agent A] and [agent B] for [goal]" |
| Behavior tree | "Build a behavior tree for [workflow with fallbacks]" |
| Debug agent | "Show me the reasoning trace from the last agent run" |
| Add knowledge | "Index the documents in data/knowledge_base/" |
