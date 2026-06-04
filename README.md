# Agentic AI Workbench

**A production-grade framework for building, orchestrating, and deploying autonomous AI agents.**

Based on the concepts, frameworks, and methodologies from *AI Agents in Action* by Micheal Lanham, adapted for daily use in Cursor IDE.

---

## Quick Start

```bash
# 1. Clone and enter
cd agentic-ai-workbench

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Verify installation
python -m pytest tests/ -v

# 6. Run your first agent
python scripts/run_agent.py --profile agents/profiles/research_analyst.yaml --task "Summarize key AI infrastructure trends"
```

---

## Architecture Overview

This workbench implements seven core layers of agentic systems, each mapping
to chapters from the book:

```
Layer 7: Multi-Agent Orchestration  (Ch 4: Multi-agent systems)
Layer 6: Planning and Feedback      (Ch 11: Planning and feedback)
Layer 5: Reasoning and Evaluation   (Ch 10: Reasoning and evaluation)
Layer 4: Prompt Engineering         (Ch 9: Prompt flow)
Layer 3: Memory and Knowledge       (Ch 8: Memory and knowledge)
Layer 2: Tools and Actions          (Ch 5: Actions, Ch 6: Autonomous assistants)
Layer 1: Agent Core + Profiles      (Ch 1-3: Foundations, LLMs, Assistants)
```

---

## Step-by-Step Guide

### Phase 1: Agent Foundations (Chapters 1-3)

**Goal:** Understand what an agent is and build your first one.

An agent is an autonomous entity with: a profile (who it is), tools (what it
can do), memory (what it knows), and reasoning (how it thinks).

1. Define an agent profile in `agents/profiles/`
2. Configure your LLM connection in `config/`
3. Run the base agent loop in `agents/base_agent.py`
4. Observe the perception-reasoning-action cycle

### Phase 2: Tools and Actions (Chapters 5-6)

**Goal:** Give agents the ability to act on the world.

Tools are functions agents can call. Actions are sequences of tool calls
that accomplish a goal. Behavior trees structure complex decision-making.

1. Register tools in `agents/tools/`
2. Build action sequences in `agents/actions/`
3. Implement behavior trees for autonomous decision-making
4. Test tool execution with mock responses

### Phase 3: Memory and Knowledge (Chapter 8)

**Goal:** Give agents persistent knowledge and context.

Memory types: conversational (chat history), semantic (facts/embeddings),
episodic (past experiences), procedural (how-to knowledge).

1. Set up ChromaDB vector store in `memory/stores/`
2. Build RAG pipeline in `memory/rag/`
3. Implement memory compression in `memory/compression/`
4. Index your knowledge base documents

### Phase 4: Prompt Engineering (Chapter 9)

**Goal:** Systematic prompt design with templates and evaluation.

1. Create persona templates in `prompts/personas/`
2. Build prompt flows in `prompts/flows/`
3. Evaluate prompts with rubrics
4. Version and A/B test your prompts

### Phase 5: Reasoning and Evaluation (Chapter 10)

**Goal:** Implement structured reasoning patterns.

- ReAct: Thought -> Action -> Observation loop
- Chain of Thought: Step-by-step reasoning
- Tree of Thought: Branching exploration
- Self-Consistency: Multiple reasoning paths, majority vote

1. Implement ReAct loop in `reasoning/react/`
2. Add CoT prompting in `reasoning/cot/`
3. Build evaluation rubrics in `reasoning/evaluation/`
4. Score agent outputs systematically

### Phase 6: Planning and Feedback (Chapter 11)

**Goal:** Agents that plan multi-step tasks and improve over time.

1. Build sequential planner in `planning/sequential/`
2. Implement stepwise planner in `planning/sequential/planner.py`
3. Add feedback collection in `planning/feedback/`
4. Create self-improving feedback loops

### Phase 7: Multi-Agent Orchestration (Chapter 4)

**Goal:** Multiple agents collaborating on complex problems.

1. Define agent crews in `multiagent/crews/`
2. Build orchestration logic in `multiagent/orchestration/`
3. Implement agent communication protocols
4. Deploy collaborative workflows

---

## Project Structure

The repository is organized by capability layer:

- `agents/` - core agent loop, profiles, behavior tree, and tool implementations
- `memory/` - vector store integration, conversation memory, and RAG pipeline
- `reasoning/` - ReAct loop, CoT/ToT/self-consistency, and rubric evaluation
- `planning/` - sequential and stepwise planning plus feedback loops
- `multiagent/` - crew execution and orchestration routing/protocols
- `agent_platform/` - FastAPI server and Streamlit UI
- `scripts/` - runnable entrypoints (`run_agent`, `index_knowledge`, `evaluate_agent`)
- `prompts/` - prompt templates, flows, and persona definitions
- `tests/` - pytest suite for core agent, reasoning, and RAG behavior
- `docs/` - architecture and usage guides

---

## Daily Workflow in Cursor

### Starting a New Agent

1. Open Cursor in this project directory
2. Create a new profile YAML in `agents/profiles/`
3. Use Agent Mode: "Create a new agent based on the research_analyst profile
   but specialized for [your task]"
4. The Cursor rules file will guide the AI to follow the architecture

### Iterating on an Agent

1. Run the agent: `python scripts/run_agent.py --profile your_agent.yaml`
2. Review the reasoning trace in the logs
3. Adjust the profile, tools, or prompts
4. Re-run and compare outputs using evaluation rubrics

### Adding Knowledge

1. Drop documents into `data/knowledge_base/`
2. Run: `python scripts/index_knowledge.py`
3. Your agents now have access to that knowledge via RAG

---

## Key Concepts Reference

| Concept | Book Chapter | Location in Project |
|---------|-------------|-------------------|
| Agent Profiles | Ch 1, 7, 9 | agents/profiles/ |
| LLM Integration | Ch 2 | config/models.py |
| Assistants API | Ch 3 | agents/base_agent.py |
| Multi-Agent Systems | Ch 4 | multiagent/ |
| Tools and Actions | Ch 5 | agents/tools/, agents/actions/ |
| Behavior Trees | Ch 6 | agents/behavior_tree.py |
| Agent Platform | Ch 7 | agent_platform/ |
| Memory and RAG | Ch 8 | memory/ |
| Prompt Engineering | Ch 9 | prompts/ |
| Reasoning/Evaluation | Ch 10 | reasoning/ |
| Planning/Feedback | Ch 11 | planning/ |

---

## Related projects

| Project | Description |
|---------|-------------|
| [ParallelLLC/SOUS](https://github.com/ParallelLLC/SOUS) | Domain-specific production pipeline built on workbench patterns: five-agent vulnerability triage with deterministic LLM fallback and POAM exports |
| [EAName/mango-tango-cli](https://github.com/EAName/mango-tango-cli) | Civic analytics CLI with burst detection and attribution (CIB Mango Tree) |
| [EAName/Segmentation](https://github.com/EAName/Segmentation) | SAM 2 segmentation research with few-shot and zero-shot evaluation |

**Parallel LLC** uses this workbench as the general agent layer; **SOUS** is the regulated-security vertical shipped from the same engineering lineage.
