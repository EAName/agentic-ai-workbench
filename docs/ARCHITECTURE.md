# Architecture Guide

## Book-to-Code Mapping

This document maps every major concept from *AI Agents in Action* by
Micheal Lanham to its implementation in this workbench.

---

### Chapter 1: Introduction to Agents and Their World

**Core Concept:** An agent is an autonomous entity that perceives its
environment, reasons about it, and takes actions to achieve goals.

**Implementation:**
- `agents/base_agent.py` -> `BaseAgent` class with the perceive-reason-act loop
- `agents/base_agent.py` -> `AgentProfile` for defining agent identity
- `agents/base_agent.py` -> `AgentTrace` for observability

**How to use in Cursor:**
Open `agents/base_agent.py` and ask Cursor Agent Mode to "create a new
agent that specializes in [your domain]." It will follow the base class pattern.

---

### Chapter 2: Harnessing the Power of Large Language Models

**Core Concept:** LLMs are the reasoning engine. Choosing the right model,
managing tokens, and structuring API calls matters.

**Implementation:**
- `config/models.py` -> `BaseLLMClient` abstraction over Anthropic/OpenAI
- `config/models.py` -> `create_llm_client()` factory function
- `config/settings.py` -> Centralized configuration via pydantic-settings

**How to use in Cursor:**
Switch models by changing `DEFAULT_MODEL` in `.env`. The abstraction means
zero code changes when swapping between Claude and GPT.

---

### Chapter 3: Engaging GPT Assistants

**Core Concept:** Building conversational assistants with persistent context
and tool access through the OpenAI Assistants API.

**Implementation:**
- `agents/base_agent.py` -> Tool-using agent with conversation history
- `memory/stores/conversation.py` -> Persistent conversation memory

**How to use in Cursor:**
Run `python scripts/run_agent.py --profile agents/profiles/research_analyst.yaml --interactive`
to chat with an agent profile.

---

### Chapter 4: Exploring Multi-Agent Systems

**Core Concept:** Multiple specialized agents collaborating on tasks using
frameworks like AutoGen and CrewAI.

**Implementation:**
- `multiagent/crews/crew_runner.py` -> `CrewRunner` with sequential and hierarchical strategies
- `multiagent/crews/research_crew.yaml` -> Example crew definition

**Patterns:**
- Sequential: Agent A output feeds Agent B input feeds Agent C
- Hierarchical: Manager decomposes goal, delegates to workers, synthesizes

**How to use in Cursor:**
Define a crew YAML, then ask Cursor to "add a new agent to the research crew
that handles [specific task]."

---

### Chapter 5: Empowering Agents with Actions

**Core Concept:** Tools and actions let agents interact with the outside
world: search the web, read files, execute code, query databases.

**Implementation:**
- `agents/tools/base_tool.py` -> `@agent_tool` decorator for easy registration
- `agents/tools/file_ops.py` -> File read/write/list tools
- `agents/base_agent.py` -> `ToolRegistry` for managing available tools

**How to use in Cursor:**
Create a new file in `agents/tools/`, use the `@agent_tool` decorator, and
add the tool name to your agent's profile YAML under `tools:`.

---

### Chapter 6: Building Autonomous Assistants

**Core Concept:** Behavior trees structure complex decision-making with
sequences, selectors (fallbacks), conditions, and parallel execution.

**Implementation:**
- `agents/behavior_tree.py` -> Full behavior tree engine
  - `SequenceNode`: All children must succeed (ordered steps)
  - `SelectorNode`: First success wins (fallback strategies)
  - `ConditionNode`: Guard checks
  - `ActionNode`: Execute async actions
  - `ParallelNode`: Concurrent execution

**How to use in Cursor:**
Ask Cursor to "build a behavior tree for [your workflow]" and it will
compose nodes from the engine.

---

### Chapter 7: Assembling and Using an Agent Platform

**Core Concept:** A hosted agent platform with profiles, personas, and
tool access served via API.

**Implementation:**
- `agent_platform/api/` -> FastAPI server (extend for production)
- `agent_platform/ui/` -> Streamlit prototyping UI
- `agents/profiles/` -> YAML-based agent definitions

---

### Chapter 8: Understanding Agent Memory and Knowledge

**Core Concept:** RAG (Retrieval-Augmented Generation) gives agents access
to external knowledge. Memory types include conversational, semantic,
episodic, and procedural.

**Implementation:**
- `memory/rag/pipeline.py` -> Full RAG pipeline (ingest, index, retrieve, augment)
- `memory/stores/conversation.py` -> Sliding-window conversation memory with compression
- `planning/feedback/loop.py` -> Episodic memory via feedback store

**How to use in Cursor:**
1. Drop documents into `data/knowledge_base/`
2. Run `python scripts/index_knowledge.py`
3. Use `rag.retrieve(query)` in your agent code

---

### Chapter 9: Mastering Agent Prompts with Prompt Flow

**Core Concept:** Systematic prompt engineering with templates, personas,
evaluation rubrics, and versioning.

**Implementation:**
- `prompts/templates/` -> Jinja2 prompt templates
- `prompts/personas/` -> Persona definitions
- `agents/profiles/` -> Agent profiles that generate system prompts

**How to use in Cursor:**
Create a new `.jinja2` template, then ask Cursor to "wire this template
into the research agent's system prompt."

---

### Chapter 10: Agent Reasoning and Evaluation

**Core Concept:** Structured reasoning (CoT, ToT, ReAct) and systematic
evaluation with rubrics and self-consistency.

**Implementation:**
- `reasoning/react/loop.py` -> ReAct (Thought/Action/Observation) loop
- `reasoning/cot/chain.py` -> Chain of Thought, Tree of Thought, Self-Consistency
- `reasoning/evaluation/rubric.py` -> LLM-as-judge evaluation with preset rubrics

**How to use in Cursor:**
Run `python scripts/evaluate_agent.py --rubric research` to score any agent output.

---

### Chapter 11: Agent Planning and Feedback

**Core Concept:** Sequential and stepwise planners decompose goals into
executable steps. Feedback loops enable continuous improvement.

**Implementation:**
- `planning/sequential/planner.py` -> `SequentialPlanner` (full plan upfront)
- `planning/sequential/planner.py` -> `StepwisePlanner` (adaptive, one step at a time)
- `planning/feedback/loop.py` -> `FeedbackLoop` with auto-critique and persistent storage

**How to use in Cursor:**
Ask Cursor to "use the sequential planner to break down [complex task]
into steps and execute them."

---

## Adding a New Agent: Checklist

1. Create profile YAML in `agents/profiles/your_agent.yaml`
2. Define any new tools in `agents/tools/your_tool.py`
3. (Optional) Create a behavior tree for complex workflows
4. (Optional) Add evaluation rubric preset
5. Test with: `python scripts/run_agent.py --profile your_agent.yaml --task "..."  --trace`
6. Evaluate with: `python scripts/evaluate_agent.py --profile your_agent.yaml --task "..." --feedback`

## Adding Knowledge to an Agent

1. Place documents in `data/knowledge_base/`
2. Run `python scripts/index_knowledge.py`
3. In your agent code, use the RAG pipeline to retrieve context
4. Inject context into the agent's system prompt before each LLM call
