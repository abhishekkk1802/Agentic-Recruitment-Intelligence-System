# LLM-Powered File System Assistant & RAG Job Matching System

A comprehensive AI system combining file operations, RAG-based resume matching, a LangGraph conversational agent, and MCP (Model Context Protocol) integration.

## Project Overview

The project is built across four phases, each building on the last:

### Phase 1: LLM Function Calling System
- **fs_tools.py** — File system operations: read_file, list_files, write_file, search_in_file (PDF, TXT, DOCX)
- **llm_file_assistant.py** — OpenAI/OpenRouter integration with function calling for natural language file operations

### Phase 2: RAG Job Matching System
- **resume_rag.py** — Document chunking, OpenAI embeddings, ChromaDB vector storage with section-aware metadata extraction
- **job_matcher.py** — Hybrid scoring (semantic 40%, skills 25%, experience 15%, education 10%, keywords 10%) with explainable match reasoning
- **test_rag_system.py** — End-to-end system testing and benchmarking
- **rag_analysis.ipynb** — Jupyter notebook for performance analysis

### Phase 3: LangGraph Conversational Agent
- **agent_workflow.py** — 19-node LangGraph state machine: intent classification, JD parsing, requirement extraction, semantic search, candidate ranking, report generation, human feedback loop, 3-round screening, candidate comparison, ranking explanation, interview question generation
- **agent_state.py** — TypedDict state schema with 54 fields: conversation history (add_messages reducer), job requirements, candidate ranking (replace-semantics via `ranked_candidates`), refinement tracking, explainability reports, screening state
- **explainability.py** — Deterministic explainability layer: per-candidate match reports with strengths and gaps, borderline detection with improvement suggestions, side-by-side comparison tables, head-to-head ranking explanations, ranking-delta computation for refinement transparency
- **matching_agent.py** — Main entry point with CLI chat interface, checkpointer persistence, human-in-the-loop via `interrupt_before`, `--use-mcp` flag for MCP integration
- **screening_pipeline.py** — 3-round progressive screening: initial screen (top 10), deep analysis (enhanced scoring), final recommendation (hire/no-hire with confidence)
- **agent_tools.py** — 15 tools as LangChain StructuredTool functions: file ops, RAG search, job matching, requirement extraction, candidate comparison, interview questions
- **simple_workflow.py** — Legacy 2-node agent/tools loop (selectable via `--workflow simple`)
- **demo_matching_agent.py** — Single-turn demo queries
- **test_matching_agent.py** — Unit and integration test suite

### Phase 4: MCP Server Integration
- **filesystem_mcp_server.py** — MCP server (JSON-RPC 2.0 over stdio) exposing 6 tools: read_file, list_files, write_file, search_in_file, watch_directory (change detection), batch_process (multi-file read/search/stats)
- **mcp_client.py** — MCP client wrapper that spawns the server, discovers tools, and bridges them as LangChain StructuredTool objects
- **mcp_test_scenarios.py** — 6 test scenarios covering tool discovery, file read, directory listing, watch_directory, batch_process, and the LangChain tool bridge
- **demo_script.md** — Step-by-step demo video script for the conversational agent

## Quick Start

### Prerequisites
- Python 3.8+
- OpenRouter API key (provides access to OpenAI, Anthropic, and other models)

### Installation

```bash
cd "LLM-Powered File System Assistant"

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-or-v1-your-openrouter-key-here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

### Running the Systems

#### Phase 1: LLM File Assistant
```bash
python llm_file_assistant.py
```

#### Phase 2: RAG Job Matching System
```bash
python test_rag_system.py
jupyter notebook rag_analysis.ipynb
```

#### Phase 3: LangGraph Conversational Agent
```bash
# Interactive chat (recommended)
python matching_agent.py --interactive --resume-dir examples/resumes

# Skip re-ingestion if vector DB already populated
python matching_agent.py --interactive --skip-ingest

# Run without the human-feedback pause (runs straight through)
python matching_agent.py --interactive --no-hitl --skip-ingest

# Legacy simple workflow
python matching_agent.py --interactive --workflow simple

# Demo queries
python demo_matching_agent.py --examples

# Test suite
python test_matching_agent.py
```

#### Phase 4: MCP Server Integration
```bash
# Inspect the MCP server's tool catalogue
python filesystem_mcp_server.py --inspect

# MCP client smoke test (starts server, calls each tool)
python mcp_client.py

# Full MCP test scenarios (6 scenarios)
python mcp_test_scenarios.py

# Agent with MCP (file operations routed through MCP server)
python matching_agent.py --interactive --use-mcp --skip-ingest
```

## Usage Examples

### Conversational Agent Queries

```
Find me candidates with Python and Django experience for a senior developer role, minimum 5 years
```
The agent classifies intent, parses the job description, extracts must-have vs nice-to-have requirements, searches the vector store semantically, ranks candidates with hybrid scoring, generates a report with strengths and gaps, then pauses for feedback.

```
also require AWS and raise the bar to 70%
```
At the feedback checkpoint, this refinement adds AWS as a hard requirement and raises the minimum score. The agent re-searches, re-ranks, and explains what changed (who dropped out, who moved up, score deltas).

```
Compare the top 3 matches side by side
```
Builds a side-by-side comparison table with score, verdict, hard-requirement coverage, preferred extras, experience and education status, plus a recommendation.

```
Why did the top candidate rank higher than the others?
```
Explains the score difference by attributing it to concrete factors: hard requirement coverage, skill overlap, experience alignment, preferred extras. Flags small gaps as noise.

```
Run the full screening process
```
Triggers 3-round progressive screening:
- Round 1: initial screen narrows to top 10 by score threshold
- Round 2: deep analysis with enhanced scoring (content depth, skill diversity, experience alignment)
- Round 3: hire/no-hire recommendations with confidence levels, key strengths, next steps, and gap analysis for borderline finalists

```
Generate interview questions for the top candidate
```
Produces technical, behavioral, and situational questions tailored to the candidate's skills and experience gaps.

### Chat Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands and example queries |
| `/stats` | Show system statistics (RAG chunks, tools, MCP status) |
| `/reports` | Detailed match reports with strengths, gaps, improvement suggestions |
| `/threads` | List conversation threads |
| `/approve` | Approve current results at the feedback checkpoint |
| `/refine <criteria>` | Adjust criteria and re-rank (e.g. `/refine also require React`) |
| `/screen` | Run multi-round screening to a hire/no-hire call |
| `/export` | Export conversation history to JSON |
| `/clear` | Start a new conversation |
| `/quit` | End the chat session |

### MCP Server (Phase 4)

```python
from mcp_client import MCPClient

# Start the MCP server and connect
client = MCPClient()
client.start()

# Call MCP tools
files = client.call_tool("list_files", {"directory": "examples/resumes", "extension": ".txt"})
content = client.call_tool("read_file", {"filepath": "examples/resumes/john_doe_resume.txt"})
changes = client.call_tool("watch_directory", {"directory": "examples/resumes", "extensions": [".txt"]})
batch = client.call_tool("batch_process", {
    "filepaths": ["examples/resumes/john_doe_resume.txt", "examples/resumes/sarah_smith_resume.txt"],
    "operation": "search", "keyword": "Python"
})

# Get LangChain StructuredTool objects for the agent
tools = client.get_tools()  # 6 tools with matching names/schemas

client.stop()
```

### RAG Job Matching (Phase 2, direct API)

```python
from resume_rag import ResumeRAG
from job_matcher import JobMatcher

rag = ResumeRAG(vector_db="chromadb", embedding_model="openai")
rag.process_resume_directory("examples/resumes")

matcher = JobMatcher(rag_system=rag)
matches = matcher.find_matches(
    job_description=open("examples/job_descriptions/senior_python_developer.txt").read(),
    job_title="Senior Python Developer", k=10, min_score=0.3
)

for match in matches["top_matches"]:
    print(f"{match['candidate_name']}: {match['match_score']:.1f}%")
    print(f"  Skills: {', '.join(match['matched_skills'])}")
    print(f"  Reasoning: {match['reasoning']}")
```

## Project Structure

```
LLM-Powered File System Assistant/
├── Phase 1: LLM Function Calling
│   ├── fs_tools.py                    # File operations (read, list, write, search)
│   └── llm_file_assistant.py          # LLM chat with function calling
│
├── Phase 2: RAG Job Matching
│   ├── resume_rag.py                   # Document chunking, embeddings, ChromaDB
│   ├── job_matcher.py                 # Hybrid scoring and match reasoning
│   ├── test_rag_system.py            # End-to-end testing
│   ├── rag_analysis.ipynb            # Jupyter performance analysis
│   └── rag_test_results.json         # Stored test results
│
├── Phase 3: LangGraph Agent
│   ├── matching_agent.py              # Main entry point, CLI chat, MCP flag
│   ├── agent_workflow.py              # 19-node LangGraph state machine
│   ├── agent_state.py                # State schema (54 fields, reducers)
│   ├── agent_tools.py                 # 15 LangChain StructuredTool functions
│   ├── explainability.py             # Match reports, comparison, ranking deltas
│   ├── screening_pipeline.py         # 3-round progressive screening
│   ├── simple_workflow.py            # Legacy 2-node workflow (fallback)
│   ├── demo_matching_agent.py        # Demo queries
│   ├── demo_script.md                # Demo video script
│   └── test_matching_agent.py        # Test suite
│
├── Phase 4: MCP Integration
│   ├── filesystem_mcp_server.py       # MCP server (JSON-RPC 2.0, 6 tools)
│   ├── mcp_client.py                 # MCP client + LangChain tool bridge
│   └── mcp_test_scenarios.py         # 6 MCP test scenarios
│
├── examples/
│   ├── resumes/                      # 30 resume files (.txt)
│   └── job_descriptions/             # 5 job description files (.txt)
│
├── .env                               # API credentials (gitignored)
├── .gitignore
├── requirements.txt                  # All dependencies including MCP
├── vector_db/                        # ChromaDB storage (auto-created)
└── venv/                             # Virtual environment
```

## Architecture

### LangGraph Agent Workflow (Phase 3)

The graph has 19 nodes with conditional routing based on intent classification:

```
START -> classify_intent
  |-- job_match  -> parse_job -> extract_requirements -> search_resumes
  |                -> rank_candidates -> generate_report -> human_feedback_node
  |                     |-- refine  -> adjust_criteria -> search_resumes (loop)
  |                     |-- approve -> multi_round_screening -> deep_analysis
  |                                   -> final_recommendation -> END
  |-- hybrid    -> ingest_resumes -> index_check -> parse_job
  |-- file_ops  -> file_agent <-> file_tools -> END
  |-- compare   -> compare_candidates -> END
  |-- explain   -> explain_ranking -> END
  |-- questions -> generate_questions -> END
  |-- screen    -> multi_round_screening -> deep_analysis -> final_recommendation
  |-- refine    -> adjust_criteria -> search_resumes
  |-- clarify   -> END
  '-- error     -> handle_error -> END
```

Human-in-the-loop uses `interrupt_before=["human_feedback_node"]` (langgraph 0.2.45 pattern). The caller writes feedback via `update_state()`, then resumes with `invoke(None, config)`.

State persistence via MemorySaver (default) or SqliteSaver (with `--checkpoint-db`).

### MCP Integration (Phase 4)

The agent's file operations can be routed through the MCP server instead of direct function calls:

```
Agent (LangGraph)
  |
  |-- --use-mcp flag
  |     v
  MCPClient (mcp_client.py)
  |   - spawns filesystem_mcp_server.py as subprocess
  |   - connects via stdio (JSON-RPC 2.0)
  |   - discovers tools via tools/list
  |   - bridges as LangChain StructuredTool objects
  v
Filesystem MCP Server (filesystem_mcp_server.py)
  |
  |-- read_file      -> fs_tools.read_file()
  |-- list_files     -> fs_tools.list_files()
  |-- write_file     -> fs_tools.write_file()
  |-- search_in_file -> fs_tools.search_in_file()
  |-- watch_directory -> snapshot diff (added/modified/removed)
  '-- batch_process  -> multi-file read/search/stats
```

The graph structure is identical in both modes (19 nodes). When MCP is active, the `file_tools` ToolNode uses MCP-backed tools; when inactive, it uses the direct `agent_tools` functions.

### Explainability Layer (Phase 3, Part C.2)

`explainability.py` provides deterministic, LLM-free analysis:

- **Match reports**: per-candidate strengths, gaps, must-have/nice-to-have coverage, experience assessment, improvement suggestions for borderline candidates (score 55-70)
- **Comparison tables**: fixed-width side-by-side layout with score, verdict, coverage, experience, education
- **Head-to-head explanation**: attributes score differences to hard requirement coverage, skill overlap, experience alignment, preferred extras
- **Ranking deltas**: tracks entered/dropped/moved_up/moved_down candidates with score deltas and a narrative explaining what a refinement changed

## Assignment Completion Status

| Phase | Component | Requirement | Implementation | Status |
|-------|-----------|-------------|----------------|--------|
| 1 | Part A (60%) | File system tools | fs_tools.py — read_file, list_files, write_file, search_in_file | Done |
| 1 | Part B (40%) | LLM integration | llm_file_assistant.py — OpenAI/OpenRouter function calling | Done |
| 2 | Part A (50%) | RAG system | resume_rag.py — ChromaDB, OpenAI embeddings, section-aware chunking | Done |
| 2 | Part B (50%) | Job matcher | job_matcher.py — hybrid scoring, 5-component weighted, explainable reasoning | Done |
| 3 | Part A (40%) | Agent architecture | agent_workflow.py — 19-node LangGraph, agent_state.py — 54-field state, agent_tools.py — 15 tools | Done |
| 3 | Part A (40%) | Graph structure | START -> Parse JD -> Extract Requirements -> Search -> Rank -> Report -> Human Feedback -> END | Done |
| 3 | Part A (40%) | Custom tools | extract_requirements, compare_candidates, generate_interview_questions | Done |
| 3 | Part B (30%) | Conversational interface | Natural language queries, CLI chat with /commands | Done |
| 3 | Part B (30%) | Iterative refinement | Mid-conversation criteria adjustment, re-ranking with explained ranking deltas | Done |
| 3 | Part C (30%) | Multi-round screening | 3-stage: initial screen, deep analysis, final hire/no-hire recommendation | Done |
| 3 | Part C (30%) | Explainability | Match reports with strengths/gaps, improvement suggestions for borderline candidates | Done |
| 4 | Part A (50%) | MCP server | filesystem_mcp_server.py — JSON-RPC 2.0, 6 tools, resource discovery | Done |
| 4 | Part A (50%) | watch_directory | Snapshot diff: added/modified/removed files between calls | Done |
| 4 | Part A (50%) | batch_process | Multi-file read/search/stats in one call | Done |
| 4 | Part B (30%) | Agent refactoring | --use-mcp flag, MCP client integration, same functionality via MCP | Done |
| 4 | — | Test scenarios | mcp_test_scenarios.py — 6 scenarios, all passing | Done |
| 4 | — | Diagram | Mermaid auto-generated from compiled graph | Done |
| — | — | Dataset | 30 resumes + 5 job descriptions | Done |

## Supported File Formats

| Format | Library | Capabilities |
|--------|---------|--------------|
| TXT | Built-in | Full text extraction |
| PDF | PyPDF2 | Text extraction from most PDFs |
| DOCX | python-docx | Microsoft Word documents |

## Dependencies

Key dependencies (see `requirements.txt` for full pinned versions):

| Package | Purpose |
|---------|---------|
| langgraph | Graph-based agent workflow orchestration |
| langchain-core | Tool abstractions, message types |
| langchain-openai | OpenAI/OpenRouter LLM integration |
| chromadb | Vector database for embeddings |
| mcp | MCP SDK — server + client (JSON-RPC 2.0) |
| grandalf | Graph layout for ASCII diagram rendering |
| openai | OpenAI/OpenRouter API client |
| PyPDF2 | PDF text extraction |
| python-docx | DOCX text extraction |

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-or-v1-...           # OpenRouter API key
OPENAI_BASE_URL=https://openrouter.ai/api/v1  # OpenRouter endpoint
```

### Model Selection

The agent defaults to `gpt-4o-mini`. To use a different model via OpenRouter, pass a config file:

```bash
python matching_agent.py --interactive --config-file config.json --skip-ingest
```

where `config.json` contains:
```json
{"model": "openai/gpt-4o-mini"}
```

OpenRouter uses `provider/model` naming (e.g. `openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`).

## Troubleshooting

**"No API key found"**
- Check `.env` exists and contains `OPENAI_API_KEY` and `OPENAI_BASE_URL`

**"BaseTool.__call__() got an unexpected keyword argument"**
- This was a langchain-core 0.3.x issue where `@tool` functions needed `.invoke()` instead of direct calls. Fixed in the current code — all tool calls use `.invoke({...})`.

**MCP server fails to start**
- Ensure `mcp` is installed: `pip install mcp==2.3.0`
- Check that `filesystem_mcp_server.py` is in the same directory as `mcp_client.py`
- Run `python filesystem_mcp_server.py --inspect` to verify the server works standalone

**Agent pauses and never resumes**
- The human-in-the-loop checkpoint pauses after report generation. Type `approve`, `refine <criteria>`, or use `/approve`, `/refine`, `/screen` commands.
- To disable the pause entirely: `--no-hitl`

**Resume ingestion takes too long**
- Use `--skip-ingest` if the vector DB is already populated from a previous run

