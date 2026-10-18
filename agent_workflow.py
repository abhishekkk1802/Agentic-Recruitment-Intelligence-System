"""
Agent Workflow for LangGraph Matching Agent
===========================================

Defines the LangGraph workflow with nodes and conditional edges for the
conversational job matching agent with multi-round screening capabilities.

This module provides:
- Graph construction and workflow logic
- Node implementations for each workflow step
- Conditional routing and edge logic
- Human-in-the-loop integration points

Graph shape (the required Milestone 3 pipeline is the spine running down the
middle; the other branches hang off intent classification):

    START
      -> classify_intent
           |-- job_match --> parse_job -> extract_requirements
           |                     -> search_resumes -> rank_candidates
           |                     -> generate_report -> human_feedback_node
           |                          |-- refine --> adjust_criteria --,
           |                          |                   (re-search) <'
           |                          |-- screen --> multi_round_screening
           |                          |                -> deep_analysis
           |                          |                -> final_recommendation
           |                          '-- approve/export --> END
           |-- hybrid   --> ingest_resumes -> index_check -> parse_job
           |-- file_ops --> file_agent <-> file_tools -> END
           |-- compare  --> compare_candidates -> END
           |-- explain  --> explain_ranking -> END
           |-- questions--> generate_questions -> END
           |-- screen   --> multi_round_screening (reuses existing ranking)
           |-- refine   --> adjust_criteria
           |-- clarify  --> END
           '-- error    --> handle_error -> END

Human-in-the-loop note: langgraph 0.2.45 ships the ``Interrupt`` type but not
the ``interrupt()`` function or ``Command`` primitive from later releases. The
loop is therefore implemented with the version-appropriate pattern: the graph
is compiled with ``interrupt_before=[HUMAN_FEEDBACK_NODE]``, the caller writes
feedback in with ``update_state()``, and execution resumes via
``invoke(None, config)``.
"""

import logging
from typing import Dict, List, Optional, Any, Literal, Sequence, Tuple
import json
import re
import time

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Interrupt  # retained: type import used by HITL tooling
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from agent_state import (
    MatchingAgentState, 
    update_workflow_phase, 
    is_circuit_breaker_triggered,
    extract_user_message_content,
    get_current_ranking,
    get_active_requirements,
    JobRequirementDict,
    HumanFeedbackData
)
from agent_tools import (
    get_tools_by_category,
    initialize_tools,
    ALL_TOOLS,
    SAFE_TOOLS,
    WRITE_TOOLS
)
from screening_pipeline import MultiRoundScreeningPipeline
import explainability as explain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Intent classification vocabulary
# =============================================================================
# A deterministic pre-pass runs before the LLM classifier. It only fires on
# unambiguous phrasing, which keeps demo transcripts reproducible and stops a
# flaky model response from derailing a conversation. Anything it is not sure
# about falls through to the LLM.

_INTENT_PATTERNS: List[Tuple[str, str]] = [
    ("explain",   r"\bwhy\s+(?:did|does|is|was)\b.*\b(?:rank|higher|lower|above|below|better|ahead)\b"),
    ("explain",   r"\bwhy\s+(?:did|does|is|was)\b.*\bthan\b"),
    ("explain",   r"\bexplain\b.*\b(?:ranking|rank|score|scoring|difference)\b"),
    ("compare",   r"\bcompar(?:e|ison)\b"),
    ("compare",   r"\bside[\s-]by[\s-]side\b"),
    ("compare",   r"\bhead[\s-]to[\s-]head\b"),
    ("questions", r"\b(?:interview|screening)\s+questions?\b"),
    ("questions", r"\bgenerate\b.*\bquestions?\b"),
    ("screen",    r"\b(?:multi[\s-]?round|full|deep|three[\s-]round)\s+screen(?:ing)?\b"),
    ("screen",    r"\b(?:run|start|begin|do)\b.*\bscreening\b"),
    ("screen",    r"\bhire\b.*\brecommendation\b"),
    ("screen",    r"\b(?:hire|no[\s-]hire)\s+(?:decision|recommendation)\b"),
    ("refine",    r"\b(?:also|additionally)\s+(?:need|require|want)\b"),
    ("refine",    r"\b(?:re-?rank|rerank)\b"),
    ("refine",    r"\b(?:narrow|tighten|loosen|broaden|relax)\b"),
    ("refine",    r"\b(?:add|drop|remove)\b.*\brequirement\b"),
    ("refine",    r"\b(?:raise|lower)\b.*\b(?:bar|threshold|score)\b"),
    ("hybrid",    r"\b(?:ingest|index|re-?index|load)\b.*\bresumes?\b"),
    ("file_ops",  r"\b(?:list|read|write|save)\b.*\bfile"),
    ("file_ops",  r"\bwhat files\b"),
]

_VALID_INTENTS = [
    "job_match", "file_ops", "hybrid", "compare",
    "explain", "questions", "screen", "refine", "clarify",
]

# Name of the human-in-the-loop checkpoint node.
#
# It is deliberately *not* "human_feedback": LangGraph rejects any node whose
# name collides with a state key, and `human_feedback` is already a field on
# MatchingAgentState. Keeping this in one constant stops the node name and the
# `interrupt_before` target from drifting apart, which would silently disable
# the feedback loop.
HUMAN_FEEDBACK_NODE = "human_feedback_node"

# Vocabulary used by the regex fallback when parsing a refinement request.
_SKILL_VOCABULARY = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "rust", "scala", "ruby", "php", "swift", "kotlin", "r", "matlab", "sql",
    "react", "angular", "vue", "svelte", "django", "flask", "fastapi", "spring",
    "node.js", "node", "express", "rails", ".net", "graphql", "rest api",
    "tensorflow", "pytorch", "scikit-learn", "keras", "pandas", "numpy", "spark",
    "hadoop", "kafka", "airflow", "dbt", "tableau", "power bi",
    "docker", "kubernetes", "terraform", "ansible", "jenkins", "git", "ci/cd",
    "aws", "azure", "gcp", "linux", "bash",
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "snowflake", "bigquery",
    "machine learning", "deep learning", "nlp", "computer vision", "mlops",
    "data science", "devops", "security", "penetration testing", "agile", "scrum",
]


class MatchingAgentWorkflow:
    """
    LangGraph workflow implementation for the conversational job matching agent.
    
    Orchestrates the complete matching process from intent classification through
    multi-round screening with human-in-the-loop feedback and explainable results.
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        checkpointer: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        mcp_client: Optional[Any] = None
    ):
        """
        Initialize the matching agent workflow.
        
        Args:
            llm: Language model for conversation and reasoning
            checkpointer: Persistent state checkpointer. Falls back to an
                in-memory saver, which still enables multi-turn refinement
                within a process.
            config: Configuration parameters
            mcp_client: Optional MCP client. When provided, file-system
                tools (read, list, search, write, watch, batch) are routed
                through the MCP server instead of called directly. This
                demonstrates the Model Context Protocol integration without
                changing the graph structure — the tool names and schemas
                match, so ToolNode and tools_condition work unchanged.
        """
        self.config = config or {}
        self.mcp_client = mcp_client
        
        # Determine the tool set for file operations. When an MCP client is
        # provided, file operations go through the MCP server; otherwise the
        # direct fs_tools-backed tools from agent_tools are used.
        if mcp_client is not None:
            self._file_tools = mcp_client.get_all_tools()
            logger.info("File operations routed through MCP server (%d tools)",
                        len(self._file_tools))
        else:
            self._file_tools = ALL_TOOLS

        # Initialize LLM
        self.llm = llm or ChatOpenAI(
            model=self.config.get("model", "gpt-4o-mini"),
            temperature=self.config.get("temperature", 0.1),
            max_tokens=self.config.get("max_tokens", 4000)
        )

        # Bind the tool surface to the model. Without this, `tools_condition`
        # can never fire and the ToolNode is unreachable, which is what left
        # the tool layer dead in the previous revision. When MCP is active,
        # the MCP-backed file tools replace the direct ones in the bound set.
        tools_to_bind = list(ALL_TOOLS)
        if mcp_client is not None:
            mcp_tool_names = {t.name for t in self._file_tools}
            tools_to_bind = [t for t in ALL_TOOLS if t.name not in mcp_tool_names]
            tools_to_bind.extend(self._file_tools)

        try:
            self.llm_with_tools = self.llm.bind_tools(tools_to_bind)
        except Exception as exc:
            logger.warning("Could not bind tools to the LLM: %s", exc)
            self.llm_with_tools = self.llm
        
        # Initialize checkpointer.
        # NOTE: SqliteSaver.from_conn_string() is a *context manager* in
        # langgraph-checkpoint-sqlite 2.x, so it cannot be assigned directly.
        # Callers that want SQLite persistence should construct the saver and
        # pass it in; the default here is an in-memory saver.
        self.checkpointer = checkpointer or MemorySaver()

        # Human-in-the-loop is on by default because the brief requires the
        # feedback loop. Turning it off keeps the same graph shape but lets the
        # run complete end-to-end without pausing, which suits batch demos.
        self.enable_hitl = self.config.get("enable_hitl", True)

        # Multi-round screening engine (Part C.1).
        self.screening_pipeline = MultiRoundScreeningPipeline(
            config=self.config.get("screening_config")
        )
        
        # Build workflow graph
        self.graph = self._build_graph()
        
        logger.info(
            "MatchingAgentWorkflow initialized successfully (hitl=%s, mcp=%s)",
            self.enable_hitl, self.mcp_client is not None
        )
    
    def _build_graph(self):
        """
        Build the LangGraph workflow with nodes and conditional edges.
        
        Returns:
            Compiled StateGraph for the matching agent
        """
        # Create graph builder
        builder = StateGraph(MatchingAgentState)
        
        # ---- Core matching pipeline (the spine required by the brief) -----
        builder.add_node("classify_intent", self._classify_intent_node)
        builder.add_node("parse_job", self._parse_job_node)
        builder.add_node("extract_requirements", self._extract_requirements_node)
        builder.add_node("search_resumes", self._search_resumes_node)
        builder.add_node("rank_candidates", self._rank_candidates_node)
        builder.add_node("generate_report", self._generate_report_node)
        # NOTE: the node is named `human_feedback_node`, not `human_feedback`,
        # because `human_feedback` is already a key in MatchingAgentState and
        # LangGraph forbids a node sharing a name with a state field.
        builder.add_node("human_feedback_node", self._human_feedback_node)
        builder.add_node("adjust_criteria", self._adjust_criteria_node)

        # ---- Multi-round screening (Part C.1) ----------------------------
        builder.add_node("multi_round_screening", self._multi_round_screening_node)
        builder.add_node("deep_analysis", self._deep_analysis_node)
        builder.add_node("final_recommendation", self._final_recommendation_node)

        # ---- Resume ingestion branch ------------------------------------
        builder.add_node("ingest_resumes", self._ingest_resumes_node)
        builder.add_node("index_check", self._index_check_node)

        # ---- File operations branch (Milestone 1 tools) ------------------
        # When an MCP client is connected, the ToolNode dispatches file
        # operations to the MCP server process; otherwise it calls the
        # fs_tools functions directly.
        builder.add_node("file_agent", self._file_agent_node)
        builder.add_node("file_tools", ToolNode(self._file_tools))

        # ---- Specialised analysis (Part B.1) ----------------------------
        builder.add_node("compare_candidates", self._compare_candidates_node)
        builder.add_node("explain_ranking", self._explain_ranking_node)
        builder.add_node("generate_questions", self._generate_questions_node)

        # ---- Error handling ---------------------------------------------
        builder.add_node("handle_error", self._handle_error_node)

        # Wire everything up.
        self._add_edges(builder)

        # Compile. `interrupt_before` is the langgraph 0.2.x mechanism for
        # human-in-the-loop: execution halts before the named node so the
        # caller can inspect the report and inject feedback.
        compile_kwargs: Dict[str, Any] = {"checkpointer": self.checkpointer}
        if self.enable_hitl:
            compile_kwargs["interrupt_before"] = [HUMAN_FEEDBACK_NODE]

        return builder.compile(**compile_kwargs)
    
    def _add_edges(self, builder: StateGraph) -> None:
        """Add edges and conditional routing to the graph."""
        
        # Entry point
        builder.add_edge(START, "classify_intent")
        
        # Intent classification routing
        builder.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "job_match": "parse_job",
                "file_ops": "file_agent", 
                "hybrid": "ingest_resumes",
                "compare": "compare_candidates",
                "explain": "explain_ranking",
                "questions": "generate_questions",
                "screen": "multi_round_screening",
                "refine": "adjust_criteria",
                "clarify": END,
                "error": "handle_error"
            }
        )
        
        # Job matching pipeline. Parsing can fail to find a job description,
        # in which case the run ends with a clarification rather than pushing
        # an empty JD through the rest of the pipeline.
        builder.add_conditional_edges(
            "parse_job",
            self._route_after_parse_job,
            {
                "continue": "extract_requirements",
                "clarify": END,
                "error": "handle_error"
            }
        )
        builder.add_edge("extract_requirements", "search_resumes")
        builder.add_conditional_edges(
            "search_resumes",
            self._route_after_search,
            {
                "continue": "rank_candidates",
                "error": "handle_error"
            }
        )
        builder.add_edge("rank_candidates", "generate_report")
        
        # Human feedback loop
        builder.add_conditional_edges(
            "generate_report",
            self._route_after_report,
            {
                "human_feedback": HUMAN_FEEDBACK_NODE,
                "screening": "multi_round_screening",
                "end": END
            }
        )
        
        builder.add_conditional_edges(
            HUMAN_FEEDBACK_NODE,
            self._route_after_feedback,
            {
                "refine": "adjust_criteria",
                "approve": "multi_round_screening", 
                "end": END
            }
        )
        
        # Refinement re-enters the pipeline at the search step so adjusted
        # criteria actually influence retrieval, not just re-sorting.
        builder.add_edge("adjust_criteria", "search_resumes")
        
        # Multi-round screening
        builder.add_edge("multi_round_screening", "deep_analysis")
        builder.add_conditional_edges(
            "deep_analysis",
            self._route_after_analysis,
            {
                "final": "final_recommendation",
                "end": END
            }
        )
        
        builder.add_edge("final_recommendation", END)
        
        # Hybrid ingest pipeline
        builder.add_edge("ingest_resumes", "index_check")
        builder.add_conditional_edges(
            "index_check",
            self._route_after_index,
            {
                "ready": "parse_job",
                "error": "handle_error"
            }
        )
        
        # File operations subgraph. This is the branch that exercises LLM tool
        # calling, hence the bound model in `_file_agent_node`.
        builder.add_conditional_edges(
            "file_agent",
            tools_condition,
            {
                "tools": "file_tools",
                "__end__": END
            }
        )
        builder.add_edge("file_tools", "file_agent")
        
        # Specialized analysis
        builder.add_edge("compare_candidates", END)
        builder.add_edge("explain_ranking", END)
        builder.add_edge("generate_questions", END)
        
        # Error handling
        builder.add_edge("handle_error", END)

    # =========================================================================
    # Diagram export (submission deliverable: state machine diagram)
    # =========================================================================

    def get_mermaid(self) -> str:
        """Return the compiled graph as a Mermaid definition."""
        return self.graph.get_graph().draw_mermaid()

    def get_ascii(self) -> str:
        """
        Return the compiled graph as ASCII art.

        Requires the optional `grandalf` package. Rather than raising, an
        explanatory string is returned so a diagram export never fails purely
        because an optional renderer is absent.
        """
        try:
            return self.graph.get_graph().draw_ascii()
        except ImportError:
            return (
                "ASCII rendering requires the 'grandalf' package "
                "(pip install grandalf). The Mermaid diagram is unaffected."
            )

    def save_diagram_png(self, path: str) -> str:
        """
        Render the graph to PNG.

        Uses the Mermaid rendering service and therefore needs network access.
        Callers should be prepared for this to fail and fall back to the
        Mermaid source or ASCII output.
        """
        png = self.graph.get_graph().draw_mermaid_png()
        with open(path, "wb") as handle:
            handle.write(png)
        return path
    
    # =========================================================================
    # Node Implementations
    # =========================================================================
    
    def _classify_intent_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """Classify user intent and determine workflow path."""
        try:
            user_message = extract_user_message_content(state)

            intent = self._pattern_classify(user_message)
            if intent is None:
                intent = self._llm_classify(user_message)

            # A refinement only makes sense when there is something to refine.
            if intent == "refine" and not get_current_ranking(state):
                intent = "job_match"

            updates = update_workflow_phase(state, "classify")
            updates.update({
                "intent": intent,
                "raw_user_query": user_message,
                "processed_query": user_message,
                # Reset the per-turn loop guard. `iteration_count` persists in
                # the checkpoint, so without this a long conversation would
                # eventually trip the circuit breaker on an innocent turn.
                "iteration_count": 0,
            })

            # A brand new search invalidates any screening already performed.
            if intent == "job_match":
                updates.update({
                    "screening_round": 0,
                    "screening_complete": False,
                    "screening_requested": False,
                })
            
            logger.info(f"Classified intent as: {intent}")
            return updates
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {
                "intent": "error",
                "error": str(e),
                "workflow_phase": "error"
            }

    def _pattern_classify(self, message: str) -> Optional[str]:
        """
        Deterministic intent pre-pass.

        Returns an intent only when the phrasing is unambiguous, otherwise
        None so the LLM classifier gets a turn.
        """
        if not message:
            return None
        lowered = message.lower()
        for intent, pattern in _INTENT_PATTERNS:
            if re.search(pattern, lowered):
                return intent
        return None

    def _llm_classify(self, message: str) -> str:
        """Classify intent with the LLM, defaulting to clarification."""
        system_prompt = """You are an intelligent job matching assistant. Classify the user's intent into one of these categories:

- job_match: Find candidates for a job description or job requirements
- file_ops: File system operations (read, list, write, search files)
- hybrid: Ingest new resumes AND find matches
- compare: Compare multiple candidates head-to-head  
- explain: Explain why one candidate ranked higher than another
- questions: Generate interview questions for a candidate
- screen: Run the multi-round screening process or produce hire/no-hire recommendations
- refine: Adjust or change the search criteria of an existing search
- clarify: Request needs clarification or is ambiguous

Respond with just the category name."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=message)
            ])
            intent = (response.content or "").strip().lower()
            # Models sometimes answer in a sentence; recover the label.
            for candidate in _VALID_INTENTS:
                if candidate in intent:
                    return candidate
            return "clarify"
        except Exception as exc:
            logger.warning("LLM intent classification failed (%s); defaulting to job_match", exc)
            # A long message is almost certainly a job description.
            return "job_match" if len(message) > 100 else "clarify"
    
    def _parse_job_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """Parse job description from user input or file."""
        try:
            user_message = state.get("raw_user_query", "")
            
            # Extract job description - could be inline or reference to file
            job_description = None
            job_title = ""
            
            # Check if user provided job description directly
            if len(user_message) > 100:  # Likely contains JD
                job_description = user_message
            else:
                # Look for file reference patterns
                file_patterns = [
                    r"job[_\s]?description[_\s]?file[:\s]*([^\s]+)",
                    r"jd[_\s]?file[:\s]*([^\s]+)",
                    r"read[_\s]*(?:the[_\s]*)?(?:job[_\s]*)?(?:from[_\s]*)?([^\s]+\.txt)",
                    r"file[:\s]*([^\s]+\.txt)",
                    r"([^\s]+\.txt)",
                ]
                
                for pattern in file_patterns:
                    match = re.search(pattern, user_message, re.IGNORECASE)
                    if match:
                        filepath = match.group(1)
                        try:
                            from agent_tools import read_file_tool
                            file_result = read_file_tool.invoke({"filepath": filepath})
                            if file_result.get("success"):
                                job_description = file_result["content"]
                                break
                        except Exception as e:
                            logger.warning(f"Failed to read file {filepath}: {e}")

            # Multi-turn continuity: a follow-up such as "now screen them" has
            # no job description of its own, but the conversation already
            # established one and the checkpoint still holds it.
            if not job_description and state.get("job_description"):
                logger.info("Reusing job description from earlier in the conversation")
                return update_workflow_phase(state, "parse_job")

            # If the message is short but descriptive, still treat it as the
            # brief rather than refusing outright.
            if not job_description and len(user_message.split()) >= 5:
                job_description = user_message
            
            # If still no JD found, ask for clarification
            if not job_description:
                return {
                    "workflow_phase": "clarify",
                    "error": "Please provide a job description either directly in your message or specify a file path to read from.",
                    "final_response": "I need a job description to find matching candidates. Please either:\n1. Paste the job description in your message\n2. Specify a file path (e.g., 'examples/job_descriptions/senior_python_developer.txt')\n3. Or ask me to list available job description files."
                }
            
            # Extract job title if possible
            title_patterns = [
                r"(?:job\s+title|position|role)[:\s]*([^\n]+)",
                r"^([^:\n]+)(?:\s*-\s*[^:\n]+)?$",  # First line pattern
            ]
            
            for pattern in title_patterns:
                match = re.search(pattern, job_description[:200], re.IGNORECASE | re.MULTILINE)
                if match:
                    job_title = match.group(1).strip()
                    break
            
            updates = update_workflow_phase(state, "parse_job")
            updates.update({
                "job_description": job_description,
                "job_title": job_title
            })
            
            logger.info(f"Parsed job description for '{job_title}' ({len(job_description)} chars)")
            return updates
            
        except Exception as e:
            logger.error(f"Job parsing failed: {e}")
            return {
                "workflow_phase": "error",
                "error": str(e)
            }
    
    def _extract_requirements_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """Extract detailed requirements from job description."""
        try:
            job_description = state.get("job_description")
            if not job_description:
                raise ValueError("No job description available for requirement extraction")
            
            # Use the enhanced extract_requirements tool
            from agent_tools import extract_requirements
            
            requirements_result = extract_requirements.invoke({"job_description": job_description})
            
            if "error" in requirements_result and requirements_result.get("error"):
                raise ValueError(f"Requirement extraction failed: {requirements_result['error']}")
            
            # Convert to state format
            basic_req = requirements_result["basic_requirements"]
            enhanced_req = requirements_result["enhanced_classification"]
            
            parsed_requirements = JobRequirementDict(
                title=basic_req["title"],
                skills_required=basic_req["skills_required"],
                skills_preferred=basic_req["skills_preferred"],
                experience_years=enhanced_req["min_experience_years"],
                education_required=basic_req["education_required"],
                must_have_keywords=enhanced_req["must_have_extracted"],
                nice_to_have_keywords=enhanced_req["nice_to_have_extracted"],
                description=job_description
            )
            
            updates = update_workflow_phase(state, "extract_requirements")
            updates.update({
                "parsed_requirements": parsed_requirements,
                # Requirements freshly extracted from a JD become the active
                # criteria; a later refinement will diverge from this baseline.
                "active_requirements": parsed_requirements,
                "tool_calls": [{
                    "tool": "extract_requirements",
                    "timestamp": time.time(),
                    "result": "success"
                }]
            })
            
            logger.info(f"Extracted {len(enhanced_req['must_have_extracted'])} must-have and {len(enhanced_req['nice_to_have_extracted'])} nice-to-have requirements")
            return updates
            
        except Exception as e:
            logger.error(f"Requirement extraction failed: {e}")
            return {
                "workflow_phase": "error",
                "error": str(e)
            }
    
    def _search_resumes_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """Search resumes using semantic similarity."""
        try:
            job_description = state.get("job_description")
            top_k = state.get("top_k", 10)
            
            if not job_description:
                raise ValueError("No job description available for search")

            # Fold any refined criteria into the retrieval query so a
            # mid-conversation adjustment changes what comes back, not just
            # how it is sorted.
            query = self._build_search_query(state, job_description)
            
            # Use semantic search tool
            from agent_tools import semantic_search_resumes
            
            search_results = semantic_search_resumes.invoke({"query": query, "k": min(50, top_k * 2)})
            
            if not search_results or (
                isinstance(search_results, list)
                and len(search_results) == 1
                and isinstance(search_results[0], dict)
                and "error" in search_results[0]
            ):
                raise ValueError("Semantic search failed or returned no results")
            
            updates = update_workflow_phase(state, "search_resumes")
            updates.update({
                "semantic_hits": search_results,
                "tool_calls": [{
                    "tool": "semantic_search_resumes", 
                    "timestamp": time.time(),
                    "result_count": len(search_results)
                }]
            })
            
            logger.info(f"Found {len(search_results)} semantic matches")
            return updates
            
        except Exception as e:
            logger.error(f"Resume search failed: {e}")
            return {
                "workflow_phase": "error",
                "error": str(e)
            }

    def _build_search_query(self, state: MatchingAgentState, job_description: str) -> str:
        """Compose the retrieval query from the JD plus any active refinements."""
        requirements = get_active_requirements(state)
        if not requirements:
            return job_description

        emphasis: List[str] = []
        emphasis.extend(requirements.get("skills_required") or [])
        emphasis.extend(requirements.get("must_have_keywords") or [])
        if not emphasis:
            return job_description

        # Repeating the hard requirements biases the embedding toward them.
        return f"{job_description}\n\nRequired skills emphasis: {', '.join(str(e) for e in emphasis)}"
    
    def _rank_candidates_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """Rank candidates using hybrid matching algorithm."""
        try:
            job_description = state.get("job_description")
            job_title = state.get("job_title", "")
            top_k = state.get("top_k", 10)
            min_score = state.get("min_score", 0.3)
            
            if not job_description:
                raise ValueError("No job description available for ranking")
            
            # Use job matching tool
            from agent_tools import find_job_matches

            # `min_score` is carried as a 0-1 fraction in state but the tool
            # expects a 0-100 percentage.
            min_score_pct = min_score * 100 if min_score <= 1 else min_score
            
            matches_result = find_job_matches.invoke({
                "job_description": job_description,
                "job_title": job_title or "",
                "k": top_k,
                "min_score": min_score_pct
            })
            
            if matches_result.get("error"):
                raise ValueError(f"Job matching failed: {matches_result['error']}")
            
            top_matches = matches_result.get("top_matches", []) or []

            # Apply refinement filters that the underlying matcher does not
            # know about (added skills, raised experience floor).
            top_matches = self._apply_refinement_filters(state, top_matches)

            requirements = get_active_requirements(state)

            # Compute the ranking delta whenever there is a prior ranking to
            # compare against, so a re-rank can be explained rather than just
            # returned (Part B.2).
            previous = state.get("previous_ranking") or []
            ranking_changes = None
            if previous:
                ranking_changes = explain.compute_ranking_delta(
                    previous,
                    top_matches,
                    trigger=state.get("pending_refinement", {}).get(
                        "user_request", "criteria refinement"
                    ) if state.get("pending_refinement") else "criteria refinement",
                    criteria_before=(state.get("pending_refinement") or {}).get("criteria_before"),
                    criteria_after=(state.get("pending_refinement") or {}).get("criteria_after"),
                )

            # Explainability reports for every ranked candidate (Part C.2).
            reports = explain.build_match_reports(top_matches, requirements)
            
            updates = update_workflow_phase(state, "rank_candidates") 
            updates.update({
                # Append-only audit trail of everything ever matched.
                "candidate_matches": top_matches,
                "candidate_shortlist": [m["candidate_name"] for m in top_matches],
                # Authoritative current ranking (replace semantics).
                "ranked_candidates": top_matches,
                "ranking_changes": ranking_changes,
                "match_reports": reports,
                "borderline_candidates": explain.collect_borderline(reports),
                "performance_metrics": {
                    **(state.get("performance_metrics") or {}),
                    "matching_time": matches_result.get("matching_time", 0),
                    "candidates_found": len(top_matches),
                },
                "tool_calls": [{
                    "tool": "find_job_matches",
                    "timestamp": time.time(),
                    "result_count": len(top_matches)
                }],
                # The refinement has now been applied.
                "pending_refinement": None,
            })
            
            logger.info(f"Ranked candidates, found {len(top_matches)} qualified matches")
            return updates
            
        except Exception as e:
            logger.error(f"Candidate ranking failed: {e}")
            return {
                "workflow_phase": "error",
                "error": str(e)
            }

    def _apply_refinement_filters(
        self,
        state: MatchingAgentState,
        matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enforce refinement constraints the base matcher cannot express.

        The matcher scores against the original job description. When the user
        adds a hard requirement mid-conversation ("must also have React"),
        candidates lacking it are dropped here so the refinement has teeth.
        """
        refinement = state.get("pending_refinement") or {}
        adjustments = refinement.get("parsed_adjustments") or {}

        required_skills = [s.lower() for s in (adjustments.get("add_required_skills") or [])]
        min_years = adjustments.get("min_experience_years")

        if not required_skills and not min_years:
            return matches

        kept: List[Dict[str, Any]] = []
        for match in matches:
            haystack = " ".join(
                str(x).lower() for x in (
                    list(match.get("matched_skills") or [])
                    + list(match.get("relevant_excerpts") or [])
                    + [match.get("reasoning", "")]
                )
            )
            if required_skills and not all(skill in haystack for skill in required_skills):
                continue

            if min_years:
                years = match.get("experience_years")
                if years is None:
                    years = (match.get("profile_analysis") or {}).get("experience_years")
                # Only exclude when we can positively establish a shortfall.
                if isinstance(years, (int, float)) and years < min_years:
                    continue

            kept.append(match)

        if not kept:
            logger.warning(
                "Refinement filters removed every candidate; returning the unfiltered "
                "ranking so the user still has something to react to."
            )
            return matches
        return kept
    
    def _generate_report_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """Generate comprehensive match report with explanations."""
        try:
            candidate_matches = get_current_ranking(state)
            job_title = state.get("job_title") or "the role"
            requirements = get_active_requirements(state)
            
            if not candidate_matches:
                return {
                    "final_response": f"No qualified candidates found for {job_title}. You may want to:\n1. Lower the minimum score threshold\n2. Broaden the job requirements\n3. Add more resumes to the database",
                    "workflow_phase": "done"
                }

            reports = state.get("match_reports") or explain.build_match_reports(
                candidate_matches, requirements
            )

            # Build the report deterministically first. The LLM then adds a
            # narrative on top. If the model call fails the user still gets a
            # complete, accurate report rather than an error.
            sections: List[str] = []

            change = state.get("ranking_changes")
            if change:
                sections.append(explain.format_ranking_change(change))
                sections.append("")

            sections.append(
                f"Top {min(len(candidate_matches), 5)} of {len(candidate_matches)} "
                f"candidates for {job_title}:"
            )
            for report in reports[:5]:
                sections.append(explain.format_match_report(report))
                sections.append("")

            borderline = state.get("borderline_candidates") or explain.collect_borderline(reports)
            if borderline:
                sections.append(
                    f"Borderline candidates worth a decision: {', '.join(borderline)}. "
                    f"Targeted suggestions for each are listed above."
                )

            deterministic_report = "\n".join(sections).strip()

            narrative = self._llm_report_narrative(
                job_title, candidate_matches, reports, change
            )
            report_content = (
                f"{narrative}\n\n{deterministic_report}" if narrative else deterministic_report
            )
            
            updates = update_workflow_phase(state, "generate_report")
            updates.update({
                "final_response": report_content,
                "match_reports": reports,
                "borderline_candidates": borderline,
                "generated_reports": [{
                    "type": "match_report",
                    "timestamp": time.time(),
                    "content": report_content,
                    "candidates_included": len(candidate_matches)
                }],
                # Snapshot the ranking so the *next* refinement can diff
                # against it.
                "previous_ranking": list(candidate_matches),
            })
            
            logger.info(f"Generated match report for {len(candidate_matches)} candidates")
            return updates
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {
                "workflow_phase": "error",
                "error": str(e)
            }

    def _llm_report_narrative(
        self,
        job_title: str,
        candidates: Sequence[Dict[str, Any]],
        reports: Sequence[Dict[str, Any]],
        change: Optional[Dict[str, Any]]
    ) -> str:
        """Ask the LLM for an executive summary. Returns '' on any failure."""
        try:
            avg = sum(explain.candidate_score(c) for c in candidates) / max(1, len(candidates))
            lines = []
            for report in reports[:5]:
                lines.append(
                    f"{report['rank']}. {report['candidate_name']} "
                    f"({report['match_score']:.1f}%, {report['verdict']}) "
                    f"strengths={'; '.join(report['strengths'][:2])} "
                    f"gaps={'; '.join(report['gaps'][:2])}"
                )
            change_note = f"\nRanking change: {change.get('narrative')}" if change else ""

            prompt = (
                f"Position: {job_title}\n"
                f"Qualified candidates: {len(candidates)}\n"
                f"Average match score: {avg:.1f}%\n"
                f"Candidates:\n" + "\n".join(lines) + change_note + "\n\n"
                "Write a 3-5 sentence executive summary for the hiring manager. "
                "State who leads and why, flag any candidate whose ranking is "
                "not clear-cut, and recommend a concrete next step. Do not "
                "invent facts beyond what is listed."
            )
            response = self.llm.invoke([
                SystemMessage(content=(
                    "You are an expert technical recruiter writing a concise, "
                    "evidence-based candidate summary."
                )),
                HumanMessage(content=prompt)
            ])
            return (response.content or "").strip()
        except Exception as exc:
            logger.warning("Report narrative generation failed, using deterministic report: %s", exc)
            return ""
    
    def _human_feedback_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """
        Human feedback and approval checkpoint.

        With `interrupt_before=[HUMAN_FEEDBACK_NODE]` the graph halts *before*
        this node runs. The caller inspects the report, writes feedback into
        state with `update_state()`, then resumes with `invoke(None, config)`.
        Execution restarts here with the feedback already present, so this node
        only has to normalise it and let the router act on it.
        """
        try:
            raw_feedback = state.get("human_feedback")

            # No feedback supplied: nothing to approve, so finish cleanly.
            if not raw_feedback:
                return {
                    "workflow_phase": "done",
                    "awaiting_human": False,
                    "human_approved": None,
                }

            feedback_type = "approve"
            content = ""

            if isinstance(raw_feedback, dict):
                feedback_type = (raw_feedback.get("feedback_type") or "approve").lower()
                content = str(raw_feedback.get("content", ""))
            else:
                content = str(raw_feedback)
                feedback_type = self._classify_feedback(content)

            # Normalise free-text feedback that arrived inside a dict.
            if feedback_type not in {"approve", "refine", "reject", "clarify", "screen", "export"}:
                feedback_type = self._classify_feedback(content)

            feedback_data = HumanFeedbackData(
                feedback_type="approve" if feedback_type in {"screen", "export"} else feedback_type,
                content=content,
                timestamp=str(time.time()),
                approved=feedback_type in {"approve", "screen"},
            )
            
            updates = update_workflow_phase(state, "human_feedback")
            updates.update({
                "human_feedback": feedback_data,
                "human_approved": feedback_data["approved"],
                "awaiting_human": False,
                "feedback_history": [feedback_data],
                "screening_requested": feedback_type == "screen",
            })

            # A refine instruction carries the new criteria in its text.
            if feedback_type == "refine":
                updates["raw_user_query"] = content or state.get("raw_user_query", "")
            
            logger.info(f"Processed human feedback: {feedback_type}")
            return updates
            
        except Exception as e:
            logger.error(f"Human feedback processing failed: {e}")
            return {
                "workflow_phase": "error",
                "error": str(e)
            }

    @staticmethod
    def _classify_feedback(text: str) -> str:
        """Map free-text human feedback onto a feedback type."""
        lowered = (text or "").strip().lower()
        if not lowered:
            return "approve"
        if any(k in lowered for k in ("refine", "adjust", "change", "instead", "also need", "add ", "drop ", "remove ")):
            return "refine"
        if any(k in lowered for k in ("screen", "deep", "hire", "shortlist", "next round")):
            return "screen"
        if any(k in lowered for k in ("export", "save", "write")):
            return "export"
        if any(k in lowered for k in ("reject", "no thanks", "cancel", "stop")):
            return "reject"
        return "approve"
    
    # =========================================================================
    # Multi-round screening (Part C.1)
    # =========================================================================
    
    def _multi_round_screening_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """
        Round 1: initial screen narrowing the full pool to a top-N shortlist.

        Delegates to MultiRoundScreeningPipeline. The pipeline reads
        `candidate_matches`, which is append-only and can therefore hold
        duplicates from earlier turns, so it is handed a state view whose
        `candidate_matches` is the de-duplicated current ranking.
        """
        try:
            ranking = get_current_ranking(state)
            if not ranking:
                return {
                    "workflow_phase": "done",
                    "final_response": (
                        "There are no candidates to screen yet. Run a search first, "
                        "for example: 'Find candidates for a Senior Python Developer role'."
                    ),
                }

            screening_state = {**state, "candidate_matches": ranking}
            updates = self.screening_pipeline.execute_screening_round(screening_state, 1)

            if updates.get("error"):
                return {
                    "workflow_phase": "error",
                    "error": updates["error"],
                }

            selected = self._selected_from(updates, fallback=ranking)
            updates["ranked_candidates"] = selected
            updates["screening_requested"] = True
            # Do not re-append to the audit trail; round 1 selects from
            # candidates already recorded there.
            updates.pop("candidate_matches", None)

            logger.info("Screening round 1 selected %d candidates", len(selected))
            return updates

        except Exception as e:
            logger.error(f"Multi-round screening failed: {e}")
            return {"workflow_phase": "error", "error": str(e)}
    
    def _deep_analysis_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """Round 2: deep analysis of the shortlist, producing enhanced scores."""
        try:
            ranking = get_current_ranking(state)
            if not ranking:
                return {"workflow_phase": "done", "final_response": "No candidates available for deep analysis."}

            screening_state = {**state, "candidate_matches": ranking}
            updates = self.screening_pipeline.execute_screening_round(screening_state, 2)

            if updates.get("error"):
                return {"workflow_phase": "error", "error": updates["error"]}

            selected = self._selected_from(updates, fallback=ranking)
            updates["ranked_candidates"] = selected

            # Enhanced scores change the picture, so reports are rebuilt.
            requirements = get_active_requirements(state)
            reports = explain.build_match_reports(selected, requirements)
            updates["match_reports"] = reports
            updates["borderline_candidates"] = explain.collect_borderline(reports)
            updates.pop("candidate_matches", None)

            logger.info("Screening round 2 selected %d finalists", len(selected))
            return updates

        except Exception as e:
            logger.error(f"Deep analysis failed: {e}")
            return {"workflow_phase": "error", "error": str(e)}
    
    def _final_recommendation_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """Round 3: hire/no-hire recommendations for the finalists."""
        try:
            ranking = get_current_ranking(state)
            if not ranking:
                return {"workflow_phase": "done", "final_response": "No finalists available for recommendation."}

            screening_state = {**state, "candidate_matches": ranking}
            updates = self.screening_pipeline.execute_screening_round(screening_state, 3)

            if updates.get("error"):
                return {"workflow_phase": "error", "error": updates["error"]}

            selected = self._selected_from(updates, fallback=ranking)
            updates["ranked_candidates"] = selected
            updates.pop("candidate_matches", None)

            # Append the explainability layer the pipeline does not produce:
            # gaps and improvement suggestions for anyone still borderline.
            requirements = get_active_requirements(state)
            reports = explain.build_match_reports(selected, requirements)
            updates["match_reports"] = reports
            borderline = explain.collect_borderline(reports)
            updates["borderline_candidates"] = borderline

            base_response = updates.get("final_response") or ""
            addendum = self._screening_summary(state, reports, borderline)
            updates["final_response"] = f"{base_response}\n\n{addendum}".strip()
            updates["screening_complete"] = True
            updates["workflow_phase"] = "done"

            logger.info("Screening round 3 produced %d recommendations", len(selected))
            return updates

        except Exception as e:
            logger.error(f"Final recommendation failed: {e}")
            return {"workflow_phase": "error", "error": str(e)}

    @staticmethod
    def _selected_from(
        updates: Dict[str, Any],
        fallback: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Pull the selected candidates out of a screening round's updates."""
        data = updates.get("current_screening_data") or {}
        selected = data.get("candidates")
        return list(selected) if selected else list(fallback)

    def _screening_summary(
        self,
        state: MatchingAgentState,
        reports: Sequence[Dict[str, Any]],
        borderline: Sequence[str]
    ) -> str:
        """Build the funnel summary and borderline guidance for round 3."""
        lines: List[str] = []

        history = state.get("screening_history") or []
        if history:
            lines.append("Screening funnel:")
            for entry in history:
                lines.append(
                    f"  Round {entry.get('round_number')}: "
                    f"{len(entry.get('candidates') or [])} candidate(s) advanced "
                    f"(threshold {entry.get('threshold')}%)"
                )
            lines.append("")

        if borderline:
            lines.append("Borderline finalists and how to resolve them:")
            for report in reports:
                if report.get("candidate_name") not in borderline:
                    continue
                lines.append(f"  {report['candidate_name']} ({report['match_score']:.1f}%):")
                for suggestion in report.get("improvement_suggestions", [])[:3]:
                    lines.append(f"    - {suggestion}")
            lines.append("")

        lines.append("Gap analysis for each finalist:")
        for report in reports:
            gaps = report.get("gaps") or []
            lines.append(f"  {report['candidate_name']}: {gaps[0] if gaps else 'no material gaps'}")

        return "\n".join(lines)
    
    # =========================================================================
    # Ingestion and file operations
    # =========================================================================

    def _ingest_resumes_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """Ingest new resumes into the vector store."""
        try:
            from agent_tools import ingest_resume_directory

            query = state.get("raw_user_query", "") or ""
            match = re.search(r"([\w./\\-]*(?:resume|resumes)[\w./\\-]*)", query, re.IGNORECASE)
            directory = match.group(1) if match else None
            if not directory or not re.search(r"[/\\]", directory):
                directory = state.get("target_directory") or "examples/resumes"

            result = ingest_resume_directory.invoke({"directory": directory})

            updates = update_workflow_phase(state, "ingest")
            updates.update({
                "target_directory": directory,
                "file_operations": [{
                    "operation": "ingest",
                    "directory": directory,
                    "timestamp": time.time(),
                    "result": "error" if result.get("error") else "success",
                }],
                "tool_calls": [{
                    "tool": "ingest_resume_directory",
                    "timestamp": time.time(),
                    "directory": directory,
                }],
            })
            if result.get("error"):
                updates["error"] = result["error"]
            else:
                updates["performance_metrics"] = {
                    **(state.get("performance_metrics") or {}),
                    "ingested_count": result.get("processed_count", 0),
                }
            logger.info("Ingested resumes from %s", directory)
            return updates

        except Exception as e:
            logger.error(f"Resume ingestion failed: {e}")
            return {"workflow_phase": "error", "error": str(e)}
    
    def _index_check_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """Verify the vector index holds data before matching against it."""
        try:
            from agent_tools import get_rag_stats

            stats = get_rag_stats.invoke({})
            total = 0
            if isinstance(stats, dict):
                total = stats.get("total_chunks") or stats.get("count") or 0

            updates = update_workflow_phase(state, "index_check")
            updates["performance_metrics"] = {
                **(state.get("performance_metrics") or {}),
                "index_chunks": total,
            }
            if isinstance(stats, dict) and stats.get("error"):
                updates["error"] = stats["error"]
            elif not total:
                updates["error"] = (
                    "The resume index is empty. Ingest resumes before searching, e.g. "
                    "'ingest resumes from examples/resumes'."
                )
            logger.info("Index check: %s chunk(s)", total)
            return updates

        except Exception as e:
            logger.error(f"Index check failed: {e}")
            return {"workflow_phase": "error", "error": str(e)}

    def _file_agent_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """
        Handle file-system requests through LLM tool calling.

        This is the branch that needs the tool-bound model: the response may
        carry tool calls, which `tools_condition` then routes to the ToolNode.
        """
        try:
            messages = list(state.get("messages") or [])

            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=(
                    "You are a file system assistant for a recruitment tool. Use the "
                    "available tools to list, read, search and write files. Resume and "
                    "job-description files live under 'examples/'. Report tool results "
                    "plainly and do not fabricate file contents."
                ))] + messages

            # Guard against a tool-call loop that never converges.
            if state.get("iteration_count", 0) > state.get("max_iterations", 10):
                return {
                    "workflow_phase": "done",
                    "final_response": "Stopped after too many tool calls without reaching an answer.",
                }

            response = self.llm_with_tools.invoke(messages)

            updates = update_workflow_phase(state, "file_ops")
            updates["messages"] = [response]

            # Only treat this as terminal when the model stopped calling tools.
            if not getattr(response, "tool_calls", None):
                updates["final_response"] = response.content or "File operation completed."
                updates["workflow_phase"] = "done"
            return updates

        except Exception as e:
            logger.error(f"File agent failed: {e}")
            return {"workflow_phase": "error", "error": str(e)}
    
    # =========================================================================
    # Specialised analysis (Part B.1)
    # =========================================================================

    def _compare_candidates_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """Compare candidates head-to-head, side by side."""
        try:
            ranking = get_current_ranking(state)
            if not ranking:
                return {
                    "workflow_phase": "done",
                    "final_response": (
                        "There is no ranking to compare yet. Search for candidates first, "
                        "then ask me to compare them."
                    ),
                }

            query = state.get("raw_user_query", "") or ""
            selected = self._select_candidates_for_comparison(query, ranking)

            if len(selected) < 2:
                return {
                    "workflow_phase": "done",
                    "final_response": (
                        "I need at least two candidates to compare. Current ranking: "
                        + ", ".join(c.get("candidate_name", "?") for c in ranking[:10])
                    ),
                }

            requirements = get_active_requirements(state)
            table = explain.format_comparison_table(selected, requirements)

            # The dedicated tool adds a scored comparison matrix on top of the
            # deterministic table. It is capped at 5 candidates.
            tool_summary = ""
            try:
                from agent_tools import compare_candidates as compare_tool

                result = compare_tool.invoke({
                    "candidate_ids": [c["candidate_name"] for c in selected[:5]],
                    "job_requirements": requirements,
                })
                if isinstance(result, dict) and not result.get("error"):
                    recommendation = result.get("recommendation") or {}
                    if recommendation:
                        tool_summary = (
                            f"\nTool recommendation: {recommendation.get('top_candidate', 'n/a')}"
                            f" - {recommendation.get('recommendation_reason', '')}"
                        )
            except Exception as exc:
                logger.warning("compare_candidates tool failed: %s", exc)

            names = ", ".join(c.get("candidate_name", "?") for c in selected)
            response = f"Side-by-side comparison of {names}:\n\n{table}{tool_summary}"

            updates = update_workflow_phase(state, "compare")
            updates.update({
                "workflow_phase": "done",
                "final_response": response,
                "comparison_candidates": [c.get("candidate_name", "") for c in selected],
                "generated_explanations": [{
                    "type": "comparison",
                    "timestamp": time.time(),
                    "candidates": [c.get("candidate_name", "") for c in selected],
                }],
            })
            return updates

        except Exception as e:
            logger.error(f"Candidate comparison failed: {e}")
            return {"workflow_phase": "error", "error": str(e)}

    def _select_candidates_for_comparison(
        self,
        query: str,
        ranking: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Work out which candidates the user wants compared."""
        named = self._extract_candidate_names(query, ranking)
        if len(named) >= 2:
            return named[:5]

        # "compare the top 3" / "top three"
        match = re.search(r"top\s+(\d+)", query, re.IGNORECASE)
        if match:
            return ranking[: max(2, min(5, int(match.group(1))))]

        words = {"two": 2, "three": 3, "four": 4, "five": 5}
        for word, count in words.items():
            if re.search(rf"top\s+{word}\b", query, re.IGNORECASE):
                return ranking[:count]

        return ranking[:3]

    @staticmethod
    def _extract_candidate_names(
        query: str,
        ranking: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find candidates from the ranking that the query mentions by name."""
        lowered = (query or "").lower()
        found: List[Dict[str, Any]] = []
        for candidate in ranking:
            name = str(candidate.get("candidate_name", ""))
            if not name:
                continue
            if name.lower() in lowered:
                found.append(candidate)
                continue
            # Match on a distinctive single token (first or last name).
            for token in name.lower().split():
                if len(token) > 2 and re.search(rf"\b{re.escape(token)}\b", lowered):
                    found.append(candidate)
                    break
        # De-duplicate while preserving ranking order.
        seen = set()
        unique = []
        for candidate in found:
            key = candidate.get("candidate_name")
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique
    
    def _explain_ranking_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """Explain why candidates ranked as they did."""
        try:
            ranking = get_current_ranking(state)
            if not ranking:
                return {
                    "workflow_phase": "done",
                    "final_response": (
                        "There is no ranking to explain yet. Run a candidate search first."
                    ),
                }

            query = state.get("raw_user_query", "") or ""
            requirements = get_active_requirements(state)
            named = self._extract_candidate_names(query, ranking)

            if len(named) >= 2:
                # "Why did X rank higher than Y?"
                response = explain.explain_head_to_head(
                    ranking,
                    named[0].get("candidate_name", ""),
                    named[1].get("candidate_name", ""),
                    requirements,
                )
            elif len(named) == 1:
                reports = state.get("match_reports") or explain.build_match_reports(
                    ranking, requirements
                )
                target = named[0].get("candidate_name")
                report = next(
                    (r for r in reports if r.get("candidate_name") == target), None
                )
                response = (
                    explain.format_match_report(report) if report
                    else f"No report available for {target}."
                )
            else:
                # No names given: explain the ranking as a whole, and the last
                # change to it if there was one.
                reports = state.get("match_reports") or explain.build_match_reports(
                    ranking, requirements
                )
                parts = []
                change = state.get("ranking_changes")
                if change:
                    parts.append(explain.format_ranking_change(change))
                    parts.append("")
                parts.append(explain.format_match_reports(
                    reports[:5], state.get("job_title") or "the role"
                ))
                response = "\n".join(parts)

            updates = update_workflow_phase(state, "explain")
            updates.update({
                "workflow_phase": "done",
                "final_response": response,
                "explanation_request": query,
                "generated_explanations": [{
                    "type": "ranking_explanation",
                    "timestamp": time.time(),
                    "query": query,
                }],
            })
            return updates

        except Exception as e:
            logger.error(f"Ranking explanation failed: {e}")
            return {"workflow_phase": "error", "error": str(e)}
    
    def _generate_questions_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """Generate tailored interview questions for a candidate."""
        try:
            from agent_tools import generate_interview_questions

            ranking = get_current_ranking(state)
            query = state.get("raw_user_query", "") or ""
            requirements = get_active_requirements(state)

            named = self._extract_candidate_names(query, ranking)
            if named:
                candidate_name = named[0].get("candidate_name", "")
            elif ranking:
                candidate_name = ranking[0].get("candidate_name", "")
            else:
                # Fall back to a quoted or capitalised name in the query.
                match = re.search(r"for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", query)
                candidate_name = match.group(1) if match else ""

            if not candidate_name:
                return {
                    "workflow_phase": "done",
                    "final_response": (
                        "Tell me which candidate to generate questions for, or run a search "
                        "first so I can pick the top match."
                    ),
                }

            result = generate_interview_questions.invoke({
                "candidate_id": candidate_name,
                "job_requirements": requirements,
                "num_questions": 10,
            })

            if not isinstance(result, dict) or result.get("error"):
                error = result.get("error") if isinstance(result, dict) else "unknown error"
                return {
                    "workflow_phase": "done",
                    "final_response": f"Could not generate questions for {candidate_name}: {error}",
                }

            questions = result.get("questions") or []
            lines = [f"Interview questions for {candidate_name} ({len(questions)} total):", ""]
            by_type: Dict[str, List[Dict[str, Any]]] = {}
            for question in questions:
                by_type.setdefault(question.get("type", "general"), []).append(question)
            for qtype, group in by_type.items():
                lines.append(f"{qtype.title()}:")
                for index, question in enumerate(group, start=1):
                    lines.append(f"  {index}. {question.get('question', '')}")
                    focus = question.get("focus")
                    if focus:
                        lines.append(f"     (focus: {focus}, difficulty: {question.get('difficulty', 'n/a')})")
                lines.append("")

            updates = update_workflow_phase(state, "questions")
            updates.update({
                "workflow_phase": "done",
                "final_response": "\n".join(lines).strip(),
                "interview_questions": questions,
            })
            return updates

        except Exception as e:
            logger.error(f"Interview question generation failed: {e}")
            return {"workflow_phase": "error", "error": str(e)}
    
    # =========================================================================
    # Iterative refinement (Part B.2)
    # =========================================================================

    def _adjust_criteria_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """
        Apply a mid-conversation criteria change.

        Snapshots the current ranking, folds the requested adjustments into the
        active requirements, and records the change. The downstream re-search
        and re-rank then produce a new ranking, and `_rank_candidates_node`
        diffs it against the snapshot so the change can be explained.
        """
        try:
            query = state.get("raw_user_query", "") or ""
            current = get_current_ranking(state)

            requirements = dict(get_active_requirements(state) or {})
            criteria_before = self._criteria_snapshot(state, requirements)

            adjustments = self._parse_refinement_request(query)

            # Skills
            required = list(requirements.get("skills_required") or [])
            for skill in adjustments.get("add_required_skills") or []:
                if skill.lower() not in {s.lower() for s in required}:
                    required.append(skill)
            preferred = list(requirements.get("skills_preferred") or [])
            for skill in adjustments.get("add_preferred_skills") or []:
                if skill.lower() not in {s.lower() for s in preferred}:
                    preferred.append(skill)

            drop = {s.lower() for s in (adjustments.get("remove_skills") or [])}
            if drop:
                required = [s for s in required if s.lower() not in drop]
                preferred = [s for s in preferred if s.lower() not in drop]

            requirements["skills_required"] = required
            requirements["skills_preferred"] = preferred

            if adjustments.get("min_experience_years") is not None:
                requirements["experience_years"] = adjustments["min_experience_years"]

            updates = update_workflow_phase(state, "adjust_criteria")

            new_top_k = adjustments.get("top_k") or state.get("top_k", 10)
            new_min_score = state.get("min_score", 0.3)
            if adjustments.get("min_score") is not None:
                # Accept either a fraction or a percentage from the user.
                raw = float(adjustments["min_score"])
                new_min_score = raw / 100.0 if raw > 1 else raw

            criteria_after = {
                "skills_required": required,
                "skills_preferred": preferred,
                "experience_years": requirements.get("experience_years"),
                "top_k": new_top_k,
                "min_score": new_min_score,
            }

            refinement = explain.build_refinement_record(
                turn=state.get("refinement_count", 0) + 1,
                user_request=query,
                parsed_adjustments=adjustments,
                criteria_before=criteria_before,
                criteria_after=criteria_after,
            )

            updates.update({
                "active_requirements": requirements,
                "top_k": new_top_k,
                "min_score": new_min_score,
                # Snapshot for the delta computed after re-ranking.
                "previous_ranking": list(current),
                "pending_refinement": refinement,
                "refinement_history": [refinement],
                "refinement_count": state.get("refinement_count", 0) + 1,
                # A refinement invalidates any completed screening.
                "screening_round": 0,
                "screening_complete": False,
            })

            logger.info("Applied refinement: %s", adjustments)
            return updates

        except Exception as e:
            logger.error(f"Criteria adjustment failed: {e}")
            return {"workflow_phase": "error", "error": str(e)}

    @staticmethod
    def _criteria_snapshot(
        state: MatchingAgentState,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Capture the criteria currently in force, for diffing."""
        return {
            "skills_required": list(requirements.get("skills_required") or []),
            "skills_preferred": list(requirements.get("skills_preferred") or []),
            "experience_years": requirements.get("experience_years"),
            "top_k": state.get("top_k", 10),
            "min_score": state.get("min_score", 0.3),
        }

    def _parse_refinement_request(self, query: str) -> Dict[str, Any]:
        """
        Turn a natural-language refinement into structured adjustments.

        Tries the LLM first because refinement phrasing varies widely, then
        falls back to regex and a skill vocabulary so the feature still works
        without a model call.
        """
        adjustments = self._regex_refinement(query)

        try:
            prompt = (
                "Extract search criteria changes from the recruiter's request.\n"
                "Respond with ONLY a JSON object using these keys (omit what does "
                "not apply):\n"
                '{"add_required_skills": [], "add_preferred_skills": [], '
                '"remove_skills": [], "min_experience_years": null, '
                '"min_score": null, "top_k": null}\n\n'
                f"Request: {query}"
            )
            response = self.llm.invoke([
                SystemMessage(content="You extract structured search filters. Output JSON only."),
                HumanMessage(content=prompt)
            ])
            text = (response.content or "").strip()
            # Strip markdown fences if the model added them.
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                # LLM output supplements the regex pass; regex wins on numbers
                # it positively identified, since it is the more literal reader.
                for key in ("add_required_skills", "add_preferred_skills", "remove_skills"):
                    values = parsed.get(key) or []
                    if isinstance(values, list):
                        merged = list(adjustments.get(key) or [])
                        for value in values:
                            if isinstance(value, str) and value.lower() not in {m.lower() for m in merged}:
                                merged.append(value)
                        adjustments[key] = merged
                for key in ("min_experience_years", "min_score", "top_k"):
                    if adjustments.get(key) is None and isinstance(parsed.get(key), (int, float)):
                        adjustments[key] = parsed[key]
        except Exception as exc:
            logger.info("LLM refinement parsing unavailable, using regex result: %s", exc)

        return adjustments

    @staticmethod
    def _regex_refinement(query: str) -> Dict[str, Any]:
        """Deterministic refinement parsing."""
        lowered = (query or "").lower()
        adjustments: Dict[str, Any] = {
            "add_required_skills": [],
            "add_preferred_skills": [],
            "remove_skills": [],
            "min_experience_years": None,
            "min_score": None,
            "top_k": None,
        }

        # Experience floor: "3+ years", "at least 5 years"
        match = re.search(r"(\d+)\s*\+?\s*years?", lowered)
        if match:
            adjustments["min_experience_years"] = int(match.group(1))

        # Score threshold: "above 80%", "minimum score 75"
        match = re.search(r"(?:above|over|min(?:imum)?(?:\s+score)?|at least|score of)\s*(\d{2,3})\s*%?", lowered)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 100:
                adjustments["min_score"] = value

        # Result count: "top 5", "show me 8"
        match = re.search(r"top\s+(\d+)", lowered)
        if match:
            adjustments["top_k"] = int(match.group(1))

        # Skills, split by whether the phrasing adds or removes.
        removal_zone = bool(re.search(r"\b(?:drop|remove|without|no longer|forget|ignore)\b", lowered))
        original = query or ""
        for skill in _SKILL_VOCABULARY:
            match = re.search(rf"(?<![\w+#.]){re.escape(skill)}(?![\w+#])", lowered)
            if not match:
                continue

            position = match.start()
            # Carry the user's own capitalisation through ("React", not
            # "react") so the requirements list stays readable and consistent
            # with skills extracted from the job description.
            surface = original[position:position + len(skill)] or skill

            # Decide add vs remove by which verb is nearer the skill.
            window = lowered[max(0, position - 40):position]
            if re.search(r"\b(?:drop|remove|without|no longer|forget|ignore)\b", window):
                adjustments["remove_skills"].append(surface)
            elif re.search(r"\b(?:prefer|nice to have|bonus|ideally)\b", window):
                adjustments["add_preferred_skills"].append(surface)
            elif removal_zone and not re.search(r"\b(?:also|add|need|require|must)\b", window):
                adjustments["remove_skills"].append(surface)
            else:
                adjustments["add_required_skills"].append(surface)

        return adjustments
    
    def _handle_error_node(self, state: MatchingAgentState) -> Dict[str, Any]:
        """Handle errors and provide recovery options."""
        error_msg = state.get("error", "Unknown error occurred")
        return {
            "workflow_phase": "error",
            "final_response": f"I encountered an error: {error_msg}\n\nPlease try rephrasing your request or contact support if the problem persists."
        }
    
    # =========================================================================
    # Routing Functions
    # =========================================================================
    
    def _route_by_intent(self, state: MatchingAgentState) -> str:
        """Route workflow based on classified intent."""
        intent = state.get("intent", "clarify")

        if intent == "error" or state.get("error"):
            return "error"

        # Loop guard. `iteration_count` is reset at the start of every turn, so
        # tripping this means the current turn is cycling.
        if state.get("iteration_count", 0) >= state.get("max_iterations", 10):
            logger.warning("Iteration ceiling reached; routing to error handler")
            return "error"

        # Screening needs an existing ranking. Without one, fall back to
        # running a search first rather than screening an empty pool.
        if intent == "screen" and not get_current_ranking(state):
            return "job_match" if state.get("job_description") else "clarify"

        if intent not in _VALID_INTENTS:
            return "clarify"

        return intent

    def _route_after_parse_job(self, state: MatchingAgentState) -> str:
        """Continue only when a job description was actually resolved."""
        if state.get("workflow_phase") == "clarify":
            return "clarify"
        if state.get("error"):
            return "error"
        if not state.get("job_description"):
            return "clarify"
        return "continue"

    def _route_after_search(self, state: MatchingAgentState) -> str:
        """Stop on search failure instead of ranking nothing."""
        if state.get("error") or state.get("workflow_phase") == "error":
            return "error"
        return "continue"
    
    def _route_after_report(self, state: MatchingAgentState) -> str:
        """Route after report generation."""
        if state.get("error"):
            return "end"

        # An explicit screening request skips the approval checkpoint.
        if state.get("screening_requested"):
            return "screening"

        # No candidates means nothing to approve.
        if not get_current_ranking(state):
            return "end"

        if self.enable_hitl:
            return "human_feedback"

        # Without HITL the node still runs, keeping the graph shape intact,
        # but it will auto-complete rather than wait.
        return "human_feedback"
    
    def _route_after_feedback(self, state: MatchingAgentState) -> str:
        """Route based on human feedback."""
        feedback = state.get("human_feedback")
        if not feedback:
            return "end"

        feedback_type = feedback.get("feedback_type", "approve")

        if feedback_type == "refine":
            return "refine"
        if feedback_type == "approve" and state.get("screening_requested"):
            return "approve"
        return "end"
    
    def _route_after_analysis(self, state: MatchingAgentState) -> str:
        """Route after deep analysis."""
        if state.get("error"):
            return "end"
        if not get_current_ranking(state):
            return "end"
        # Rounds 1 and 2 are complete by this point, so round 3 follows.
        return "final"
    
    def _route_after_index(self, state: MatchingAgentState) -> str:
        """Route after index check."""
        if state.get("error"):
            return "error"
        return "ready"
    
    def _route_after_tools(self, state: MatchingAgentState) -> str:
        """Route after tool execution."""
        return "continue"
