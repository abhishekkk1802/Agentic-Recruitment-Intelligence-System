"""
LangGraph Matching Agent - Main Implementation
=============================================

The main conversational AI agent using LangGraph for intelligent job-resume matching.
Integrates all components including state management, workflow orchestration, and
human-in-the-loop capabilities for multi-round screening processes.

This module provides:
- Complete LangGraph agent implementation  
- Conversational interface with natural language processing
- Multi-round screening pipeline with progressive filtering
- Human feedback integration and explainable AI results
"""

import os
import logging
import sqlite3
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
import json

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

# NOTE ON LANGGRAPH VERSION: this project pins langgraph 0.2.45, which predates
# the `Command` primitive and the `interrupt()` function. Human-in-the-loop is
# therefore driven by the 0.2.x mechanism instead:
#   1. the graph is compiled with interrupt_before=[HUMAN_FEEDBACK_NODE]
#   2. the caller writes feedback in with graph.update_state(...)
#   3. execution resumes with graph.invoke(None, config)
# An earlier revision called `Command(resume=...)` without importing it, which
# raised NameError and left the resume path dead.

# Import our components
from agent_state import (
    MatchingAgentState, 
    create_initial_state,
    get_current_ranking,
    AgentInput,
    AgentOutput
)

class ResponseWrapper:
    """Wrapper to provide both dict and attribute access to response data."""

    # Attributes that callers may read on any response, including the reduced
    # payloads returned on error paths. Without these defaults, something like
    # `response.awaiting_feedback` would raise AttributeError on an error
    # response and mask the original failure.
    _DEFAULTS = {
        "messages": [],
        "final_response": None,
        "candidate_matches": [],
        "recommendations": None,
        "error": None,
        "thread_id": None,
        "awaiting_feedback": False,
        "workflow_phase": None,
        "match_reports": [],
        "borderline_candidates": [],
        "ranking_changes": None,
        "refinement_count": 0,
        "screening_complete": False,
        "screening_history": [],
        "intent": None,
    }

    def __init__(self, data: dict):
        self._data = data
        for key, value in data.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        """Supply documented defaults for keys a given response omitted."""
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._DEFAULTS:
            default = self._DEFAULTS[name]
            return list(default) if isinstance(default, list) else default
        raise AttributeError(
            f"{type(self).__name__!r} has no attribute {name!r}"
        )
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __contains__(self, key):
        return key in self._data
    
    def get(self, key, default=None):
        return self._data.get(key, default)
    
    def keys(self):
        return self._data.keys()
    
    def items(self):
        return self._data.items()
    
    def values(self):
        return self._data.values()
from agent_workflow import MatchingAgentWorkflow, HUMAN_FEEDBACK_NODE
from simple_workflow import SimpleMatchingWorkflow  # retained: legacy fallback
from agent_tools import initialize_tools, get_tools_by_category
from resume_rag import ResumeRAG
from job_matcher import JobMatcher

# MCP client is optional — the agent works with direct tools by default.
# When config["use_mcp"] is True, file operations are routed through the
# filesystem MCP server instead, demonstrating the Model Context Protocol.
try:
    from mcp_client import MCPClient
    _MCP_AVAILABLE = True
except ImportError:
    MCPClient = None  # type: ignore[misc,assignment]
    _MCP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangGraphMatchingAgent:
    """
    Main LangGraph-based conversational job matching agent.
    
    Orchestrates the complete matching process from intent classification through
    multi-round screening with advanced explainability and human collaboration.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_db: Optional[str] = None
    ):
        """
        Initialize the LangGraph matching agent.
        
        Args:
            config: Configuration parameters
            checkpoint_db: Path to checkpoint database (None for in-memory)
        """
        self.config = config or {}
        self.mcp_client = None  # set by _initialize_mcp when use_mcp is True
        
        # Initialize core components
        self._initialize_rag_system()
        self._initialize_llm()
        self._initialize_checkpointer(checkpoint_db)
        if self.config.get("use_mcp"):
            self._initialize_mcp()
        self._initialize_workflow()
        
        logger.info("LangGraphMatchingAgent initialized successfully (mcp=%s)",
                    self.config.get("use_mcp", False))
    
    def _initialize_rag_system(self) -> None:
        """Initialize RAG and JobMatcher instances."""
        try:
            # Initialize RAG system
            self.rag = ResumeRAG(
                vector_db="chromadb",
                embedding_model=self.config.get("embedding_model", "openai"),
                collection_name=self.config.get("collection_name", "matching_agent_collection")
            )
            
            # Initialize job matcher
            self.matcher = JobMatcher(
                rag_system=self.rag,
                weight_config=self.config.get("scoring_weights")
            )
            
            # Initialize agent tools with RAG instances
            initialize_tools(self.rag, self.matcher)
            
            logger.info("RAG system and JobMatcher initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def _initialize_llm(self) -> None:
        """Initialize the language model."""
        try:
            self.llm = ChatOpenAI(
                model=self.config.get("model", "gpt-4o-mini"),
                temperature=self.config.get("temperature", 0.1),
                max_tokens=self.config.get("max_tokens", 4000)
            )
            logger.info(f"LLM initialized: {self.config.get('model', 'gpt-4o-mini')}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _initialize_checkpointer(self, checkpoint_db: Optional[str]) -> None:
        """
        Initialize the state checkpointer.

        The checkpointer is what makes multi-turn conversation possible: it
        persists state between turns so a follow-up like "also require React"
        has an existing ranking to refine.

        Note that `SqliteSaver.from_conn_string()` is a *context manager* in
        langgraph-checkpoint-sqlite 2.x, so it cannot be assigned directly. A
        connection is opened explicitly instead, with `check_same_thread=False`
        because LangGraph may touch it from a worker thread.
        """
        try:
            if checkpoint_db:
                connection = sqlite3.connect(checkpoint_db, check_same_thread=False)
                self.checkpointer = SqliteSaver(connection)
                self._checkpoint_connection = connection
                logger.info("SQLite checkpointer initialized at %s", checkpoint_db)
            else:
                self.checkpointer = MemorySaver()
                self._checkpoint_connection = None
                logger.info("In-memory checkpointer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize checkpointer: {e}")
            raise
    
    def _initialize_mcp(self) -> None:
        """
        Start the filesystem MCP server and connect the client.

        When MCP mode is enabled, file-system operations (read, list, write,
        search, watch, batch) are mediated by the MCP server process rather
        than called directly. The workflow receives the MCP-backed tools and
        uses them for the file_tools node, so the graph structure is unchanged.
        """
        if not _MCP_AVAILABLE:
            logger.warning(
                "MCP mode requested but mcp_client is not available. "
                "Falling back to direct tools."
            )
            self.config["use_mcp"] = False
            return

        try:
            self.mcp_client = MCPClient()
            self.mcp_client.start()
            logger.info("MCP client connected, file tools now MCP-mediated")

            discovered = self.mcp_client.list_tools()
            logger.info("MCP server exposes %d tools: %s",
                        len(discovered),
                        ", ".join(t["name"] for t in discovered))
        except Exception as e:
            logger.error("Failed to initialize MCP client: %s", e)
            logger.warning("Falling back to direct tools")
            self.mcp_client = None
            self.config["use_mcp"] = False

    def _initialize_workflow(self) -> None:
        """
        Initialize the LangGraph workflow.

        Two workflows exist in this project:

        - ``full`` (default): ``MatchingAgentWorkflow``, the complete graph with
          intent routing, requirement extraction, the human feedback loop and
          multi-round screening. This is the Milestone 3 deliverable.
        - ``simple``: ``SimpleMatchingWorkflow``, an earlier two-node
          agent/tools loop. Kept selectable via ``config["workflow"] =
          "simple"`` so earlier demos remain runnable.
        """
        try:
            workflow_kind = self.config.get("workflow", "full")

            if workflow_kind == "simple":
                workflow = SimpleMatchingWorkflow(
                    llm=self.llm,
                    checkpointer=self.checkpointer
                )
                self.workflow = workflow
                self.graph = workflow.graph
                self.enable_hitl = False
                logger.info("Workflow graph initialized (legacy simple workflow)")
                return

            workflow = MatchingAgentWorkflow(
                llm=self.llm,
                checkpointer=self.checkpointer,
                config=self.config,
                mcp_client=self.mcp_client,
            )
            self.workflow = workflow
            self.graph = workflow.graph
            self.enable_hitl = workflow.enable_hitl
            
            logger.info(
                "Workflow graph initialized (full matching workflow, hitl=%s)",
                self.enable_hitl
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize workflow: {e}")
            raise
    
    # =========================================================================
    # Main Interface Methods
    # =========================================================================
    
    def process_query(
        self, 
        query: str, 
        thread_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        """
        Process a single user query through the agent workflow.
        
        Args:
            query: User query string
            thread_id: Thread ID for conversation tracking
            config: Additional configuration parameters
            
        Returns:
            AgentOutput with results and recommendations
        """
        try:
            # Generate thread ID if not provided
            if thread_id is None:
                thread_id = f"thread_{int(time.time())}_{hash(query) % 10000}"

            # The thread_id must travel in `configurable` or the checkpointer
            # has no key to store against, and every turn starts from scratch.
            agent_config: Dict[str, Any] = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": self.config.get("recursion_limit", 50),
            }
            if config:
                agent_config["configurable"].update(config)

            existing = self._get_state_snapshot(thread_id)

            if existing is not None and existing.next:
                # This thread is parked at the feedback checkpoint. The message
                # might be the awaited feedback ("approve", "also add React"),
                # or it might be an unrelated new request ("compare the top 3").
                # Answering a new request as though it were feedback would
                # silently drop it, so the two cases are told apart here.
                if self._looks_like_feedback(query):
                    logger.info(
                        "Thread %s awaiting feedback at %s; routing query as feedback",
                        thread_id, existing.next
                    )
                    return self.resume_conversation(thread_id, query, config=config)

                logger.info(
                    "Thread %s awaiting feedback, but the message reads as a new "
                    "request; starting a fresh turn", thread_id
                )

            if existing is not None and existing.values:
                # Continuing an established conversation: append the new turn
                # so prior context (job description, ranking) is preserved.
                initial_state: Any = {"messages": [HumanMessage(content=query)]}
            else:
                # First turn on this thread: build the full default state.
                initial_state = create_initial_state(
                    messages=[HumanMessage(content=query)],
                    thread_id=thread_id,
                    config=self.config
                )
            
            # Run the workflow
            start_time = time.time()
            final_state = self.graph.invoke(initial_state, config=agent_config)
            processing_time = time.time() - start_time
            
            # Extract results
            output = self._extract_output(final_state, processing_time, thread_id)
            
            logger.info(f"Query processed in {processing_time:.3f}s, thread: {thread_id}")
            return output
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            result_data = {
                "messages": [AIMessage(content=f"I encountered an error: {str(e)}")],
                "final_response": f"Error: {str(e)}",
                "candidate_matches": [],
                "recommendations": None,
                "error": str(e)
            }
            return ResponseWrapper(result_data)
    
    def stream_query(
        self,
        query: str,
        thread_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream query processing with real-time updates.
        
        Args:
            query: User query string
            thread_id: Thread ID for conversation tracking
            config: Additional configuration parameters
            
        Yields:
            Streaming updates from the workflow execution
        """
        try:
            # Generate thread ID if not provided
            if thread_id is None:
                thread_id = f"stream_{int(time.time())}_{hash(query) % 10000}"
            
            # Create input state
            input_messages = [HumanMessage(content=query)]
            agent_config = {"configurable": {"thread_id": thread_id}}
            
            # Merge additional config
            if config:
                agent_config["configurable"].update(config)
            
            # Create initial state
            initial_state = create_initial_state(
                messages=input_messages,
                thread_id=thread_id,
                config=self.config
            )
            
            # Stream the workflow
            for event in self.graph.stream(initial_state, config=agent_config, stream_mode="updates"):
                yield {
                    "type": "workflow_update",
                    "data": event,
                    "timestamp": time.time(),
                    "thread_id": thread_id
                }
            
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield {
                "type": "error",
                "data": {"error": str(e)},
                "timestamp": time.time(),
                "thread_id": thread_id
            }
    
    def resume_conversation(
        self,
        thread_id: str,
        human_input: Any,
        config: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        """
        Resume a conversation after human feedback or interruption.
        
        Args:
            thread_id: Thread ID of the conversation to resume
            human_input: Human feedback or input data
            config: Additional configuration parameters
            
        Returns:
            AgentOutput with continued results
        """
        try:
            agent_config: Dict[str, Any] = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": self.config.get("recursion_limit", 50),
            }
            
            # Merge additional config
            if config:
                agent_config["configurable"].update(config)

            snapshot = self._get_state_snapshot(thread_id)
            if snapshot is None:
                return ResponseWrapper({
                    "messages": [],
                    "final_response": (
                        f"No conversation found for thread '{thread_id}'. "
                        f"Start one by asking me to find candidates."
                    ),
                    "candidate_matches": [],
                    "recommendations": None,
                    "error": "unknown_thread",
                })

            if not snapshot.next:
                # Nothing is paused, so this is an ordinary follow-up turn.
                logger.info("Thread %s is not paused; treating input as a new query", thread_id)
                return self.process_query(str(human_input), thread_id=thread_id, config=config)

            # Write the feedback into state, then resume. Passing None as the
            # input tells LangGraph to continue from the checkpoint rather than
            # begin a new run.
            self.graph.update_state(agent_config, {"human_feedback": human_input})

            start_time = time.time()
            final_state = self.graph.invoke(None, config=agent_config)
            processing_time = time.time() - start_time
            
            # Extract results
            output = self._extract_output(final_state, processing_time, thread_id)
            
            logger.info(f"Conversation resumed in {processing_time:.3f}s, thread: {thread_id}")
            return output
            
        except Exception as e:
            logger.error(f"Conversation resume failed: {e}")
            result_data = {
                "messages": [AIMessage(content=f"Resume error: {str(e)}")],
                "final_response": f"Error resuming conversation: {str(e)}",
                "candidate_matches": [],
                "recommendations": None,
                "error": str(e)
            }
            return ResponseWrapper(result_data)
    
    # Phrases that unambiguously answer the feedback prompt rather than
    # starting a new line of enquiry.
    _APPROVE_WORDS = (
        "approve", "approved", "looks good", "sounds good", "lgtm", "accept",
        "proceed", "go ahead", "yes", "ok", "okay", "fine", "perfect",
    )
    _EXPORT_WORDS = ("export", "save", "write to file", "download")
    _REJECT_WORDS = ("reject", "cancel", "stop", "no thanks", "never mind", "discard")

    def _looks_like_feedback(self, query: str) -> bool:
        """
        Decide whether a message answers the pending feedback prompt.

        Returns True for approvals, refinements, screening requests, exports and
        rejections. Returns False for anything that reads as a fresh request
        (comparisons, explanations, interview questions, a new job description),
        so those get a new turn instead of being consumed as feedback.
        """
        text = (query or "").strip().lower()
        if not text:
            return True  # empty input at a prompt means "just proceed"

        # Reuse the workflow's intent classifier where available: it already
        # knows which phrasings are refinements or screening requests.
        classifier = getattr(self.workflow, "_pattern_classify", None)
        if callable(classifier):
            intent = classifier(text)
            if intent in {"refine", "screen"}:
                return True
            if intent in {"compare", "explain", "questions", "hybrid", "file_ops"}:
                return False

        if any(word in text for word in self._APPROVE_WORDS):
            return True
        if any(word in text for word in self._EXPORT_WORDS):
            return True
        if any(word in text for word in self._REJECT_WORDS):
            return True

        # A long message is almost certainly a new job description.
        if len(text) > 100:
            return False

        # Short and unclassifiable: treat as a new request rather than
        # guessing at approval, since a wrong approval loses the message.
        return False

    def _get_state_snapshot(self, thread_id: str):
        """
        Fetch the raw LangGraph state snapshot for a thread.

        The snapshot carries `.next`, which is non-empty when the graph is
        parked at an interrupt. That is how the agent tells "awaiting human
        feedback" apart from "finished".
        """
        try:
            snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
            if snapshot is None or not snapshot.values:
                return None
            return snapshot
        except Exception as exc:
            logger.debug("No state snapshot for thread %s: %s", thread_id, exc)
            return None

    def get_conversation_state(self, thread_id: str) -> Optional[MatchingAgentState]:
        """
        Get the current state of a conversation.
        
        Args:
            thread_id: Thread ID to get state for
            
        Returns:
            Current MatchingAgentState or None if not found
        """
        try:
            snapshot = self._get_state_snapshot(thread_id)
            return snapshot.values if snapshot else None
            
        except Exception as e:
            logger.error(f"Failed to get conversation state: {e}")
            return None

    def is_awaiting_feedback(self, thread_id: str) -> bool:
        """
        Whether the graph is paused waiting on human feedback.

        Args:
            thread_id: Thread to inspect

        Returns:
            True when execution is parked at the feedback checkpoint
        """
        snapshot = self._get_state_snapshot(thread_id)
        if snapshot is None or not snapshot.next:
            return False
        return HUMAN_FEEDBACK_NODE in snapshot.next

    def get_pending_feedback_request(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Describe what the agent is waiting for, for display to the user.

        Args:
            thread_id: Thread to inspect

        Returns:
            Details of the pending decision, or None if nothing is pending
        """
        if not self.is_awaiting_feedback(thread_id):
            return None

        state = self.get_conversation_state(thread_id) or {}
        ranking = get_current_ranking(state)
        return {
            "message": "Review the candidate matches and choose how to proceed.",
            "options": [
                "approve - accept these results",
                "refine <new criteria> - adjust and re-rank",
                "screen - run multi-round screening to a hire/no-hire call",
                "export - save the results to a file",
            ],
            "current_matches": len(ranking),
            "top_candidate": ranking[0].get("candidate_name") if ranking else None,
            "borderline_candidates": state.get("borderline_candidates", []),
        }

    def provide_feedback(
        self,
        thread_id: str,
        feedback: Any,
        config: Optional[Dict[str, Any]] = None
    ) -> AgentOutput:
        """
        Supply human feedback and continue the workflow.

        Thin alias over `resume_conversation` that reads more naturally at call
        sites in the chat loop and test scenarios.

        Args:
            thread_id: Thread awaiting feedback
            feedback: Either free text ("also require React") or a dict such as
                {"feedback_type": "screen"}
            config: Additional configuration

        Returns:
            AgentOutput after the workflow continues
        """
        return self.resume_conversation(thread_id, feedback, config=config)
    
    def list_active_conversations(self) -> List[Dict[str, Any]]:
        """
        List all active conversation threads.
        
        Returns:
            List of conversation thread information
        """
        try:
            conversations: Dict[str, Dict[str, Any]] = {}

            # The checkpointer is the only record of which threads exist. Its
            # listing is ordered newest-first per thread, so the first
            # checkpoint seen for a thread is its current one.
            for checkpoint in self.checkpointer.list(None):
                configurable = (checkpoint.config or {}).get("configurable", {})
                thread_id = configurable.get("thread_id")
                if not thread_id or thread_id in conversations:
                    continue

                values = checkpoint.checkpoint.get("channel_values", {}) or {}
                ranking = values.get("ranked_candidates") or values.get("candidate_matches") or []
                conversations[thread_id] = {
                    "thread_id": thread_id,
                    "workflow_phase": values.get("workflow_phase"),
                    "job_title": values.get("job_title"),
                    "message_count": len(values.get("messages") or []),
                    "candidate_count": len(ranking),
                    "refinement_count": values.get("refinement_count", 0),
                    "screening_complete": values.get("screening_complete", False),
                    "awaiting_feedback": self.is_awaiting_feedback(thread_id),
                }

            return list(conversations.values())
            
        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            return []
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _extract_output(
        self,
        final_state: MatchingAgentState,
        processing_time: float,
        thread_id: Optional[str] = None
    ) -> AgentOutput:
        """Extract AgentOutput from final workflow state."""
        try:
            # Extract messages
            messages = final_state.get("messages", [])
            
            # Extract final response
            final_response = final_state.get("final_response")
            if not final_response and messages:
                # Use last AI message as response
                for msg in reversed(messages):
                    if hasattr(msg, 'type') and msg.type == "ai":
                        final_response = msg.content
                        break
            
            # Use the authoritative ranking rather than the append-only audit
            # trail, which can hold duplicates from earlier refinements.
            candidate_matches = get_current_ranking(final_state)
            
            # Extract recommendations
            recommendations = final_state.get("final_recommendations")
            
            # Add performance metadata
            if not recommendations:
                recommendations = [{
                    "type": "performance_metrics",
                    "processing_time": processing_time,
                    "workflow_phase": final_state.get("workflow_phase"),
                    "candidates_found": len(candidate_matches),
                    "tools_used": len(final_state.get("tool_calls", []))
                }]

            awaiting = self.is_awaiting_feedback(thread_id) if thread_id else False
            
            result_data = {
                "messages": messages,
                "final_response": final_response or "Processing completed.",
                "candidate_matches": candidate_matches,
                "recommendations": recommendations,
                "error": final_state.get("error"),
                # Milestone 3 additions surfaced to callers and the chat UI.
                "thread_id": thread_id,
                "awaiting_feedback": awaiting,
                "workflow_phase": final_state.get("workflow_phase"),
                "match_reports": final_state.get("match_reports", []),
                "borderline_candidates": final_state.get("borderline_candidates", []),
                "ranking_changes": final_state.get("ranking_changes"),
                "refinement_count": final_state.get("refinement_count", 0),
                "screening_complete": final_state.get("screening_complete", False),
                "screening_history": final_state.get("screening_history", []),
                "intent": final_state.get("intent"),
            }
            return ResponseWrapper(result_data)
            
        except Exception as e:
            logger.error(f"Output extraction failed: {e}")
            result_data = {
                "messages": [AIMessage(content="Output extraction error")],
                "final_response": f"Error extracting results: {str(e)}",
                "candidate_matches": [],
                "recommendations": None,
                "error": str(e)
            }
            return ResponseWrapper(result_data)
    
    def initialize_with_resumes(self, resume_directory: str) -> Dict[str, Any]:
        """
        Initialize the agent with resumes from a directory.
        
        Args:
            resume_directory: Path to directory containing resume files
            
        Returns:
            Initialization results and statistics
        """
        try:
            start_time = time.time()
            
            # Process resumes using RAG system
            processed_resumes = self.rag.process_resume_directory(resume_directory)
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "directory": resume_directory,
                "processed_count": len(processed_resumes),
                "processing_time": processing_time,
                "avg_time_per_resume": processing_time / len(processed_resumes) if processed_resumes else 0,
                "resumes": [
                    {
                        "name": resume.name,
                        "skills_count": len(resume.skills),
                        "experience_years": resume.experience_years,
                        "file_path": resume.file_path
                    }
                    for resume in processed_resumes
                ]
            }
            
            logger.info(f"Initialized with {len(processed_resumes)} resumes in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Resume initialization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "directory": resume_directory,
                "processed_count": 0
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics and health metrics.
        
        Returns:
            System statistics and performance metrics
        """
        try:
            # Get RAG stats
            rag_stats = self.rag.get_collection_stats()
            
            # Add agent-specific stats
            agent_stats = {
                "agent_version": "langgraph_matching_agent_v1.0",
                "initialization_time": datetime.now().isoformat(),
                "config": {
                    "model": self.config.get("model", "gpt-4o-mini"),
                    "embedding_model": self.config.get("embedding_model", "openai"),
                    "collection_name": self.config.get("collection_name", "matching_agent_collection"),
                    "use_mcp": self.config.get("use_mcp", False),
                },
                "capabilities": [
                    "job_matching",
                    "candidate_comparison", 
                    "interview_questions",
                    "multi_round_screening",
                    "human_feedback_loops",
                    "explainable_rankings",
                    "mcp_file_operations" if self.mcp_client else "direct_file_operations",
                ]
            }
            
            # Add MCP stats if available
            if self.mcp_client:
                mcp_tools = self.mcp_client.list_tools()
                agent_stats["mcp_server"] = {
                    "tools_available": len(mcp_tools),
                    "tool_names": [t["name"] for t in mcp_tools],
                }
            
            return {
                "rag_system": rag_stats,
                "agent_system": agent_stats,
                "tools_available": len(get_tools_by_category("all")),
                "status": "healthy"
            }
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def shutdown(self) -> None:
        """Clean up resources, including the MCP server subprocess."""
        if self.mcp_client is not None:
            logger.info("Shutting down MCP client")
            self.mcp_client.stop()
            self.mcp_client = None
        if hasattr(self, "_checkpoint_connection") and self._checkpoint_connection:
            self._checkpoint_connection.close()
            self._checkpoint_connection = None
        logger.info("Agent shutdown complete")


# =============================================================================
# Interactive Chat Interface
# =============================================================================

class InteractiveChatSession:
    """
    Interactive command-line chat interface for the matching agent.
    
    Provides a user-friendly conversational interface with command support,
    multi-turn conversations, and human feedback handling.
    """
    
    def __init__(self, agent: LangGraphMatchingAgent):
        """Initialize chat session with agent instance."""
        self.agent = agent
        self.thread_id = f"chat_{int(time.time())}"
        self.session_active = True
        
        # Chat commands
        self.commands = {
            "/help": self._show_help,
            "/stats": self._show_stats,
            "/clear": self._clear_session,
            "/quit": self._quit_session,
            "/export": self._export_conversation,
            "/approve": self._approve,
            "/refine": self._refine,
            "/screen": self._screen,
            "/reports": self._show_reports,
            "/threads": self._show_threads,
        }
    
    def start(self) -> None:
        """Start the interactive chat session."""
        print("🤖 LangGraph Matching Agent - Interactive Chat")
        print("=" * 50)
        print(f"Session ID: {self.thread_id}")
        print("Type '/help' for commands or just ask me about job matching!")
        print("=" * 50)
        
        while self.session_active:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue
                
                # Process query
                print("\n🤖 Agent: Processing your request...")
                
                response = self.agent.process_query(
                    query=user_input,
                    thread_id=self.thread_id
                )

                self._render(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
        
        print("Chat session ended.")

    def _render(self, response) -> None:
        """Render an agent response, including any pending decision."""
        print(f"\n🤖 Agent: {response.final_response}")

        # Show what changed after a refinement, so a re-rank is never silent.
        change = response.ranking_changes
        if change and change.get("narrative"):
            print(f"\n🔄 What changed:\n{change['narrative']}")

        # Show candidate matches if available
        if response.candidate_matches:
            print(f"\n📊 Found {len(response.candidate_matches)} matching candidates:")
            for i, match in enumerate(response.candidate_matches[:3], 1):
                score = match.get("enhanced_score", match.get("match_score", 0))
                skills = ", ".join((match.get("matched_skills") or [])[:3]) or "n/a"
                print(f"   {i}. {match.get('candidate_name', '?')} - {score}% match")
                print(f"      Skills: {skills}")

        if response.borderline_candidates:
            print(
                f"\n⚖️  Borderline (see /reports for how to resolve): "
                f"{', '.join(response.borderline_candidates)}"
            )

        if response.screening_complete:
            print("\n✅ Multi-round screening complete.")

        # Handle errors
        if response.error:
            print(f"\n⚠️ Error: {response.error}")

        # Surface the human-in-the-loop checkpoint.
        if response.awaiting_feedback:
            request = self.agent.get_pending_feedback_request(self.thread_id)
            if request:
                print(f"\n⏸️  {request['message']}")
                for option in request["options"]:
                    print(f"     • {option}")
                print(
                    "   Reply in plain language, or use /approve, "
                    "/refine <criteria>, /screen."
                )

    # ---- Feedback commands ------------------------------------------------

    def _require_pending(self) -> bool:
        """Warn and return False when nothing is awaiting feedback."""
        if self.agent.is_awaiting_feedback(self.thread_id):
            return True
        print("\nℹ️  Nothing is awaiting feedback right now. Run a search first.")
        return False

    def _approve(self, args: List[str]) -> None:
        """Approve the current results."""
        if not self._require_pending():
            return
        print("\n🤖 Agent: Approving results...")
        self._render(self.agent.provide_feedback(
            self.thread_id, {"feedback_type": "approve", "content": "approved"}
        ))

    def _refine(self, args: List[str]) -> None:
        """Refine the search criteria and re-rank."""
        if not args:
            print("\nUsage: /refine <new criteria>   e.g. /refine also require React and 5+ years")
            return
        criteria = " ".join(args)
        if not self._require_pending():
            # Not paused, so send it as an ordinary refinement turn.
            print(f"\n🤖 Agent: Refining with '{criteria}'...")
            self._render(self.agent.process_query(criteria, thread_id=self.thread_id))
            return
        print(f"\n🤖 Agent: Refining with '{criteria}'...")
        self._render(self.agent.provide_feedback(
            self.thread_id, {"feedback_type": "refine", "content": criteria}
        ))

    def _screen(self, args: List[str]) -> None:
        """Run multi-round screening through to a hire/no-hire call."""
        if not self._require_pending():
            print("   Tip: ask 'run the full screening process' after a search.")
            return
        print("\n🤖 Agent: Running multi-round screening...")
        self._render(self.agent.provide_feedback(
            self.thread_id, {"feedback_type": "screen", "content": "run screening"}
        ))

    def _show_reports(self, args: List[str]) -> None:
        """Show the detailed per-candidate match reports."""
        state = self.agent.get_conversation_state(self.thread_id) or {}
        reports = state.get("match_reports") or []
        if not reports:
            print("\nℹ️  No match reports yet. Run a candidate search first.")
            return

        import explainability as explain
        print("\n" + explain.format_match_reports(
            reports, state.get("job_title") or "the role"
        ))

    def _show_threads(self, args: List[str]) -> None:
        """List known conversation threads."""
        conversations = self.agent.list_active_conversations()
        if not conversations:
            print("\nℹ️  No conversation threads recorded yet.")
            return
        print(f"\n🧵 {len(conversations)} conversation thread(s):")
        for conversation in conversations:
            marker = " (awaiting feedback)" if conversation["awaiting_feedback"] else ""
            current = " <- current" if conversation["thread_id"] == self.thread_id else ""
            print(
                f"   {conversation['thread_id']}: phase={conversation['workflow_phase']}, "
                f"candidates={conversation['candidate_count']}, "
                f"refinements={conversation['refinement_count']}{marker}{current}"
            )
    
    def _handle_command(self, command: str) -> None:
        """Handle chat commands."""
        cmd_parts = command.split()
        cmd_name = cmd_parts[0]
        
        if cmd_name in self.commands:
            self.commands[cmd_name](cmd_parts[1:] if len(cmd_parts) > 1 else [])
        else:
            print(f"Unknown command: {cmd_name}. Type '/help' for available commands.")
    
    def _show_help(self, args: List[str]) -> None:
        """Show help information."""
        print("\n📖 Available Commands:")
        print("  /help              - Show this help message")
        print("  /stats             - Show system statistics")
        print("  /reports           - Detailed match reports (strengths, gaps, suggestions)")
        print("  /threads           - List conversation threads")
        print("  /clear             - Start a new conversation")
        print("  /quit              - End the chat session")
        print("  /export            - Export conversation history")
        print("\n⏸️  When the agent pauses for your decision:")
        print("  /approve           - Accept the current results")
        print("  /refine <criteria> - Adjust criteria and re-rank")
        print("  /screen            - Run multi-round screening to a hire/no-hire call")
        print("\n💡 Example Queries:")
        print("  • 'Find candidates for a Python developer role'")
        print("  • 'Find me candidates with React and 3+ years experience'")
        print("  • 'Compare the top 3 matches side by side'")
        print("  • 'Why did John rank higher than Jane?'")
        print("  • 'Run the full screening process'")
        print("  • 'Generate interview questions for Sarah Wilson'")
        print("  • 'Read the job description from senior_developer.txt'")
    
    def _show_stats(self, args: List[str]) -> None:
        """Show system statistics."""
        stats = self.agent.get_system_stats()
        print("\n📊 System Statistics:")
        print(f"  Status: {stats.get('status', 'unknown')}")
        print(f"  RAG Database: {stats.get('rag_system', {}).get('total_chunks', 0)} chunks")
        print(f"  Available Tools: {stats.get('tools_available', 0)}")
        print(f"  Agent Version: {stats.get('agent_system', {}).get('agent_version', 'unknown')}")
    
    def _clear_session(self, args: List[str]) -> None:
        """Start a new conversation."""
        self.thread_id = f"chat_{int(time.time())}"
        print(f"\n🔄 Started new conversation session: {self.thread_id}")
    
    def _quit_session(self, args: List[str]) -> None:
        """End the chat session."""
        self.session_active = False
        print("\n👋 Ending chat session. Goodbye!")
    
    def _export_conversation(self, args: List[str]) -> None:
        """Export conversation history."""
        try:
            filename = args[0] if args else f"conversation_{self.thread_id}.json"
            
            state = self.agent.get_conversation_state(self.thread_id)
            if state:
                export_data = {
                    "thread_id": self.thread_id,
                    "export_time": datetime.now().isoformat(),
                    "messages": [{"type": getattr(msg, 'type', 'unknown'), "content": getattr(msg, 'content', str(msg))} for msg in state.get("messages", [])],
                    "workflow_history": state.get("workflow_history", []),
                    "candidate_matches": state.get("candidate_matches", [])
                }
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                print(f"\n💾 Conversation exported to {filename}")
            else:
                print("\n⚠️ No conversation history found to export.")
                
        except Exception as e:
            print(f"\n❌ Export failed: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the matching agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangGraph Matching Agent")
    parser.add_argument("--resume-dir", default="examples/resumes", help="Resume directory to initialize with")
    parser.add_argument("--config-file", help="JSON configuration file")
    parser.add_argument("--checkpoint-db", help="SQLite checkpoint database file")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat session")
    parser.add_argument(
        "--workflow", choices=["full", "simple"], default="full",
        help="Graph to run: 'full' is the complete Milestone 3 workflow, "
             "'simple' is the legacy two-node loop"
    )
    parser.add_argument(
        "--no-hitl", action="store_true",
        help="Disable the human feedback pause (runs straight through)"
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip resume ingestion and use the existing vector index"
    )
    parser.add_argument(
        "--use-mcp", action="store_true",
        help="Route file operations through the filesystem MCP server "
             "instead of calling fs_tools directly"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Failed to load config file: {e}")
            return

    # Command-line flags override the config file.
    config["workflow"] = args.workflow
    if args.no_hitl:
        config["enable_hitl"] = False
    if args.use_mcp:
        config["use_mcp"] = True
    
    try:
        # Initialize agent
        print("🚀 Initializing LangGraph Matching Agent...")
        if args.use_mcp:
            print("🔌 MCP mode enabled — file operations will use the MCP server")
        agent = LangGraphMatchingAgent(
            config=config,
            checkpoint_db=args.checkpoint_db
        )
        
        # Initialize with resumes if directory exists
        if args.skip_ingest:
            stats = agent.get_system_stats()
            chunks = stats.get("rag_system", {}).get("total_chunks", 0)
            print(f"⏭️  Skipping ingestion, using existing index ({chunks} chunks)")
        elif os.path.exists(args.resume_dir):
            print(f"📂 Processing resumes from {args.resume_dir}...")
            init_result = agent.initialize_with_resumes(args.resume_dir)
            if init_result["success"]:
                print(f"✅ Processed {init_result['processed_count']} resumes in {init_result['processing_time']:.2f}s")
            else:
                print(f"❌ Resume initialization failed: {init_result['error']}")
        
        # Start interactive session or run example
        if args.interactive:
            chat_session = InteractiveChatSession(agent)
            chat_session.start()
        else:
            # Run example query
            print("\n🧪 Running example query...")
            response = agent.process_query(
                "Find candidates for a Senior Python Developer role with machine learning experience"
            )
            print(f"Response: {response.final_response}")
            if response.candidate_matches:
                print(f"Found {len(response.candidate_matches)} matches")
    
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return 1
    finally:
        # Ensure the MCP server subprocess is cleaned up.
        try:
            agent.shutdown()
        except Exception:
            pass
    
    return 0


if __name__ == "__main__":
    exit(main())