"""
Agent State Schema for LangGraph Matching Agent
===============================================

Defines the state schema and type definitions for the conversational
job matching agent with multi-round screening capabilities.

This module provides:
- MatchingAgentState with typed fields and reducers
- Conversation tracking with message history
- Job requirement and candidate management
- Workflow phase tracking and human feedback integration
"""

from typing import Annotated, Dict, List, Literal, Optional, Any
from typing_extensions import TypedDict
import operator

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class JobRequirementDict(TypedDict, total=False):
    """Structured job requirement information as dictionary."""
    title: str
    skills_required: List[str]
    skills_preferred: List[str]
    experience_years: Optional[int]
    education_required: List[str]
    must_have_keywords: List[str]
    nice_to_have_keywords: List[str]
    description: str


class CandidateMatchDict(TypedDict, total=False):
    """Candidate match information as dictionary."""
    candidate_name: str
    resume_path: str
    match_score: int
    matched_skills: List[str]
    missing_skills: List[str]
    relevant_excerpts: List[str]
    reasoning: str
    section_scores: Dict[str, float]
    keyword_matches: Dict[str, int]
    experience_match: bool
    education_match: bool


class ScreeningRoundData(TypedDict, total=False):
    """Screening round information."""
    round_number: int
    candidates: List[CandidateMatchDict]
    threshold: float
    criteria: Dict[str, Any]
    decision: Optional[str]
    reasoning: str


class HumanFeedbackData(TypedDict, total=False):
    """Human feedback information."""
    feedback_type: Literal["approve", "refine", "reject", "clarify"]
    content: str
    timestamp: str
    approved: bool
    modifications: Optional[Dict[str, Any]]


class MatchReportData(TypedDict, total=False):
    """
    Detailed per-candidate explainability report (Part C.2).

    Captures not just what matched, but what did not, and what a borderline
    candidate would need in order to clear the bar.
    """
    candidate_name: str
    rank: int
    match_score: float
    verdict: Literal["strong", "solid", "borderline", "weak"]
    strengths: List[str]
    gaps: List[str]
    must_have_met: List[str]
    must_have_missing: List[str]
    nice_to_have_met: List[str]
    experience_assessment: str
    improvement_suggestions: List[str]
    is_borderline: bool
    narrative: str


class RankingChangeData(TypedDict, total=False):
    """
    Ranking delta between two successive rankings (Part B.2).

    Lets the agent explain *why* the leaderboard moved after a refinement
    instead of silently returning a different list.
    """
    trigger: str
    criteria_before: Dict[str, Any]
    criteria_after: Dict[str, Any]
    entered: List[str]
    dropped: List[str]
    moved_up: List[Dict[str, Any]]
    moved_down: List[Dict[str, Any]]
    unchanged: List[str]
    score_deltas: Dict[str, float]
    narrative: str


class RefinementRecord(TypedDict, total=False):
    """A single mid-conversation criteria adjustment."""
    turn: int
    timestamp: float
    user_request: str
    parsed_adjustments: Dict[str, Any]
    criteria_before: Dict[str, Any]
    criteria_after: Dict[str, Any]


class MatchingAgentState(TypedDict):
    """
    Comprehensive state schema for the LangGraph matching agent.
    
    Tracks conversation history, job requirements, candidate data,
    workflow phases, and human feedback throughout the matching process.
    """
    
    # Conversation Management (append-only with reducer)
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Session Control
    thread_id: str
    intent: Optional[Literal[
        "file_ops", "job_match", "hybrid", "clarify", "compare", "explain",
        "questions", "screen", "refine", "error"
    ]]
    workflow_phase: Literal[
        "classify", "gather_context", "parse_job", "extract_requirements", 
        "search_resumes", "rank_candidates", "generate_report", "screening",
        "deep_analysis", "final_recommendation", "human_feedback", "done", "error",
        # Additive phases used by the full workflow graph
        "started", "clarify", "ingest", "index_check", "adjust_criteria",
        "file_ops", "compare", "explain", "questions"
    ]
    iteration_count: int
    max_iterations: int
    
    # User Query Context
    raw_user_query: str
    processed_query: Optional[str]
    query_intent: Optional[str]
    
    # Job Matching Domain
    job_description: Optional[str]
    job_title: Optional[str]
    parsed_requirements: Optional[JobRequirementDict]
    search_parameters: Dict[str, Any]
    
    # Matching Configuration
    top_k: int
    min_score: float
    scoring_weights: Optional[Dict[str, float]]
    
    # Results and Candidates (accumulative with reducer)
    semantic_hits: Annotated[List[Dict[str, Any]], operator.add]
    candidate_matches: Annotated[List[CandidateMatchDict], operator.add]
    candidate_shortlist: Annotated[List[str], operator.add]  # candidate names/ids
    
    # Multi-Round Screening
    screening_round: int
    max_screening_rounds: int
    screening_history: Annotated[List[ScreeningRoundData], operator.add]
    current_screening_data: Optional[ScreeningRoundData]
    
    # Audit and Logging (accumulative)
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]
    workflow_history: Annotated[List[str], operator.add]
    performance_metrics: Dict[str, Any]
    
    # File Operations Context
    target_directory: Optional[str]
    file_operations: Annotated[List[Dict[str, Any]], operator.add]
    pending_write: Optional[Dict[str, Any]]  # {filepath, content} awaiting approval
    
    # Human-in-the-Loop
    awaiting_human: bool
    human_feedback: Optional[HumanFeedbackData]
    human_approved: Optional[bool]
    feedback_history: Annotated[List[HumanFeedbackData], operator.add]
    
    # Explanation and Reporting
    explanation_request: Optional[str]
    comparison_candidates: List[str]
    generated_explanations: Annotated[List[Dict[str, Any]], operator.add]
    generated_reports: Annotated[List[Dict[str, Any]], operator.add]
    
    # Interview and Assessment
    interview_questions: Annotated[List[Dict[str, Any]], operator.add]
    assessment_notes: Annotated[List[str], operator.add]
    
    # =====================================================================
    # Authoritative ranking view + refinement tracking
    # =====================================================================
    # NOTE ON SEMANTICS: `candidate_matches` above uses an operator.add
    # reducer, so it *accumulates* across the conversation and serves as an
    # append-only audit trail of every match ever produced. That is the wrong
    # semantics for re-ranking, where a refined search must *replace* the
    # previous leaderboard rather than concatenate onto it.
    #
    # `ranked_candidates` therefore holds the current authoritative ranking
    # with plain replace-on-write semantics. Nodes dual-write: they append to
    # `candidate_matches` for audit and overwrite `ranked_candidates` for
    # truth. Readers that need "the current top N" should use
    # `ranked_candidates` and fall back to `candidate_matches`.
    ranked_candidates: List[CandidateMatchDict]
    previous_ranking: List[CandidateMatchDict]
    ranking_changes: Optional[RankingChangeData]

    # Iterative refinement (Part B.2)
    active_requirements: Optional[JobRequirementDict]
    refinement_history: Annotated[List[RefinementRecord], operator.add]
    refinement_count: int
    pending_refinement: Optional[Dict[str, Any]]

    # Explainability (Part C.2)
    match_reports: List[MatchReportData]
    borderline_candidates: List[str]

    # Multi-round screening control (Part C.1)
    screening_complete: bool
    screening_requested: bool
    
    # Final Output
    final_response: Optional[str]
    final_recommendations: Optional[List[Dict[str, Any]]]
    error: Optional[str]
    error_details: Optional[Dict[str, Any]]


class AgentInput(TypedDict):
    """Input schema for agent invocation."""
    messages: List[AnyMessage]
    thread_id: Optional[str]
    config: Optional[Dict[str, Any]]


class AgentOutput(TypedDict):
    """Output schema for agent responses."""
    messages: List[AnyMessage]
    final_response: Optional[str]
    candidate_matches: List[CandidateMatchDict]
    recommendations: Optional[List[Dict[str, Any]]]
    error: Optional[str]


def create_initial_state(
    messages: List[AnyMessage], 
    thread_id: str,
    config: Optional[Dict[str, Any]] = None
) -> MatchingAgentState:
    """
    Create initial agent state with default values.
    
    Args:
        messages: Initial conversation messages
        thread_id: Unique thread identifier for conversation tracking
        config: Optional configuration parameters
        
    Returns:
        Initialized MatchingAgentState
    """
    config = config or {}
    
    return MatchingAgentState(
        # Conversation
        messages=messages,
        thread_id=thread_id,
        
        # Session Control
        intent=None,
        workflow_phase="classify",
        iteration_count=0,
        max_iterations=config.get("max_iterations", 10),
        
        # Query Context
        raw_user_query="",
        processed_query=None,
        query_intent=None,
        
        # Job Matching
        job_description=None,
        job_title=None,
        parsed_requirements=None,
        search_parameters={},
        
        # Configuration
        top_k=config.get("top_k", 10),
        min_score=config.get("min_score", 0.3),
        scoring_weights=None,
        
        # Results
        semantic_hits=[],
        candidate_matches=[],
        candidate_shortlist=[],
        
        # Screening
        screening_round=0,
        max_screening_rounds=config.get("max_screening_rounds", 3),
        screening_history=[],
        current_screening_data=None,
        
        # Audit
        tool_calls=[],
        workflow_history=[],
        performance_metrics={},
        
        # File Operations
        target_directory=None,
        file_operations=[],
        pending_write=None,
        
        # Human Feedback
        awaiting_human=False,
        human_feedback=None,
        human_approved=None,
        feedback_history=[],
        
        # Explanation
        explanation_request=None,
        comparison_candidates=[],
        generated_explanations=[],
        generated_reports=[],
        
        # Assessment
        interview_questions=[],
        assessment_notes=[],

        # Authoritative ranking view (replace semantics)
        ranked_candidates=[],
        previous_ranking=[],
        ranking_changes=None,

        # Iterative refinement (Part B.2)
        active_requirements=None,
        refinement_history=[],
        refinement_count=0,
        pending_refinement=None,

        # Explainability (Part C.2)
        match_reports=[],
        borderline_candidates=[],

        # Multi-round screening control (Part C.1)
        screening_complete=False,
        screening_requested=False,
        
        # Output
        final_response=None,
        final_recommendations=None,
        error=None,
        error_details=None
    )


def update_workflow_phase(
    state: MatchingAgentState, 
    new_phase: str,
    iteration_increment: bool = True
) -> Dict[str, Any]:
    """
    Helper function to update workflow phase with proper tracking.
    
    Args:
        state: Current agent state
        new_phase: New workflow phase to transition to
        iteration_increment: Whether to increment iteration count
        
    Returns:
        State update dictionary
    """
    updates = {
        "workflow_phase": new_phase,
        "workflow_history": [f"Phase: {new_phase}"]
    }
    
    if iteration_increment:
        updates["iteration_count"] = state.get("iteration_count", 0) + 1
    
    return updates


def is_circuit_breaker_triggered(state: MatchingAgentState) -> bool:
    """
    Check if circuit breaker should be triggered to prevent infinite loops.
    
    Args:
        state: Current agent state
        
    Returns:
        True if circuit breaker should be triggered
    """
    return (
        state.get("iteration_count", 0) >= state.get("max_iterations", 10) or
        state.get("screening_round", 0) >= state.get("max_screening_rounds", 3)
    )


def extract_user_message_content(state: MatchingAgentState) -> str:
    """
    Extract the latest user message content from the conversation.
    
    Args:
        state: Current agent state
        
    Returns:
        Content of the most recent user message
    """
    messages = state.get("messages", [])
    for message in reversed(messages):
        if hasattr(message, 'type') and message.type == "human":
            return message.content
    return ""


def get_current_ranking(state: MatchingAgentState) -> List[CandidateMatchDict]:
    """
    Resolve the current authoritative candidate ranking.

    Prefers `ranked_candidates` (replace-on-write, always the latest ranking).
    Falls back to `candidate_matches`, which uses an append-only reducer and
    may therefore contain duplicates from earlier rounds; in that case the
    list is de-duplicated by candidate name, keeping the highest-scoring
    record for each person.

    Args:
        state: Current agent state

    Returns:
        Current ranking, ordered best-first
    """
    ranked = state.get("ranked_candidates") or []
    if ranked:
        return list(ranked)

    # Fallback: de-duplicate the append-only audit trail.
    seen: Dict[str, CandidateMatchDict] = {}
    for match in state.get("candidate_matches", []) or []:
        name = match.get("candidate_name")
        if not name:
            continue
        score = match.get("enhanced_score", match.get("match_score", 0)) or 0
        existing = seen.get(name)
        if existing is None:
            seen[name] = match
            continue
        existing_score = existing.get("enhanced_score", existing.get("match_score", 0)) or 0
        if score > existing_score:
            seen[name] = match

    return sorted(
        seen.values(),
        key=lambda m: m.get("enhanced_score", m.get("match_score", 0)) or 0,
        reverse=True
    )


def get_active_requirements(state: MatchingAgentState) -> Optional[JobRequirementDict]:
    """
    Resolve the requirements currently in force.

    After a mid-conversation refinement, `active_requirements` holds the
    adjusted criteria while `parsed_requirements` keeps the original
    extraction from the job description. Nodes that score or filter should
    use this helper so refinements actually take effect.

    Args:
        state: Current agent state

    Returns:
        Active requirements, or the original parsed requirements, or None
    """
    return state.get("active_requirements") or state.get("parsed_requirements")