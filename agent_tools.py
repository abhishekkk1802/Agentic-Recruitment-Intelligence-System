"""
Agent Tools for LangGraph Matching Agent
========================================

Tool wrappers and implementations for the conversational job matching agent.
Provides LangGraph-compatible tools that wrap existing fs_tools, RAG, and
job matching functionality with proper schemas and error handling.

This module provides:
- File system tool wrappers (read, list, write, search)
- RAG system tool wrappers (search, ingest, stats)
- Job matching tool wrappers (parse, match, analyze)
- New specialized tools for agent workflows
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Import existing system components
from fs_tools import read_file, list_files, write_file, search_in_file
from resume_rag import ResumeRAG
from job_matcher import JobMatcher, JobRequirement
from agent_state import JobRequirementDict, CandidateMatchDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances - will be initialized by the agent
_rag_instance: Optional[ResumeRAG] = None
_matcher_instance: Optional[JobMatcher] = None


def initialize_tools(rag: ResumeRAG, matcher: JobMatcher) -> None:
    """
    Initialize global tool instances.
    
    Args:
        rag: ResumeRAG instance for vector operations
        matcher: JobMatcher instance for candidate matching
    """
    global _rag_instance, _matcher_instance
    _rag_instance = rag
    _matcher_instance = matcher
    logger.info("Agent tools initialized with RAG and Matcher instances")


def get_rag_instance() -> ResumeRAG:
    """Get the global RAG instance."""
    if _rag_instance is None:
        raise RuntimeError("RAG instance not initialized. Call initialize_tools() first.")
    return _rag_instance


def get_matcher_instance() -> JobMatcher:
    """Get the global JobMatcher instance."""
    if _matcher_instance is None:
        raise RuntimeError("JobMatcher instance not initialized. Call initialize_tools() first.")
    return _matcher_instance


# =============================================================================
# File System Tools (Wrapped from fs_tools.py)
# =============================================================================

class ReadFileInput(BaseModel):
    """Input schema for read_file tool."""
    filepath: str = Field(description="Path to the file to read (supports PDF, TXT, DOCX)")


@tool(args_schema=ReadFileInput)
def read_file_tool(filepath: str) -> Dict[str, Any]:
    """
    Read and extract text content from files (PDF, TXT, DOCX).
    Returns structured data with content, metadata, and success status.
    """
    try:
        result = read_file(filepath)
        logger.info(f"Successfully read file: {filepath}")
        return result
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "content": None,
            "metadata": {"filepath": filepath}
        }
        logger.error(f"Failed to read file {filepath}: {e}")
        return error_result


class ListFilesInput(BaseModel):
    """Input schema for list_files tool."""
    directory: str = Field(description="Directory path to list files from")
    extension: Optional[str] = Field(default=None, description="Optional file extension filter (e.g., '.txt', '.pdf')")


@tool(args_schema=ListFilesInput)
def list_files_tool(directory: str, extension: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List files in a directory with optional extension filtering.
    Returns list of file information including paths, sizes, and metadata.
    """
    try:
        result = list_files(directory, extension)
        logger.info(f"Listed {len(result)} files in {directory}")
        return result
    except Exception as e:
        logger.error(f"Failed to list files in {directory}: {e}")
        return [{"error": str(e), "directory": directory}]


class WriteFileInput(BaseModel):
    """Input schema for write_file tool."""
    filepath: str = Field(description="Path where to write the file")
    content: str = Field(description="Content to write to the file")


@tool(args_schema=WriteFileInput)
def write_file_tool(filepath: str, content: str) -> Dict[str, Any]:
    """
    Write content to a file with automatic directory creation.
    Returns success status and file information.
    """
    try:
        result = write_file(filepath, content)
        logger.info(f"Successfully wrote file: {filepath}")
        return result
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "filepath": filepath,
            "bytes_written": 0
        }
        logger.error(f"Failed to write file {filepath}: {e}")
        return error_result


class SearchFileInput(BaseModel):
    """Input schema for search_in_file tool."""
    filepath: str = Field(description="Path to the file to search in")
    keyword: str = Field(description="Keyword or phrase to search for")


@tool(args_schema=SearchFileInput)
def search_in_file_tool(filepath: str, keyword: str) -> Dict[str, Any]:
    """
    Search for keywords within a file with context.
    Returns matches with surrounding context and line numbers.
    """
    try:
        result = search_in_file(filepath, keyword)
        logger.info(f"Found {result.get('total_matches', 0)} matches for '{keyword}' in {filepath}")
        return result
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "matches": [],
            "total_matches": 0,
            "filepath": filepath,
            "keyword": keyword
        }
        logger.error(f"Failed to search in file {filepath}: {e}")
        return error_result


# =============================================================================
# RAG System Tools (Wrapped from ResumeRAG)
# =============================================================================

class SemanticSearchInput(BaseModel):
    """Input schema for semantic search tool."""
    query: str = Field(description="Search query for semantic similarity matching")
    k: int = Field(default=10, ge=1, le=50, description="Number of top results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filters")


@tool(args_schema=SemanticSearchInput)
def semantic_search_resumes(query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Search indexed resumes using semantic similarity.
    Returns ranked results with similarity scores and metadata.
    """
    try:
        rag = get_rag_instance()
        start_time = time.time()
        
        results = rag.semantic_search(query, k=k, filters=filters)
        
        search_time = time.time() - start_time
        logger.info(f"Semantic search completed in {search_time:.3f}s, found {len(results)} results")
        
        # Add performance metadata
        for result in results:
            result["search_time"] = search_time
            result["search_query"] = query
        
        return results
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return [{"error": str(e), "query": query}]


class IngestResumeInput(BaseModel):
    """Input schema for resume ingestion tool."""
    directory: str = Field(description="Directory containing resume files to ingest")


@tool(args_schema=IngestResumeInput)
def ingest_resume_directory(directory: str) -> Dict[str, Any]:
    """
    Ingest and index all resumes from a directory.
    Processes files, extracts metadata, and updates vector database.
    """
    try:
        rag = get_rag_instance()
        start_time = time.time()
        
        processed_resumes = rag.process_resume_directory(directory)
        
        processing_time = time.time() - start_time
        
        result = {
            "success": True,
            "directory": directory,
            "processed_count": len(processed_resumes),
            "processing_time": processing_time,
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
        
        logger.info(f"Processed {len(processed_resumes)} resumes in {processing_time:.3f}s")
        return result
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "directory": directory,
            "processed_count": 0
        }
        logger.error(f"Failed to ingest resumes from {directory}: {e}")
        return error_result


@tool
def get_rag_stats() -> Dict[str, Any]:
    """
    Get statistics about the RAG system and vector database.
    Returns collection info, chunk counts, and system health metrics.
    """
    try:
        rag = get_rag_instance()
        stats = rag.get_collection_stats()
        
        logger.info("Retrieved RAG system statistics")
        return stats
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "total_chunks": 0,
            "collection_name": "unknown",
            "embedding_model": "unknown"
        }
        logger.error(f"Failed to get RAG stats: {e}")
        return error_result


# =============================================================================
# Job Matching Tools (Wrapped from JobMatcher)
# =============================================================================

class ParseJobInput(BaseModel):
    """Input schema for job parsing tool."""
    job_description: str = Field(description="Raw job description text to parse")
    job_title: Optional[str] = Field(default="", description="Optional job title")


@tool(args_schema=ParseJobInput)
def parse_job_description(job_description: str, job_title: str = "") -> JobRequirementDict:
    """
    Parse job description into structured requirements.
    Extracts required/preferred skills, experience, education, and keywords.
    """
    try:
        matcher = get_matcher_instance()
        
        job_req = matcher.parse_job_description(job_description, job_title)
        
        # Convert to dictionary for JSON serialization
        result = JobRequirementDict(
            title=job_req.title,
            skills_required=job_req.skills_required,
            skills_preferred=job_req.skills_preferred,
            experience_years=job_req.experience_years,
            education_required=job_req.education_required,
            must_have_keywords=job_req.must_have_keywords,
            nice_to_have_keywords=job_req.nice_to_have_keywords,
            description=job_req.description
        )
        
        logger.info(f"Parsed job requirements for '{job_req.title}' with {len(job_req.skills_required)} required skills")
        return result
        
    except Exception as e:
        logger.error(f"Failed to parse job description: {e}")
        raise


class FindMatchesInput(BaseModel):
    """Input schema for finding job matches."""
    job_description: str = Field(description="Job description text")
    job_title: Optional[str] = Field(default="", description="Job title")
    k: int = Field(default=10, ge=1, le=50, description="Number of top matches to return")
    min_score: float = Field(default=30.0, ge=0.0, le=100.0, description="Minimum match score (0-100)")


@tool(args_schema=FindMatchesInput)
def find_job_matches(
    job_description: str, 
    job_title: str = "", 
    k: int = 10, 
    min_score: float = 30.0
) -> Dict[str, Any]:
    """
    Find top resume candidates matching a job description using hybrid RAG scoring.
    Returns ranked candidates with match scores, skills, and reasoning.
    """
    try:
        matcher = get_matcher_instance()
        start_time = time.time()
        
        # Convert min_score to 0-1 range for internal processing
        min_score_normalized = min_score / 100.0
        
        matches = matcher.find_matches(
            job_description=job_description,
            job_title=job_title,
            k=k,
            min_score=min_score_normalized
        )
        
        matching_time = time.time() - start_time
        
        # Add performance metadata
        matches["matching_time"] = matching_time
        matches["search_parameters"] = {
            "k": k,
            "min_score": min_score,
            "job_title": job_title
        }
        
        logger.info(f"Found {len(matches.get('top_matches', []))} matches in {matching_time:.3f}s")
        return matches
        
    except Exception as e:
        error_result = {
            "job_description": job_description,
            "top_matches": [],
            "error": str(e),
            "search_parameters": {"k": k, "min_score": min_score}
        }
        logger.error(f"Failed to find job matches: {e}")
        return error_result


class AnalyzeCandidateInput(BaseModel):
    """Input schema for detailed candidate analysis."""
    candidate_name: str = Field(description="Name of the candidate to analyze")
    job_requirements: Optional[JobRequirementDict] = Field(default=None, description="Job requirements for comparison")


@tool(args_schema=AnalyzeCandidateInput)
def analyze_candidate_profile(
    candidate_name: str, 
    job_requirements: Optional[JobRequirementDict] = None
) -> Dict[str, Any]:
    """
    Perform detailed analysis of a specific candidate profile.
    Returns comprehensive breakdown of skills, experience, and job fit.
    """
    try:
        rag = get_rag_instance()
        
        # Search for candidate's resume chunks
        candidate_results = rag.semantic_search(
            query=candidate_name, 
            k=20,
            filters={"resume_name": candidate_name} if candidate_name else None
        )
        
        if not candidate_results:
            return {
                "candidate_name": candidate_name,
                "error": "Candidate not found in database",
                "analysis": {}
            }
        
        # Aggregate candidate information
        candidate_data = candidate_results[0]["metadata"]
        
        analysis = {
            "candidate_name": candidate_name,
            "resume_path": candidate_data.get("file_path", ""),
            "skills": candidate_data.get("skills", "").split(",") if candidate_data.get("skills") else [],
            "programming_languages": candidate_data.get("programming_languages", "").split(",") if candidate_data.get("programming_languages") else [],
            "experience_years": candidate_data.get("experience_years"),
            "sections_found": len(candidate_results),
            "total_content_length": sum(len(chunk["content"]) for chunk in candidate_results),
            "sections": [
                {
                    "type": chunk["metadata"].get("section", "unknown"),
                    "content_preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                    "chunk_index": chunk["metadata"].get("chunk_index", 0)
                }
                for chunk in candidate_results[:5]  # Top 5 sections
            ]
        }
        
        # Add job fit analysis if requirements provided
        if job_requirements:
            required_skills = set(skill.lower() for skill in job_requirements.get("skills_required", []))
            candidate_skills = set(skill.lower().strip() for skill in analysis["skills"])
            
            analysis["job_fit"] = {
                "skills_overlap": list(required_skills.intersection(candidate_skills)),
                "missing_skills": list(required_skills - candidate_skills),
                "skill_match_percentage": len(required_skills.intersection(candidate_skills)) / len(required_skills) * 100 if required_skills else 0,
                "experience_match": (
                    analysis["experience_years"] >= job_requirements.get("experience_years", 0) 
                    if analysis["experience_years"] and job_requirements.get("experience_years") 
                    else None
                )
            }
        
        logger.info(f"Analyzed candidate profile for {candidate_name}")
        return analysis
        
    except Exception as e:
        logger.error(f"Failed to analyze candidate {candidate_name}: {e}")
        return {
            "candidate_name": candidate_name,
            "error": str(e),
            "analysis": {}
        }


# =============================================================================
# New Specialized Tools (Assignment Requirements)
# =============================================================================

class ExtractRequirementsInput(BaseModel):
    """Input schema for enhanced requirement extraction."""
    job_description: str = Field(description="Job description text to analyze")
    classification_focus: str = Field(default="all", description="Focus area: 'skills', 'experience', 'education', or 'all'")


@tool(args_schema=ExtractRequirementsInput)
def extract_requirements(job_description: str, classification_focus: str = "all") -> Dict[str, Any]:
    """
    Advanced requirement extraction with must-have vs nice-to-have classification.
    Uses enhanced NLP to distinguish requirement priorities and extract detailed criteria.
    """
    try:
        matcher = get_matcher_instance()
        
        # Use existing parser as base
        base_requirements = matcher.parse_job_description(job_description, "")
        
        # Enhanced extraction with priority classification
        import re
        
        jd_lower = job_description.lower()
        
        # Enhanced keyword extraction patterns
        must_have_patterns = [
            r'must have\s+([\w\s,]+?)(?:\.|;|\n|required)',
            r'required[:\s]+([\w\s,]+?)(?:\.|;|\n)',  
            r'essential[:\s]+([\w\s,]+?)(?:\.|;|\n)',
            r'mandatory[:\s]+([\w\s,]+?)(?:\.|;|\n)',
            r'minimum[:\s]+(\d+\+?\s*years?)\s+(?:of\s+)?experience',
            r'at least\s+(\d+\+?\s*years?)',
        ]
        
        nice_to_have_patterns = [
            r'nice to have\s+([\w\s,]+?)(?:\.|;|\n)',
            r'preferred[:\s]+([\w\s,]+?)(?:\.|;|\n)',
            r'desired[:\s]+([\w\s,]+?)(?:\.|;|\n)',
            r'bonus[:\s]+([\w\s,]+?)(?:\.|;|\n)',
            r'plus[:\s]+([\w\s,]+?)(?:\.|;|\n)',
        ]
        
        # Extract enhanced requirements
        enhanced_must_haves = []
        enhanced_nice_to_haves = []
        
        for pattern in must_have_patterns:
            matches = re.findall(pattern, jd_lower, re.IGNORECASE)
            enhanced_must_haves.extend(matches)
        
        for pattern in nice_to_have_patterns:
            matches = re.findall(pattern, jd_lower, re.IGNORECASE)
            enhanced_nice_to_haves.extend(matches)
        
        # Experience extraction
        experience_patterns = [
            r'(\d+)\+?\s*(?:to\s+\d+)?\s*years?\s+(?:of\s+)?experience',
            r'minimum\s+of\s+(\d+)\s*years?',
            r'at\s+least\s+(\d+)\s*years?',
        ]
        
        experience_requirements = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, jd_lower)
            experience_requirements.extend([int(match) for match in matches])
        
        min_experience = min(experience_requirements) if experience_requirements else base_requirements.experience_years
        
        # Skill categorization
        technical_skills = []
        soft_skills = []
        
        all_skills = base_requirements.skills_required + base_requirements.skills_preferred
        
        # Common technical skill keywords
        technical_keywords = {
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js',
            'sql', 'mysql', 'postgresql', 'mongodb', 'aws', 'azure', 'gcp',
            'docker', 'kubernetes', 'git', 'linux', 'windows', 'machine learning',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'flask',
            'django', 'spring', 'hibernate', 'rest', 'api', 'microservices'
        }
        
        for skill in all_skills:
            if any(tech in skill.lower() for tech in technical_keywords):
                technical_skills.append(skill)
            else:
                soft_skills.append(skill)
        
        result = {
            "basic_requirements": {
                "title": base_requirements.title,
                "skills_required": base_requirements.skills_required,
                "skills_preferred": base_requirements.skills_preferred,
                "experience_years": base_requirements.experience_years,
                "education_required": base_requirements.education_required
            },
            "enhanced_classification": {
                "must_have_extracted": [item.strip() for item in enhanced_must_haves if item.strip()],
                "nice_to_have_extracted": [item.strip() for item in enhanced_nice_to_haves if item.strip()],
                "min_experience_years": min_experience,
                "technical_skills": technical_skills,
                "soft_skills": soft_skills
            },
            "priority_analysis": {
                "critical_requirements": len(enhanced_must_haves),
                "optional_requirements": len(enhanced_nice_to_haves),
                "experience_emphasis": "high" if len(experience_requirements) > 2 else "medium" if experience_requirements else "low",
                "technical_focus": len(technical_skills) / (len(all_skills) + 1) if all_skills else 0
            },
            "extraction_metadata": {
                "timestamp": datetime.now().isoformat(),
                "jd_length": len(job_description),
                "classification_focus": classification_focus,
                "patterns_matched": {
                    "must_have": len(enhanced_must_haves),
                    "nice_to_have": len(enhanced_nice_to_haves),
                    "experience": len(experience_requirements)
                }
            }
        }
        
        logger.info(f"Enhanced requirement extraction completed with {result['priority_analysis']['critical_requirements']} critical requirements")
        return result
        
    except Exception as e:
        logger.error(f"Enhanced requirement extraction failed: {e}")
        return {
            "error": str(e),
            "basic_requirements": {},
            "enhanced_classification": {},
            "priority_analysis": {},
            "extraction_metadata": {"timestamp": datetime.now().isoformat()}
        }


class CompareCandidatesInput(BaseModel):
    """Input schema for candidate comparison."""
    candidate_ids: List[str] = Field(description="List of candidate names/IDs to compare", min_items=2, max_items=5)
    comparison_criteria: Optional[List[str]] = Field(default=None, description="Specific criteria to focus on")
    job_requirements: Optional[JobRequirementDict] = Field(default=None, description="Job requirements for contextualized comparison")


@tool(args_schema=CompareCandidatesInput)
def compare_candidates(
    candidate_ids: List[str], 
    comparison_criteria: Optional[List[str]] = None,
    job_requirements: Optional[JobRequirementDict] = None
) -> Dict[str, Any]:
    """
    Perform head-to-head comparison of multiple candidates.
    Analyzes skills, experience, and job fit with detailed comparison matrix.
    """
    try:
        if len(candidate_ids) < 2:
            raise ValueError("At least 2 candidates required for comparison")
        
        # Get detailed profiles for each candidate
        candidate_profiles = {}
        comparison_errors = []
        
        for candidate_id in candidate_ids:
            try:
                profile = analyze_candidate_profile(candidate_id, job_requirements)
                if "error" not in profile:
                    candidate_profiles[candidate_id] = profile
                else:
                    comparison_errors.append(f"Failed to analyze {candidate_id}: {profile['error']}")
            except Exception as e:
                comparison_errors.append(f"Error analyzing {candidate_id}: {str(e)}")
        
        if len(candidate_profiles) < 2:
            return {
                "error": "Insufficient candidate data for comparison",
                "errors": comparison_errors,
                "candidates_analyzed": list(candidate_profiles.keys())
            }
        
        # Default comparison criteria
        if comparison_criteria is None:
            comparison_criteria = [
                "skills_count", 
                "experience_years", 
                "programming_languages",
                "job_fit_score",
                "content_depth"
            ]
        
        # Build comparison matrix
        comparison_matrix = {}
        
        for criterion in comparison_criteria:
            comparison_matrix[criterion] = {}
            
            for candidate_id, profile in candidate_profiles.items():
                if criterion == "skills_count":
                    value = len(profile.get("skills", []))
                elif criterion == "experience_years":
                    value = profile.get("experience_years", 0) or 0
                elif criterion == "programming_languages":
                    value = len(profile.get("programming_languages", []))
                elif criterion == "job_fit_score":
                    job_fit = profile.get("job_fit", {})
                    value = job_fit.get("skill_match_percentage", 0) if job_fit else 0
                elif criterion == "content_depth":
                    value = profile.get("sections_found", 0)
                else:
                    value = 0  # Unknown criterion
                
                comparison_matrix[criterion][candidate_id] = value
        
        # Calculate rankings for each criterion
        rankings = {}
        for criterion, values in comparison_matrix.items():
            # Sort candidates by criterion value (descending)
            sorted_candidates = sorted(values.items(), key=lambda x: x[1], reverse=True)
            rankings[criterion] = {
                candidate: rank + 1 
                for rank, (candidate, _) in enumerate(sorted_candidates)
            }
        
        # Overall ranking (average of all criterion rankings)
        overall_scores = {}
        for candidate_id in candidate_profiles.keys():
            candidate_ranks = [rankings[criterion][candidate_id] for criterion in comparison_criteria]
            overall_scores[candidate_id] = sum(candidate_ranks) / len(candidate_ranks)
        
        overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1])
        
        # Generate comparison insights
        insights = []
        
        # Best performer per criterion
        for criterion in comparison_criteria:
            best_candidate = min(rankings[criterion].items(), key=lambda x: x[1])
            insights.append(f"Best in {criterion}: {best_candidate[0]} (rank {best_candidate[1]})")
        
        # Strengths and weaknesses analysis
        strengths_weaknesses = {}
        for candidate_id in candidate_profiles.keys():
            strengths = []
            weaknesses = []
            
            for criterion in comparison_criteria:
                rank = rankings[criterion][candidate_id]
                if rank == 1:
                    strengths.append(f"Leading in {criterion}")
                elif rank == len(candidate_profiles):
                    weaknesses.append(f"Trailing in {criterion}")
            
            strengths_weaknesses[candidate_id] = {
                "strengths": strengths,
                "weaknesses": weaknesses,
                "overall_rank": next(i for i, (cand, _) in enumerate(overall_ranking, 1) if cand == candidate_id)
            }
        
        result = {
            "candidates_compared": list(candidate_profiles.keys()),
            "comparison_criteria": comparison_criteria,
            "comparison_matrix": comparison_matrix,
            "rankings_by_criteria": rankings,
            "overall_ranking": overall_ranking,
            "strengths_weaknesses": strengths_weaknesses,
            "insights": insights,
            "recommendation": {
                "top_candidate": overall_ranking[0][0],
                "top_candidate_score": overall_ranking[0][1],
                "recommendation_reason": f"Best average ranking across {len(comparison_criteria)} criteria"
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "candidates_requested": len(candidate_ids),
                "candidates_analyzed": len(candidate_profiles),
                "errors": comparison_errors
            }
        }
        
        logger.info(f"Compared {len(candidate_profiles)} candidates across {len(comparison_criteria)} criteria")
        return result
        
    except Exception as e:
        logger.error(f"Candidate comparison failed: {e}")
        return {
            "error": str(e),
            "candidates_compared": candidate_ids,
            "metadata": {"timestamp": datetime.now().isoformat()}
        }


class GenerateQuestionsInput(BaseModel):
    """Input schema for interview question generation."""
    candidate_id: str = Field(description="Candidate name/ID to generate questions for")
    job_requirements: Optional[JobRequirementDict] = Field(default=None, description="Job requirements to tailor questions")
    question_types: Optional[List[str]] = Field(default=None, description="Types of questions: 'technical', 'behavioral', 'situational'")
    num_questions: int = Field(default=10, ge=3, le=20, description="Number of questions to generate")


@tool(args_schema=GenerateQuestionsInput)
def generate_interview_questions(
    candidate_id: str,
    job_requirements: Optional[JobRequirementDict] = None,
    question_types: Optional[List[str]] = None,
    num_questions: int = 10
) -> Dict[str, Any]:
    """
    Generate tailored interview questions based on candidate profile and job requirements.
    Creates technical, behavioral, and situational questions specific to the candidate's background.
    """
    try:
        # Get candidate profile
        candidate_profile = analyze_candidate_profile(candidate_id, job_requirements)
        
        if "error" in candidate_profile:
            return {
                "error": f"Could not analyze candidate: {candidate_profile['error']}",
                "candidate_id": candidate_id
            }
        
        # Default question types
        if question_types is None:
            question_types = ["technical", "behavioral", "situational"]
        
        questions = []
        
        # Technical questions based on candidate skills
        if "technical" in question_types:
            candidate_skills = candidate_profile.get("skills", [])
            prog_languages = candidate_profile.get("programming_languages", [])
            
            technical_questions = []
            
            # Language-specific questions
            for lang in prog_languages[:3]:  # Top 3 languages
                technical_questions.append({
                    "type": "technical",
                    "category": "programming",
                    "question": f"Can you walk me through a challenging {lang} project you've worked on? What were the main technical hurdles?",
                    "focus": lang,
                    "difficulty": "intermediate"
                })
            
            # Skill-based questions
            for skill in candidate_skills[:3]:  # Top 3 skills
                if skill.lower() in ['aws', 'azure', 'gcp', 'cloud']:
                    technical_questions.append({
                        "type": "technical", 
                        "category": "cloud",
                        "question": f"Describe your experience with {skill}. How would you architect a scalable solution using {skill} services?",
                        "focus": skill,
                        "difficulty": "advanced"
                    })
                elif skill.lower() in ['machine learning', 'ml', 'ai', 'data science']:
                    technical_questions.append({
                        "type": "technical",
                        "category": "data_science",
                        "question": f"Tell me about a {skill} project where you had to choose between different algorithms. How did you make the decision?",
                        "focus": skill,
                        "difficulty": "advanced"
                    })
                else:
                    technical_questions.append({
                        "type": "technical",
                        "category": "general",
                        "question": f"How do you stay current with {skill} best practices and new developments?",
                        "focus": skill,
                        "difficulty": "basic"
                    })
            
            questions.extend(technical_questions[:max(1, num_questions // 3)])
        
        # Behavioral questions based on experience
        if "behavioral" in question_types:
            experience_years = candidate_profile.get("experience_years", 0) or 0
            
            behavioral_questions = []
            
            if experience_years >= 5:
                behavioral_questions.extend([
                    {
                        "type": "behavioral",
                        "category": "leadership",
                        "question": "Tell me about a time when you had to mentor a junior team member. What was your approach?",
                        "focus": "leadership",
                        "difficulty": "intermediate"
                    },
                    {
                        "type": "behavioral",
                        "category": "problem_solving",
                        "question": "Describe a situation where you had to make a critical technical decision under pressure. What was the outcome?",
                        "focus": "decision_making",
                        "difficulty": "advanced"
                    }
                ])
            else:
                behavioral_questions.extend([
                    {
                        "type": "behavioral",
                        "category": "learning",
                        "question": "Tell me about a time when you had to quickly learn a new technology or skill for a project.",
                        "focus": "adaptability",
                        "difficulty": "basic"
                    },
                    {
                        "type": "behavioral", 
                        "category": "collaboration",
                        "question": "Describe a challenging team project. How did you handle conflicts or disagreements?",
                        "focus": "teamwork",
                        "difficulty": "intermediate"
                    }
                ])
            
            # Job-specific behavioral questions
            if job_requirements:
                job_title = job_requirements.get("title", "")
                if "senior" in job_title.lower():
                    behavioral_questions.append({
                        "type": "behavioral",
                        "category": "leadership",
                        "question": "Give me an example of how you've influenced technical decisions in your previous role.",
                        "focus": "influence",
                        "difficulty": "advanced"
                    })
                elif "lead" in job_title.lower():
                    behavioral_questions.append({
                        "type": "behavioral",
                        "category": "leadership",
                        "question": "How do you balance hands-on coding with team leadership responsibilities?",
                        "focus": "management",
                        "difficulty": "advanced"
                    })
            
            questions.extend(behavioral_questions[:max(1, num_questions // 3)])
        
        # Situational questions based on job requirements
        if "situational" in question_types and job_requirements:
            situational_questions = []
            
            required_skills = job_requirements.get("skills_required", [])
            
            for skill in required_skills[:2]:  # Top 2 required skills
                situational_questions.append({
                    "type": "situational",
                    "category": "skill_application",
                    "question": f"If you were tasked with implementing a new feature using {skill}, how would you approach the design and development process?",
                    "focus": skill,
                    "difficulty": "intermediate"
                })
            
            # Experience-based situational
            req_experience = job_requirements.get("experience_years", 0)
            candidate_exp = candidate_profile.get("experience_years", 0) or 0
            
            if req_experience and candidate_exp < req_experience:
                situational_questions.append({
                    "type": "situational",
                    "category": "experience_gap",
                    "question": f"This role requires {req_experience} years of experience, and you have {candidate_exp}. How would you bridge this gap quickly?",
                    "focus": "experience",
                    "difficulty": "challenging"
                })
            
            questions.extend(situational_questions[:max(1, num_questions // 3)])
        
        # Fill remaining slots with general questions
        while len(questions) < num_questions:
            questions.append({
                "type": "general",
                "category": "motivation",
                "question": "What interests you most about this role and our company?",
                "focus": "motivation",
                "difficulty": "basic"
            })
        
        # Trim to requested number
        questions = questions[:num_questions]
        
        result = {
            "candidate_id": candidate_id,
            "total_questions": len(questions),
            "question_types_generated": list(set(q["type"] for q in questions)),
            "questions": questions,
            "interview_structure": {
                "technical_questions": len([q for q in questions if q["type"] == "technical"]),
                "behavioral_questions": len([q for q in questions if q["type"] == "behavioral"]),
                "situational_questions": len([q for q in questions if q["type"] == "situational"]),
                "general_questions": len([q for q in questions if q["type"] == "general"])
            },
            "candidate_context": {
                "experience_years": candidate_profile.get("experience_years"),
                "key_skills": candidate_profile.get("skills", [])[:5],
                "programming_languages": candidate_profile.get("programming_languages", [])
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "generation_parameters": {
                    "requested_types": question_types,
                    "num_questions": num_questions,
                    "job_requirements_provided": job_requirements is not None
                }
            }
        }
        
        logger.info(f"Generated {len(questions)} interview questions for {candidate_id}")
        return result
        
    except Exception as e:
        logger.error(f"Interview question generation failed: {e}")
        return {
            "error": str(e),
            "candidate_id": candidate_id,
            "metadata": {"timestamp": datetime.now().isoformat()}
        }


# =============================================================================
# Utility Tools
# =============================================================================

@tool
def get_current_timestamp() -> str:
    """Get current timestamp in ISO format for logging and tracking."""
    return datetime.now().isoformat()


class SaveResultsInput(BaseModel):
    """Input schema for saving results."""
    filepath: str = Field(description="Path to save the results file")
    data: Dict[str, Any] = Field(description="Data to save as JSON")


@tool(args_schema=SaveResultsInput)
def save_agent_results(filepath: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save agent results to a JSON file with proper formatting.
    Includes timestamp and metadata for audit trails.
    """
    try:
        # Add metadata
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "agent_version": "langgraph_matching_agent_v1.0",
            "data": data
        }
        
        # Use existing write_file function
        content = json.dumps(save_data, indent=2, ensure_ascii=False)
        result = write_file(filepath, content)
        
        logger.info(f"Saved agent results to {filepath}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to save results to {filepath}: {e}")
        return {
            "success": False,
            "error": str(e),
            "filepath": filepath
        }


# =============================================================================
# Tool Collections for Different Agent Nodes
# =============================================================================

# File system tools - safe for auto-execution
FILE_TOOLS = [
    read_file_tool,
    list_files_tool,
    search_in_file_tool
]

# RAG tools - safe for auto-execution
RAG_TOOLS = [
    semantic_search_resumes,
    get_rag_stats,
    analyze_candidate_profile
]

# Matching tools - safe for auto-execution
MATCHING_TOOLS = [
    parse_job_description,
    find_job_matches
]

# New specialized tools - safe for auto-execution
SPECIALIZED_TOOLS = [
    extract_requirements,
    compare_candidates,
    generate_interview_questions
]

# Write tools - require human approval
WRITE_TOOLS = [
    write_file_tool,
    save_agent_results
]

# Ingestion tools - moderate impact, log but auto-execute
INGESTION_TOOLS = [
    ingest_resume_directory
]

# All safe tools (no approval needed)
SAFE_TOOLS = FILE_TOOLS + RAG_TOOLS + MATCHING_TOOLS + SPECIALIZED_TOOLS + [get_current_timestamp]

# All tools
ALL_TOOLS = SAFE_TOOLS + WRITE_TOOLS + INGESTION_TOOLS


def get_tools_by_category(category: str) -> List:
    """
    Get tools by category for different agent nodes.
    
    Args:
        category: Tool category ('safe', 'write', 'ingestion', 'specialized', 'all')
        
    Returns:
        List of tools for the specified category
    """
    categories = {
        'safe': SAFE_TOOLS,
        'write': WRITE_TOOLS, 
        'ingestion': INGESTION_TOOLS,
        'file': FILE_TOOLS,
        'rag': RAG_TOOLS,
        'matching': MATCHING_TOOLS,
        'specialized': SPECIALIZED_TOOLS,
        'all': ALL_TOOLS
    }
    
    return categories.get(category, SAFE_TOOLS)