"""
Multi-Round Screening Pipeline
=============================

Implements the multi-round screening system with progressive filtering
and detailed candidate analysis for hire/no-hire recommendations.

This module provides:
- Three-stage screening process (Initial → Deep Analysis → Final Recommendation)
- Progressive filtering with configurable thresholds
- Detailed candidate analysis and comparison
- Hire/no-hire recommendation generation with justification
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import statistics

from agent_state import (
    MatchingAgentState, 
    ScreeningRoundData,
    CandidateMatchDict,
    JobRequirementDict
)
from agent_tools import (
    analyze_candidate_profile,
    compare_candidates,
    find_job_matches,
    generate_interview_questions
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiRoundScreeningPipeline:
    """
    Multi-round screening pipeline for progressive candidate filtering.
    
    Implements a three-stage process:
    1. Initial Screen: Process all resumes → Top 10-15 candidates
    2. Deep Analysis: Enhanced scoring of top candidates → Top 3-5 finalists
    3. Final Round: Detailed analysis and hire/no-hire recommendations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the screening pipeline.
        
        Args:
            config: Configuration parameters for screening thresholds and criteria
        """
        self.config = config or {}
        
        # Default screening configuration
        self.screening_config = {
            "round_1": {
                "name": "Initial Screen",
                "target_candidates": self.config.get("round_1_target", 10),
                "min_score": self.config.get("round_1_min_score", 60.0),
                "scoring_weights": {
                    "semantic_similarity": 0.4,
                    "skill_match": 0.3,
                    "experience_match": 0.2,
                    "keyword_bonus": 0.1
                }
            },
            "round_2": {
                "name": "Deep Analysis", 
                "target_candidates": self.config.get("round_2_target", 5),
                "min_score": self.config.get("round_2_min_score", 70.0),
                "scoring_weights": {
                    "semantic_similarity": 0.3,
                    "skill_match": 0.25,
                    "experience_match": 0.25,
                    "education_match": 0.1,
                    "keyword_bonus": 0.1
                }
            },
            "round_3": {
                "name": "Final Recommendation",
                "target_candidates": self.config.get("round_3_target", 3),
                "min_score": self.config.get("round_3_min_score", 75.0),
                "scoring_weights": {
                    "semantic_similarity": 0.25,
                    "skill_match": 0.3,
                    "experience_match": 0.25,
                    "education_match": 0.1,
                    "keyword_bonus": 0.1
                }
            }
        }
        
        logger.info("MultiRoundScreeningPipeline initialized")
    
    def execute_screening_round(
        self,
        state: MatchingAgentState,
        round_number: int
    ) -> Dict[str, Any]:
        """
        Execute a single screening round with progressive filtering.
        
        Args:
            state: Current agent state with candidate data
            round_number: Round number (1, 2, or 3)
            
        Returns:
            State updates with screening results
        """
        try:
            round_key = f"round_{round_number}"
            round_config = self.screening_config.get(round_key)
            
            if not round_config:
                raise ValueError(f"Invalid screening round: {round_number}")
            
            logger.info(f"Executing {round_config['name']} (Round {round_number})")
            
            # Get input candidates
            if round_number == 1:
                # First round: use all candidates from initial matching
                input_candidates = state.get("candidate_matches", [])
            else:
                # Subsequent rounds: use candidates from previous round
                previous_screening = self._get_previous_screening(state, round_number - 1)
                input_candidates = previous_screening.get("candidates", []) if previous_screening else []
            
            if not input_candidates:
                return {
                    "error": f"No candidates available for Round {round_number}",
                    "screening_round": round_number
                }
            
            # Execute round-specific screening logic
            if round_number == 1:
                screening_result = self._execute_initial_screen(input_candidates, round_config, state)
            elif round_number == 2:
                screening_result = self._execute_deep_analysis(input_candidates, round_config, state)
            elif round_number == 3:
                screening_result = self._execute_final_recommendation(input_candidates, round_config, state)
            else:
                raise ValueError(f"Unsupported screening round: {round_number}")
            
            # Create screening round data
            screening_data = ScreeningRoundData(
                round_number=round_number,
                candidates=screening_result["selected_candidates"],
                threshold=round_config["min_score"],
                criteria=round_config,
                decision=screening_result.get("decision"),
                reasoning=screening_result.get("reasoning", f"Completed {round_config['name']}")
            )
            
            # Return state updates
            updates = {
                "screening_round": round_number,
                "current_screening_data": screening_data,
                "screening_history": [screening_data],
                "workflow_phase": "screening",
                "performance_metrics": {
                    **state.get("performance_metrics", {}),
                    f"round_{round_number}_processing_time": screening_result.get("processing_time", 0),
                    f"round_{round_number}_candidates": len(screening_result["selected_candidates"])
                }
            }
            
            # Add final response for last round
            if round_number == 3:
                updates["final_response"] = screening_result.get("final_response")
                updates["final_recommendations"] = screening_result.get("recommendations")
                updates["workflow_phase"] = "done"
            
            logger.info(f"Round {round_number} completed: {len(screening_result['selected_candidates'])} candidates selected")
            return updates
            
        except Exception as e:
            logger.error(f"Screening round {round_number} failed: {e}")
            return {
                "error": str(e),
                "screening_round": round_number,
                "workflow_phase": "error"
            }
    
    def _execute_initial_screen(
        self,
        candidates: List[CandidateMatchDict],
        round_config: Dict[str, Any],
        state: MatchingAgentState
    ) -> Dict[str, Any]:
        """Execute the initial screening round."""
        start_time = time.time()
        
        # Filter candidates by minimum score
        min_score = round_config["min_score"]
        qualified_candidates = [
            candidate for candidate in candidates 
            if candidate["match_score"] >= min_score
        ]
        
        # Sort by match score
        qualified_candidates.sort(key=lambda x: x["match_score"], reverse=True)
        
        # Take top N candidates
        target_count = round_config["target_candidates"]
        selected_candidates = qualified_candidates[:target_count]
        
        # Calculate statistics
        if selected_candidates:
            avg_score = statistics.mean(c["match_score"] for c in selected_candidates)
            score_range = (
                min(c["match_score"] for c in selected_candidates),
                max(c["match_score"] for c in selected_candidates)
            )
        else:
            avg_score = 0
            score_range = (0, 0)
        
        reasoning = f"""Initial screening completed:
- {len(candidates)} candidates evaluated
- {len(qualified_candidates)} met minimum score threshold ({min_score}%)
- {len(selected_candidates)} candidates selected for deep analysis
- Average score of selected candidates: {avg_score:.1f}%
- Score range: {score_range[0]:.1f}% - {score_range[1]:.1f}%"""
        
        processing_time = time.time() - start_time
        
        return {
            "selected_candidates": selected_candidates,
            "processing_time": processing_time,
            "reasoning": reasoning,
            "statistics": {
                "total_evaluated": len(candidates),
                "qualified": len(qualified_candidates),
                "selected": len(selected_candidates),
                "avg_score": avg_score,
                "score_range": score_range
            }
        }
    
    def _execute_deep_analysis(
        self,
        candidates: List[CandidateMatchDict],
        round_config: Dict[str, Any],
        state: MatchingAgentState
    ) -> Dict[str, Any]:
        """Execute the deep analysis screening round."""
        start_time = time.time()
        
        # Perform detailed analysis for each candidate
        analyzed_candidates = []
        job_requirements = state.get("parsed_requirements")
        
        for candidate in candidates:
            try:
                # Get detailed candidate profile
                profile = analyze_candidate_profile.invoke({
                    "candidate_name": candidate["candidate_name"],
                    "job_requirements": job_requirements
                })
                
                if "error" not in profile:
                    # Enhanced scoring based on deep analysis
                    enhanced_score = self._calculate_enhanced_score(
                        candidate,
                        profile,
                        job_requirements,
                        round_config["scoring_weights"]
                    )
                    
                    # Create enhanced candidate record
                    enhanced_candidate = {
                        **candidate,
                        "enhanced_score": enhanced_score,
                        "profile_analysis": profile,
                        "deep_analysis": {
                            "content_depth": profile.get("sections_found", 0),
                            "skill_diversity": len(profile.get("skills", [])),
                            "experience_alignment": self._assess_experience_alignment(
                                profile, job_requirements
                            ),
                            "technical_focus": self._assess_technical_focus(profile)
                        }
                    }
                    
                    analyzed_candidates.append(enhanced_candidate)
                else:
                    logger.warning(f"Failed to analyze {candidate['candidate_name']}: {profile['error']}")
                    # Keep original candidate if analysis fails
                    analyzed_candidates.append(candidate)
                    
            except Exception as e:
                logger.error(f"Deep analysis failed for {candidate['candidate_name']}: {e}")
                # Keep original candidate if analysis fails
                analyzed_candidates.append(candidate)
        
        # Sort by enhanced score or original match score
        analyzed_candidates.sort(
            key=lambda x: x.get("enhanced_score", x["match_score"]), 
            reverse=True
        )
        
        # Apply minimum score threshold for enhanced scoring
        min_score = round_config["min_score"]
        qualified_candidates = [
            candidate for candidate in analyzed_candidates
            if candidate.get("enhanced_score", candidate["match_score"]) >= min_score
        ]
        
        # Select top candidates
        target_count = round_config["target_candidates"]
        selected_candidates = qualified_candidates[:target_count]
        
        # Generate comparison insights
        comparison_insights = ""
        if len(selected_candidates) > 1:
            try:
                candidate_names = [c["candidate_name"] for c in selected_candidates[:3]]
                comparison = compare_candidates.invoke({
                    "candidate_ids": candidate_names,
                    "job_requirements": job_requirements
                })
                
                if "error" not in comparison:
                    top_candidate = comparison["recommendation"]["top_candidate"]
                    comparison_insights = f"\nComparative Analysis:\n- Top candidate: {top_candidate}\n- Recommendation: {comparison['recommendation']['recommendation_reason']}"
                    
            except Exception as e:
                logger.warning(f"Comparison analysis failed: {e}")
        
        # Calculate statistics
        if selected_candidates:
            enhanced_scores = [c.get("enhanced_score", c["match_score"]) for c in selected_candidates]
            avg_enhanced_score = statistics.mean(enhanced_scores)
            enhanced_range = (min(enhanced_scores), max(enhanced_scores))
        else:
            avg_enhanced_score = 0
            enhanced_range = (0, 0)
        
        reasoning = f"""Deep analysis completed:
- {len(analyzed_candidates)} candidates analyzed in detail
- {len(qualified_candidates)} met enhanced scoring threshold ({min_score}%)
- {len(selected_candidates)} finalists selected
- Average enhanced score: {avg_enhanced_score:.1f}%
- Enhanced score range: {enhanced_range[0]:.1f}% - {enhanced_range[1]:.1f}%{comparison_insights}"""
        
        processing_time = time.time() - start_time
        
        return {
            "selected_candidates": selected_candidates,
            "processing_time": processing_time,
            "reasoning": reasoning,
            "statistics": {
                "analyzed": len(analyzed_candidates),
                "qualified": len(qualified_candidates),
                "selected": len(selected_candidates),
                "avg_enhanced_score": avg_enhanced_score,
                "enhanced_range": enhanced_range
            }
        }
    
    def _execute_final_recommendation(
        self,
        candidates: List[CandidateMatchDict],
        round_config: Dict[str, Any],
        state: MatchingAgentState
    ) -> Dict[str, Any]:
        """Execute the final recommendation round."""
        start_time = time.time()
        
        job_requirements = state.get("parsed_requirements")
        job_title = state.get("job_title", "Position")
        
        # Generate hire/no-hire recommendations for each candidate
        recommendations = []
        
        for i, candidate in enumerate(candidates):
            try:
                # Generate detailed recommendation
                recommendation = self._generate_hire_recommendation(
                    candidate,
                    job_requirements,
                    rank=i + 1,
                    total_candidates=len(candidates)
                )
                
                recommendations.append(recommendation)
                
                # Generate interview questions for top candidates
                if i < 2:  # Top 2 candidates
                    try:
                        questions = generate_interview_questions.invoke({
                            "candidate_id": candidate["candidate_name"],
                            "job_requirements": job_requirements,
                            "num_questions": 8
                        })
                        
                        if "error" not in questions:
                            recommendation["interview_questions"] = questions["questions"][:5]  # Top 5 questions
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate interview questions for {candidate['candidate_name']}: {e}")
                
            except Exception as e:
                logger.error(f"Failed to generate recommendation for {candidate['candidate_name']}: {e}")
                # Create basic recommendation
                recommendations.append({
                    "candidate_name": candidate["candidate_name"],
                    "recommendation": "REVIEW",
                    "confidence": "LOW",
                    "reasoning": f"Analysis failed: {str(e)}"
                })
        
        # Generate overall hiring recommendation
        hire_candidates = [r for r in recommendations if r.get("recommendation") == "HIRE"]
        strong_candidates = [r for r in recommendations if r.get("confidence") in ["HIGH", "VERY_HIGH"]]
        
        # Create final response
        final_response = self._generate_final_response(
            recommendations,
            job_title,
            job_requirements
        )
        
        reasoning = f"""Final recommendation analysis completed:
- {len(candidates)} finalists evaluated
- {len(hire_candidates)} candidates recommended for hire
- {len(strong_candidates)} candidates with high confidence scores
- Detailed hire/no-hire analysis provided for each candidate"""
        
        processing_time = time.time() - start_time
        
        return {
            "selected_candidates": candidates,  # Keep all finalists
            "processing_time": processing_time,
            "reasoning": reasoning,
            "decision": "RECOMMENDATIONS_GENERATED",
            "recommendations": recommendations,
            "final_response": final_response,
            "statistics": {
                "finalists": len(candidates),
                "hire_recommendations": len(hire_candidates),
                "high_confidence": len(strong_candidates)
            }
        }
    
    def _calculate_enhanced_score(
        self,
        candidate: CandidateMatchDict,
        profile: Dict[str, Any],
        job_requirements: Optional[JobRequirementDict],
        scoring_weights: Dict[str, float]
    ) -> float:
        """Calculate enhanced score based on deep analysis."""
        try:
            base_score = candidate["match_score"]
            
            # Content depth bonus (0-10 points)
            content_depth = profile.get("sections_found", 0)
            depth_bonus = min(10, content_depth * 2)
            
            # Skill diversity bonus (0-5 points)
            skill_count = len(profile.get("skills", []))
            diversity_bonus = min(5, skill_count * 0.5)
            
            # Job fit bonus (0-15 points)
            job_fit_bonus = 0
            if job_requirements and profile.get("job_fit"):
                job_fit = profile["job_fit"]
                skill_match_pct = job_fit.get("skill_match_percentage", 0)
                job_fit_bonus = skill_match_pct * 0.15  # Up to 15 points for 100% skill match
            
            # Experience alignment bonus (0-10 points)
            experience_bonus = self._assess_experience_alignment(profile, job_requirements)
            
            # Calculate enhanced score
            enhanced_score = base_score + depth_bonus + diversity_bonus + job_fit_bonus + experience_bonus
            
            # Cap at 100
            return min(100, enhanced_score)
            
        except Exception as e:
            logger.error(f"Enhanced score calculation failed: {e}")
            return candidate["match_score"]
    
    def _assess_experience_alignment(
        self,
        profile: Dict[str, Any],
        job_requirements: Optional[JobRequirementDict]
    ) -> float:
        """Assess experience alignment and return bonus points (0-10)."""
        if not job_requirements:
            return 0
        
        candidate_exp = profile.get("experience_years", 0) or 0
        required_exp = job_requirements.get("experience_years", 0) or 0
        
        if required_exp == 0:
            return 5  # Default bonus if no requirement specified
        
        if candidate_exp >= required_exp * 1.5:
            return 10  # Significantly exceeds requirement
        elif candidate_exp >= required_exp:
            return 8   # Meets requirement
        elif candidate_exp >= required_exp * 0.8:
            return 5   # Close to requirement
        else:
            return 2   # Below requirement
    
    def _assess_technical_focus(self, profile: Dict[str, Any]) -> float:
        """Assess technical focus based on programming languages (0-1 scale)."""
        prog_languages = profile.get("programming_languages", [])
        total_skills = len(profile.get("skills", []))
        
        if total_skills == 0:
            return 0
        
        return len(prog_languages) / total_skills
    
    def _generate_hire_recommendation(
        self,
        candidate: CandidateMatchDict,
        job_requirements: Optional[JobRequirementDict],
        rank: int,
        total_candidates: int
    ) -> Dict[str, Any]:
        """Generate hire/no-hire recommendation for a candidate."""
        try:
            candidate_name = candidate["candidate_name"]
            match_score = candidate.get("enhanced_score", candidate["match_score"])
            matched_skills = candidate.get("matched_skills", [])
            
            # Determine recommendation based on score and rank
            if match_score >= 85 and rank <= 2:
                recommendation = "HIRE"
                confidence = "VERY_HIGH"
            elif match_score >= 75 and rank <= 3:
                recommendation = "HIRE"
                confidence = "HIGH"
            elif match_score >= 65:
                recommendation = "INTERVIEW"
                confidence = "MEDIUM"
            elif match_score >= 55:
                recommendation = "REVIEW"
                confidence = "LOW"
            else:
                recommendation = "NO_HIRE"
                confidence = "LOW"
            
            # Generate reasoning
            reasoning_parts = []
            
            # Score-based reasoning
            if match_score >= 85:
                reasoning_parts.append(f"Excellent match with {match_score:.1f}% compatibility")
            elif match_score >= 75:
                reasoning_parts.append(f"Strong match with {match_score:.1f}% compatibility")
            elif match_score >= 65:
                reasoning_parts.append(f"Good match with {match_score:.1f}% compatibility")
            else:
                reasoning_parts.append(f"Moderate match with {match_score:.1f}% compatibility")
            
            # Skills-based reasoning
            if matched_skills:
                if len(matched_skills) >= 5:
                    reasoning_parts.append(f"Strong skill alignment with {len(matched_skills)} matching competencies")
                elif len(matched_skills) >= 3:
                    reasoning_parts.append(f"Good skill match with {len(matched_skills)} relevant skills")
                else:
                    reasoning_parts.append(f"Limited skill overlap with {len(matched_skills)} matching skills")
            
            # Ranking-based reasoning
            if rank == 1:
                reasoning_parts.append("Top candidate among all applicants")
            elif rank <= 2:
                reasoning_parts.append("Among top candidates")
            elif rank <= total_candidates // 2:
                reasoning_parts.append("Above-average candidate")
            
            reasoning = ". ".join(reasoning_parts) + "."
            
            return {
                "candidate_name": candidate_name,
                "recommendation": recommendation,
                "confidence": confidence,
                "match_score": match_score,
                "rank": rank,
                "reasoning": reasoning,
                "key_strengths": matched_skills[:5],  # Top 5 strengths
                "next_steps": self._get_next_steps(recommendation, candidate_name)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate recommendation for {candidate.get('candidate_name', 'unknown')}: {e}")
            return {
                "candidate_name": candidate.get("candidate_name", "Unknown"),
                "recommendation": "REVIEW",
                "confidence": "LOW",
                "reasoning": f"Analysis error: {str(e)}"
            }
    
    def _get_next_steps(self, recommendation: str, candidate_name: str) -> List[str]:
        """Get recommended next steps based on hiring recommendation."""
        next_steps = []
        
        if recommendation == "HIRE":
            next_steps = [
                f"Schedule final interview with {candidate_name}",
                "Conduct reference checks",
                "Prepare offer package",
                "Complete background verification"
            ]
        elif recommendation == "INTERVIEW":
            next_steps = [
                f"Schedule technical interview with {candidate_name}",
                "Assess cultural fit",
                "Verify technical competencies",
                "Compare with other candidates"
            ]
        elif recommendation == "REVIEW":
            next_steps = [
                f"Conduct phone screening with {candidate_name}",
                "Clarify experience and skills",
                "Assess interest and availability",
                "Consider for future positions"
            ]
        else:  # NO_HIRE
            next_steps = [
                f"Send polite rejection to {candidate_name}",
                "Provide constructive feedback if requested",
                "Add to talent pool for future opportunities"
            ]
        
        return next_steps
    
    def _generate_final_response(
        self,
        recommendations: List[Dict[str, Any]],
        job_title: str,
        job_requirements: Optional[JobRequirementDict]
    ) -> str:
        """Generate comprehensive final response with all recommendations."""
        try:
            # Count recommendations by type
            hire_count = len([r for r in recommendations if r.get("recommendation") == "HIRE"])
            interview_count = len([r for r in recommendations if r.get("recommendation") == "INTERVIEW"])
            review_count = len([r for r in recommendations if r.get("recommendation") == "REVIEW"])
            
            response_parts = []
            
            # Executive summary
            response_parts.append(f"🎯 **Final Hiring Recommendations for {job_title}**")
            response_parts.append("=" * 60)
            
            response_parts.append(f"**Summary:** {len(recommendations)} finalists evaluated")
            response_parts.append(f"• **HIRE**: {hire_count} candidates")
            response_parts.append(f"• **INTERVIEW**: {interview_count} candidates") 
            response_parts.append(f"• **REVIEW**: {review_count} candidates")
            
            # Top recommendations
            response_parts.append("\n📋 **Detailed Recommendations:**")
            
            for i, rec in enumerate(recommendations, 1):
                candidate_name = rec["candidate_name"]
                recommendation = rec["recommendation"]
                confidence = rec.get("confidence", "MEDIUM")
                match_score = rec.get("match_score", 0)
                reasoning = rec.get("reasoning", "")
                
                response_parts.append(f"\n**{i}. {candidate_name}**")
                response_parts.append(f"   • **Decision**: {recommendation} (Confidence: {confidence})")
                response_parts.append(f"   • **Match Score**: {match_score:.1f}%")
                response_parts.append(f"   • **Analysis**: {reasoning}")
                
                # Add key strengths
                strengths = rec.get("key_strengths", [])
                if strengths:
                    response_parts.append(f"   • **Key Strengths**: {', '.join(strengths[:3])}")
                
                # Add next steps
                next_steps = rec.get("next_steps", [])
                if next_steps:
                    response_parts.append(f"   • **Next Steps**: {next_steps[0]}")
            
            # Overall recommendation
            response_parts.append(f"\n🎖️ **Overall Recommendation:**")
            
            if hire_count >= 2:
                response_parts.append(f"**Proceed with hiring process** - {hire_count} strong candidates identified")
            elif hire_count == 1:
                response_parts.append("**Proceed with top candidate** - One strong hire identified")
            elif interview_count > 0:
                response_parts.append(f"**Continue interview process** - {interview_count} candidates need further evaluation")
            else:
                response_parts.append("**Consider expanding candidate pool** - No strong hires identified")
            
            # Add recommendations for hiring manager
            response_parts.append(f"\n💡 **Recommendations:**")
            
            if hire_count > 0:
                top_hire = next((r for r in recommendations if r.get("recommendation") == "HIRE"), None)
                if top_hire:
                    response_parts.append(f"1. Prioritize {top_hire['candidate_name']} for immediate offer")
            
            if interview_count > 0:
                response_parts.append("2. Schedule technical interviews for INTERVIEW candidates")
            
            if hire_count == 0:
                response_parts.append("3. Consider adjusting requirements or sourcing additional candidates")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Final response generation failed: {e}")
            return f"Final recommendations completed with {len(recommendations)} candidates evaluated. Please review detailed analysis above."
    
    def _get_previous_screening(self, state: MatchingAgentState, round_number: int) -> Optional[ScreeningRoundData]:
        """Get screening data from a previous round."""
        screening_history = state.get("screening_history", [])
        
        for screening in screening_history:
            if screening.get("round_number") == round_number:
                return screening
        
        return None