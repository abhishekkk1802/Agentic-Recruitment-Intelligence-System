"""
Comprehensive Test Suite for LangGraph Matching Agent
====================================================

Tests for all components of the conversational job matching agent including:
- Agent state management and workflow execution
- Tool functionality and integration
- Multi-round screening pipeline
- Human-in-the-loop feedback mechanisms
- Performance validation and benchmarking
"""

import unittest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, List, Any

# Import components to test
from agent_state import (
    MatchingAgentState,
    create_initial_state,
    update_workflow_phase,
    is_circuit_breaker_triggered,
    extract_user_message_content
)
from agent_tools import (
    initialize_tools,
    extract_requirements,
    compare_candidates,
    generate_interview_questions,
    get_tools_by_category
)
from screening_pipeline import MultiRoundScreeningPipeline
from matching_agent import LangGraphMatchingAgent, InteractiveChatSession

# Import existing components
from resume_rag import ResumeRAG
from job_matcher import JobMatcher
from langchain_core.messages import HumanMessage, AIMessage


class TestAgentState(unittest.TestCase):
    """Test agent state management and helper functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_messages = [
            HumanMessage(content="Find candidates for Python developer"),
            AIMessage(content="I'll help you find Python developers.")
        ]
        self.thread_id = "test_thread_123"
    
    def test_create_initial_state(self):
        """Test initial state creation with default values."""
        state = create_initial_state(
            messages=self.test_messages,
            thread_id=self.thread_id
        )
        
        self.assertEqual(state["thread_id"], self.thread_id)
        self.assertEqual(len(state["messages"]), 2)
        self.assertEqual(state["workflow_phase"], "classify")
        self.assertEqual(state["iteration_count"], 0)
        self.assertEqual(state["top_k"], 10)
        self.assertFalse(state["awaiting_human"])
    
    def test_update_workflow_phase(self):
        """Test workflow phase updates."""
        state = create_initial_state(self.test_messages, self.thread_id)
        
        updates = update_workflow_phase(state, "parse_job", iteration_increment=True)
        
        self.assertEqual(updates["workflow_phase"], "parse_job")
        self.assertEqual(updates["iteration_count"], 1)
        self.assertIn("Phase: parse_job", updates["workflow_history"])
    
    def test_circuit_breaker_triggered(self):
        """Test circuit breaker logic."""
        # Test normal operation
        state = create_initial_state(self.test_messages, self.thread_id)
        self.assertFalse(is_circuit_breaker_triggered(state))
        
        # Test max iterations reached
        state["iteration_count"] = 15
        state["max_iterations"] = 10
        self.assertTrue(is_circuit_breaker_triggered(state))
        
        # Test max screening rounds reached
        state["iteration_count"] = 5
        state["screening_round"] = 4
        state["max_screening_rounds"] = 3
        self.assertTrue(is_circuit_breaker_triggered(state))
    
    def test_extract_user_message_content(self):
        """Test extraction of user message content."""
        state = create_initial_state(self.test_messages, self.thread_id)
        
        content = extract_user_message_content(state)
        self.assertEqual(content, "Find candidates for Python developer")


class TestAgentTools(unittest.TestCase):
    """Test agent tools and their functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for all tool tests."""
        # Create mock RAG and JobMatcher instances
        cls.mock_rag = Mock(spec=ResumeRAG)
        cls.mock_matcher = Mock(spec=JobMatcher)
        
        # Initialize tools with mocks
        initialize_tools(cls.mock_rag, cls.mock_matcher)
    
    def test_extract_requirements_tool(self):
        """Test enhanced requirement extraction."""
        job_description = """
        Senior Python Developer - TechCorp
        
        REQUIRED:
        - 5+ years Python experience
        - Django or Flask experience
        - Must have PostgreSQL knowledge
        
        PREFERRED:
        - AWS experience nice to have
        - Machine learning background preferred
        """
        
        result = extract_requirements(job_description)
        
        self.assertIn("basic_requirements", result)
        self.assertIn("enhanced_classification", result)
        self.assertIn("priority_analysis", result)
        
        # Check enhanced classification
        enhanced = result["enhanced_classification"]
        self.assertIsInstance(enhanced["must_have_extracted"], list)
        self.assertIsInstance(enhanced["nice_to_have_extracted"], list)
        self.assertGreater(enhanced["min_experience_years"], 0)
    
    def test_compare_candidates_tool(self):
        """Test candidate comparison functionality."""
        candidate_ids = ["John Doe", "Jane Smith"]
        
        # Mock the analyze_candidate_profile function
        with patch('agent_tools.analyze_candidate_profile') as mock_analyze:
            mock_analyze.return_value = {
                "candidate_name": "John Doe",
                "skills": ["Python", "Django", "PostgreSQL"],
                "experience_years": 5,
                "programming_languages": ["Python", "JavaScript"],
                "sections_found": 4
            }
            
            result = compare_candidates(candidate_ids)
            
            self.assertIn("candidates_compared", result)
            self.assertIn("comparison_matrix", result)
            self.assertIn("overall_ranking", result)
            self.assertIn("recommendation", result)
    
    def test_generate_interview_questions_tool(self):
        """Test interview question generation."""
        candidate_id = "John Doe"
        job_requirements = {
            "title": "Senior Python Developer",
            "skills_required": ["Python", "Django"],
            "experience_years": 5
        }
        
        # Mock the analyze_candidate_profile function
        with patch('agent_tools.analyze_candidate_profile') as mock_analyze:
            mock_analyze.return_value = {
                "candidate_name": "John Doe",
                "skills": ["Python", "Django", "PostgreSQL"],
                "experience_years": 5,
                "programming_languages": ["Python"]
            }
            
            result = generate_interview_questions(
                candidate_id,
                job_requirements=job_requirements,
                num_questions=5
            )
            
            self.assertEqual(result["candidate_id"], candidate_id)
            self.assertEqual(result["total_questions"], 5)
            self.assertIsInstance(result["questions"], list)
            self.assertEqual(len(result["questions"]), 5)
    
    def test_get_tools_by_category(self):
        """Test tool categorization."""
        safe_tools = get_tools_by_category("safe")
        write_tools = get_tools_by_category("write")
        all_tools = get_tools_by_category("all")
        
        self.assertIsInstance(safe_tools, list)
        self.assertIsInstance(write_tools, list)
        self.assertIsInstance(all_tools, list)
        
        self.assertGreater(len(safe_tools), 0)
        self.assertGreater(len(write_tools), 0)
        self.assertGreater(len(all_tools), len(safe_tools))


class TestScreeningPipeline(unittest.TestCase):
    """Test multi-round screening pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = MultiRoundScreeningPipeline()
        
        # Sample candidate data
        self.sample_candidates = [
            {
                "candidate_name": "John Doe",
                "match_score": 85,
                "matched_skills": ["Python", "Django", "PostgreSQL"],
                "reasoning": "Strong technical match"
            },
            {
                "candidate_name": "Jane Smith", 
                "match_score": 78,
                "matched_skills": ["Python", "Flask", "MongoDB"],
                "reasoning": "Good technical skills"
            },
            {
                "candidate_name": "Bob Johnson",
                "match_score": 65,
                "matched_skills": ["Python", "FastAPI"],
                "reasoning": "Moderate match"
            }
        ]
        
        # Sample state
        self.sample_state = create_initial_state(
            messages=[HumanMessage(content="Test")],
            thread_id="test_123"
        )
        self.sample_state["candidate_matches"] = self.sample_candidates
    
    def test_initial_screening_round(self):
        """Test initial screening round execution."""
        result = self.pipeline.execute_screening_round(self.sample_state, 1)
        
        self.assertNotIn("error", result)
        self.assertEqual(result["screening_round"], 1)
        self.assertIn("current_screening_data", result)
        
        screening_data = result["current_screening_data"]
        self.assertEqual(screening_data["round_number"], 1)
        self.assertIsInstance(screening_data["candidates"], list)
    
    def test_deep_analysis_round(self):
        """Test deep analysis screening round."""
        # Add screening history for round 1
        self.sample_state["screening_history"] = [{
            "round_number": 1,
            "candidates": self.sample_candidates[:2]  # Top 2 from first round
        }]
        
        with patch('agent_tools.analyze_candidate_profile') as mock_analyze:
            mock_analyze.return_value = {
                "candidate_name": "John Doe",
                "skills": ["Python", "Django"],
                "experience_years": 5,
                "sections_found": 4
            }
            
            result = self.pipeline.execute_screening_round(self.sample_state, 2)
            
            self.assertNotIn("error", result)
            self.assertEqual(result["screening_round"], 2)
    
    def test_final_recommendation_round(self):
        """Test final recommendation round."""
        # Add screening history for round 2
        self.sample_state["screening_history"] = [{
            "round_number": 2,
            "candidates": self.sample_candidates[:2]
        }]
        
        with patch('agent_tools.generate_interview_questions') as mock_questions:
            mock_questions.return_value = {
                "questions": [
                    {"question": "Tell me about your Python experience", "type": "technical"}
                ]
            }
            
            result = self.pipeline.execute_screening_round(self.sample_state, 3)
            
            self.assertNotIn("error", result)
            self.assertEqual(result["screening_round"], 3)
            self.assertEqual(result["workflow_phase"], "done")
            self.assertIn("final_response", result)
            self.assertIn("recommendations", result)


class TestMatchingAgentIntegration(unittest.TestCase):
    """Integration tests for the complete matching agent."""
    
    def setUp(self):
        """Set up test agent with minimal configuration."""
        # Use minimal config for faster testing
        self.test_config = {
            "model": "gpt-4o-mini",
            "max_iterations": 5,
            "embedding_model": "sentence-transformers"  # Avoid OpenAI costs in tests
        }
        
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Mock the expensive initialization
        with patch('matching_agent.ResumeRAG') as mock_rag_class, \
             patch('matching_agent.JobMatcher') as mock_matcher_class:
            
            # Configure mocks
            mock_rag = Mock()
            mock_rag.get_collection_stats.return_value = {"total_chunks": 100}
            mock_rag_class.return_value = mock_rag
            
            mock_matcher = Mock()
            mock_matcher_class.return_value = mock_matcher
            
            # Initialize agent
            self.agent = LangGraphMatchingAgent(
                config=self.test_config,
                checkpoint_db=None  # In-memory for testing
            )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        self.assertIsNotNone(self.agent.rag)
        self.assertIsNotNone(self.agent.matcher)
        self.assertIsNotNone(self.agent.llm)
        self.assertIsNotNone(self.agent.graph)
    
    def test_simple_query_processing(self):
        """Test basic query processing without external dependencies."""
        # Mock the graph invocation
        mock_final_state = {
            "messages": [
                HumanMessage(content="Find Python developers"),
                AIMessage(content="I found 3 Python developers")
            ],
            "final_response": "I found 3 qualified Python developers for your position.",
            "candidate_matches": [
                {
                    "candidate_name": "John Doe",
                    "match_score": 85,
                    "matched_skills": ["Python", "Django"]
                }
            ],
            "workflow_phase": "done"
        }
        
        with patch.object(self.agent.graph, 'invoke', return_value=mock_final_state):
            result = self.agent.process_query("Find Python developers")
            
            self.assertIsNotNone(result.final_response)
            self.assertEqual(len(result.candidate_matches), 1)
            self.assertIsNone(result.error)
    
    def test_conversation_state_management(self):
        """Test conversation state tracking."""
        thread_id = "test_conversation_123"
        
        # Mock state retrieval
        mock_state = create_initial_state(
            messages=[HumanMessage(content="Test")],
            thread_id=thread_id
        )
        
        with patch.object(self.agent.graph, 'get_state') as mock_get_state:
            mock_get_state.return_value = Mock(values=mock_state)
            
            state = self.agent.get_conversation_state(thread_id)
            
            self.assertIsNotNone(state)
            self.assertEqual(state["thread_id"], thread_id)
    
    def test_system_stats_retrieval(self):
        """Test system statistics retrieval."""
        stats = self.agent.get_system_stats()
        
        self.assertIn("rag_system", stats)
        self.assertIn("agent_system", stats)
        self.assertIn("status", stats)
        self.assertEqual(stats["status"], "healthy")


class TestPerformanceValidation(unittest.TestCase):
    """Performance validation and benchmarking tests."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.performance_thresholds = {
            "query_processing_time": 5.0,  # seconds
            "tool_execution_time": 2.0,    # seconds
            "state_update_time": 0.1       # seconds
        }
    
    def test_state_creation_performance(self):
        """Test state creation performance."""
        messages = [HumanMessage(content="Performance test")]
        
        start_time = time.time()
        
        for _ in range(100):  # Create 100 states
            create_initial_state(messages, f"thread_{_}")
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / 100
        
        self.assertLess(avg_time, 0.01)  # Should be very fast
    
    def test_tool_categorization_performance(self):
        """Test tool categorization performance."""
        start_time = time.time()
        
        for _ in range(1000):  # Get tools 1000 times
            get_tools_by_category("safe")
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / 1000
        
        self.assertLess(avg_time, 0.001)  # Should be very fast
    
    def test_workflow_phase_update_performance(self):
        """Test workflow phase update performance."""
        state = create_initial_state([HumanMessage(content="Test")], "thread_123")
        
        start_time = time.time()
        
        for i in range(100):
            update_workflow_phase(state, f"phase_{i}")
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / 100
        
        self.assertLess(avg_time, self.performance_thresholds["state_update_time"])


class TestErrorHandling(unittest.TestCase):
    """Test error handling and recovery mechanisms."""
    
    def test_invalid_tool_input_handling(self):
        """Test handling of invalid tool inputs."""
        # Test extract_requirements with invalid input
        result = extract_requirements("")
        self.assertIn("error", result)
        
        # Test compare_candidates with insufficient candidates
        result = compare_candidates(["only_one_candidate"])
        self.assertIn("error", result)
    
    def test_missing_data_handling(self):
        """Test handling of missing or corrupted data."""
        # Test circuit breaker with malformed state
        malformed_state = {"iteration_count": "not_a_number"}
        
        # Should not raise exception
        try:
            is_circuit_breaker_triggered(malformed_state)
        except Exception as e:
            self.fail(f"Circuit breaker raised exception with malformed state: {e}")
    
    def test_tool_initialization_error_handling(self):
        """Test tool initialization error handling."""
        # Test tool access without initialization
        from agent_tools import get_rag_instance, get_matcher_instance
        
        # Reset global instances to None
        import agent_tools
        agent_tools._rag_instance = None
        agent_tools._matcher_instance = None
        
        with self.assertRaises(RuntimeError):
            get_rag_instance()
        
        with self.assertRaises(RuntimeError):
            get_matcher_instance()


def create_test_suite() -> unittest.TestSuite:
    """Create comprehensive test suite for the matching agent."""
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestAgentState,
        TestAgentTools,
        TestScreeningPipeline,
        TestMatchingAgentIntegration,
        TestPerformanceValidation,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_comprehensive_validation() -> Dict[str, Any]:
    """
    Run comprehensive validation of the matching agent system.
    
    Returns:
        Validation results with performance metrics and test outcomes
    """
    print("🧪 Running Comprehensive Matching Agent Validation")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run test suite
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    total_time = time.time() - start_time
    
    # Compile results
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0,
        "total_time": total_time,
        "avg_time_per_test": total_time / result.testsRun if result.testsRun > 0 else 0,
        "test_categories": {
            "state_management": "TestAgentState",
            "tool_functionality": "TestAgentTools", 
            "screening_pipeline": "TestScreeningPipeline",
            "integration": "TestMatchingAgentIntegration",
            "performance": "TestPerformanceValidation",
            "error_handling": "TestErrorHandling"
        }
    }
    
    # Print summary
    print(f"\n📊 Validation Summary:")
    print(f"   Tests Run: {validation_results['total_tests']}")
    print(f"   Success Rate: {validation_results['success_rate']:.1f}%")
    print(f"   Total Time: {validation_results['total_time']:.2f}s")
    print(f"   Failures: {validation_results['failures']}")
    print(f"   Errors: {validation_results['errors']}")
    
    if validation_results['success_rate'] >= 95:
        print("   ✅ System validation PASSED")
    elif validation_results['success_rate'] >= 80:
        print("   ⚠️ System validation PARTIAL - minor issues detected")
    else:
        print("   ❌ System validation FAILED - significant issues detected")
    
    return validation_results


if __name__ == "__main__":
    # Run comprehensive validation
    results = run_comprehensive_validation()
    
    # Save results to file
    results_file = "matching_agent_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Validation results saved to {results_file}")
    
    # Exit with appropriate code
    if results['success_rate'] >= 95:
        exit(0)
    else:
        exit(1)