"""
LangGraph Matching Agent Demonstration
=====================================

Interactive demonstration of the conversational job matching agent capabilities.
Showcases all major features including natural language processing, multi-round
screening, candidate comparison, and human feedback loops.

Usage:
    python demo_matching_agent.py --interactive
    python demo_matching_agent.py --examples
    python demo_matching_agent.py --benchmark
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List
from datetime import datetime

# Import the matching agent
from matching_agent import LangGraphMatchingAgent, InteractiveChatSession


class MatchingAgentDemo:
    """
    Comprehensive demonstration of the LangGraph matching agent capabilities.
    
    Provides scripted examples, interactive sessions, and performance benchmarks
    to showcase the agent's conversational AI and job matching features.
    """
    
    def __init__(self, resume_dir: str = "examples/resumes"):
        """
        Initialize the demonstration with the matching agent.
        
        Args:
            resume_dir: Directory containing resume files for matching
        """
        self.resume_dir = resume_dir
        self.agent = None
        self.demo_queries = [
            "Find candidates for a Senior Python Developer with machine learning experience",
            "Compare John Doe and Sarah Jackson side by side",
            "Generate interview questions for the top candidate", 
            "Why did this candidate rank higher than the others?",
            "List all resume files in the examples directory",
            "Read the job description from examples/job_descriptions/senior_python_developer.txt and find matches"
        ]
    
    def initialize_agent(self) -> None:
        """Initialize the matching agent with resume data."""
        print("🚀 Initializing LangGraph Matching Agent...")
        print("-" * 50)
        
        try:
            # Initialize agent
            config = {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "embedding_model": "openai",
                "max_iterations": 10,
                "top_k": 10,
                "min_score": 0.3
            }
            
            self.agent = LangGraphMatchingAgent(config=config)
            print("✅ Agent initialized successfully")
            
            # Initialize with resumes if directory exists
            if os.path.exists(self.resume_dir):
                print(f"\n📂 Processing resumes from {self.resume_dir}...")
                init_result = self.agent.initialize_with_resumes(self.resume_dir)
                
                if init_result["success"]:
                    print(f"✅ Processed {init_result['processed_count']} resumes")
                    print(f"⏱️ Processing time: {init_result['processing_time']:.2f} seconds")
                    print(f"📊 Average time per resume: {init_result['avg_time_per_resume']:.3f} seconds")
                    
                    # Show sample resume data
                    if init_result["resumes"]:
                        print(f"\n📋 Sample processed resumes:")
                        for resume in init_result["resumes"][:3]:
                            print(f"   • {resume['name']} - {resume['skills_count']} skills, {resume['experience_years']} years exp")
                else:
                    print(f"❌ Resume processing failed: {init_result['error']}")
                    return False
            else:
                print(f"⚠️ Resume directory not found: {self.resume_dir}")
            
            # Show system stats
            stats = self.agent.get_system_stats()
            print(f"\n📊 System Statistics:")
            print(f"   Status: {stats['status']}")
            print(f"   RAG Database: {stats.get('rag_system', {}).get('total_chunks', 0)} chunks")
            print(f"   Available Tools: {stats.get('tools_available', 0)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            return False
    
    def run_example_queries(self) -> None:
        """Run scripted example queries to demonstrate capabilities."""
        if not self.agent:
            print("❌ Agent not initialized. Run initialize_agent() first.")
            return
        
        print("\n🧪 Running Example Query Demonstrations")
        print("=" * 50)
        
        for i, query in enumerate(self.demo_queries, 1):
            print(f"\n📝 Example {i}: {query}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                response = self.agent.process_query(query)
                processing_time = time.time() - start_time
                
                print(f"🤖 Response: {response.final_response}")
                
                # Show candidate matches if available
                if response.candidate_matches:
                    print(f"\n📊 Found {len(response.candidate_matches)} candidate matches:")
                    for j, match in enumerate(response.candidate_matches[:3], 1):
                        print(f"   {j}. {match['candidate_name']} - {match['match_score']}% match")
                        skills = match.get('matched_skills', [])
                        if skills:
                            print(f"      Key Skills: {', '.join(skills[:3])}")
                
                # Show performance info
                print(f"⏱️ Processing time: {processing_time:.3f} seconds")
                
                if response.error:
                    print(f"⚠️ Error: {response.error}")
                
            except Exception as e:
                print(f"❌ Query failed: {e}")
            
            # Pause between examples
            if i < len(self.demo_queries):
                time.sleep(1)
    
    def run_interactive_session(self) -> None:
        """Start an interactive chat session with the agent."""
        if not self.agent:
            print("❌ Agent not initialized. Run initialize_agent() first.")
            return
        
        print("\n💬 Starting Interactive Chat Session")
        print("=" * 50)
        
        chat_session = InteractiveChatSession(self.agent)
        chat_session.start()
    
    def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmark tests."""
        if not self.agent:
            print("❌ Agent not initialized. Run initialize_agent() first.")
            return {}
        
        print("\n⚡ Running Performance Benchmarks")
        print("=" * 50)
        
        benchmark_queries = [
            "Find Python developers",
            "Find machine learning engineers", 
            "Find full stack developers",
            "Compare top 3 candidates",
            "Generate questions for John Doe"
        ]
        
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "queries": [],
            "summary": {}
        }
        
        total_time = 0
        successful_queries = 0
        
        for i, query in enumerate(benchmark_queries, 1):
            print(f"\n🧪 Benchmark {i}: {query}")
            
            try:
                start_time = time.time()
                response = self.agent.process_query(query)
                processing_time = time.time() - start_time
                
                total_time += processing_time
                
                query_result = {
                    "query": query,
                    "processing_time": processing_time,
                    "success": response.error is None,
                    "candidates_found": len(response.candidate_matches),
                    "response_length": len(response.final_response) if response.final_response else 0
                }
                
                benchmark_results["queries"].append(query_result)
                
                if query_result["success"]:
                    successful_queries += 1
                    print(f"   ✅ Success in {processing_time:.3f}s")
                    print(f"   📊 {query_result['candidates_found']} candidates found")
                else:
                    print(f"   ❌ Failed: {response.error}")
                
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                query_result = {
                    "query": query,
                    "processing_time": 0,
                    "success": False,
                    "error": str(e)
                }
                benchmark_results["queries"].append(query_result)
        
        # Calculate summary metrics
        if successful_queries > 0:
            avg_processing_time = total_time / successful_queries
            success_rate = successful_queries / len(benchmark_queries) * 100
        else:
            avg_processing_time = 0
            success_rate = 0
        
        benchmark_results["summary"] = {
            "total_queries": len(benchmark_queries),
            "successful_queries": successful_queries,
            "success_rate": success_rate,
            "total_time": total_time,
            "avg_processing_time": avg_processing_time,
            "queries_per_second": successful_queries / total_time if total_time > 0 else 0
        }
        
        # Print summary
        print(f"\n📊 Benchmark Summary:")
        print(f"   Total Queries: {benchmark_results['summary']['total_queries']}")
        print(f"   Success Rate: {benchmark_results['summary']['success_rate']:.1f}%")
        print(f"   Average Processing Time: {benchmark_results['summary']['avg_processing_time']:.3f}s")
        print(f"   Throughput: {benchmark_results['summary']['queries_per_second']:.2f} queries/second")
        
        # Performance assessment
        if avg_processing_time < 1.0 and success_rate >= 90:
            print("   🟢 Performance: EXCELLENT")
        elif avg_processing_time < 2.0 and success_rate >= 80:
            print("   🟡 Performance: GOOD") 
        else:
            print("   🔴 Performance: NEEDS IMPROVEMENT")
        
        return benchmark_results
    
    def run_feature_showcase(self) -> None:
        """Showcase specific agent features with detailed examples."""
        if not self.agent:
            print("❌ Agent not initialized. Run initialize_agent() first.")
            return
        
        print("\n🌟 Agent Feature Showcase")
        print("=" * 50)
        
        features = [
            {
                "name": "Natural Language Job Matching",
                "query": "I need a senior developer with at least 5 years of Python experience who knows machine learning",
                "description": "Demonstrates natural language understanding and requirement extraction"
            },
            {
                "name": "Candidate Comparison",
                "query": "Compare John Doe and Sarah Jackson for technical skills and experience",
                "description": "Shows head-to-head candidate analysis with detailed breakdown"
            },
            {
                "name": "Interview Question Generation", 
                "query": "Generate 5 technical interview questions for Linda Kim based on her background",
                "description": "Creates tailored interview questions based on candidate profile"
            },
            {
                "name": "File System Integration",
                "query": "List all job description files and then read the Python developer one",
                "description": "Demonstrates file operations and multi-step workflow execution"
            },
            {
                "name": "Explainable Rankings",
                "query": "Why did the top candidate rank higher than the second candidate?",
                "description": "Provides detailed explanations of ranking decisions and scoring factors"
            }
        ]
        
        for i, feature in enumerate(features, 1):
            print(f"\n🎯 Feature {i}: {feature['name']}")
            print(f"📝 Description: {feature['description']}")
            print(f"💬 Query: {feature['query']}")
            print("-" * 40)
            
            try:
                response = self.agent.process_query(feature['query'])
                print(f"🤖 Response: {response.final_response[:300]}...")
                
                if response.candidate_matches:
                    print(f"📊 {len(response.candidate_matches)} candidates involved")
                
            except Exception as e:
                print(f"❌ Feature demo failed: {e}")
            
            print()
    
    def save_demo_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save demonstration results to a file."""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Demo results saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


def main():
    """Main demonstration entry point."""
    parser = argparse.ArgumentParser(description="LangGraph Matching Agent Demonstration")
    parser.add_argument("--resume-dir", default="examples/resumes", help="Resume directory path")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat session")
    parser.add_argument("--examples", action="store_true", help="Run example query demonstrations")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--showcase", action="store_true", help="Run feature showcase")
    parser.add_argument("--all", action="store_true", help="Run all demonstrations")
    parser.add_argument("--save-results", help="Save results to specified file")
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = MatchingAgentDemo(resume_dir=args.resume_dir)
    
    # Initialize agent
    if not demo.initialize_agent():
        print("❌ Demo initialization failed. Exiting.")
        return 1
    
    demo_results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "resume_dir": args.resume_dir,
            "agent_config": demo.agent.config if demo.agent else {}
        },
        "results": {}
    }
    
    try:
        # Run requested demonstrations
        if args.all or args.examples:
            print("\n" + "="*60)
            demo.run_example_queries()
        
        if args.all or args.showcase:
            print("\n" + "="*60)
            demo.run_feature_showcase()
        
        if args.all or args.benchmark:
            print("\n" + "="*60)
            benchmark_results = demo.run_performance_benchmark()
            demo_results["results"]["benchmark"] = benchmark_results
        
        if args.interactive:
            print("\n" + "="*60)
            demo.run_interactive_session()
        
        # Default to example queries if no specific option chosen
        if not any([args.interactive, args.examples, args.benchmark, args.showcase, args.all]):
            print("\n" + "="*60)
            demo.run_example_queries()
        
        # Save results if requested
        if args.save_results:
            demo.save_demo_results(demo_results, args.save_results)
        
        print("\n🎉 LangGraph Matching Agent Demonstration Complete!")
        print("   The agent successfully demonstrated:")
        print("   ✅ Natural language job matching")
        print("   ✅ Multi-round screening capabilities")
        print("   ✅ Human-in-the-loop feedback mechanisms")
        print("   ✅ Explainable AI recommendations")
        print("   ✅ Conversational interface")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Demonstration interrupted. Thank you!")
        return 0
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())