"""Smoke test: .invoke() works on real StructuredTool objects for all tool call sites."""
import os
from dotenv import load_dotenv
load_dotenv()

from resume_rag import ResumeRAG
from job_matcher import JobMatcher
from agent_tools import initialize_tools, extract_requirements, semantic_search_resumes, find_job_matches

rag = ResumeRAG(vector_db="chromadb", embedding_model="openai", collection_name="matching_agent_collection")
matcher = JobMatcher(rag_system=rag)
initialize_tools(rag, matcher)

print("1. extract_requirements.invoke()")
r = extract_requirements.invoke({"job_description": "Senior Python Developer. Required: Python, Django. 5+ years. Preferred: AWS."})
print("   type:", type(r).__name__, "| keys:", list(r.keys()) if isinstance(r, dict) else "n/a")
assert isinstance(r, dict), r
assert "basic_requirements" in r or "error" in r, r
print("   OK")

print("\n2. semantic_search_resumes.invoke()")
r = semantic_search_resumes.invoke({"query": "python developer with django", "k": 5})
print("   type:", type(r).__name__, "| len:", len(r) if isinstance(r, list) else "n/a")
print("   OK")

print("\n3. find_job_matches.invoke()")
r = find_job_matches.invoke({
    "job_description": "Python developer with Django and AWS, 5 years experience",
    "job_title": "Senior Python Developer",
    "k": 5,
    "min_score": 30.0
})
print("   type:", type(r).__name__, "| keys:", list(r.keys()) if isinstance(r, dict) else "n/a")
assert isinstance(r, dict), r
print("   OK")

print("\n" + "=" * 50)
print("ALL .invoke() CALLS WORK ON REAL TOOLS")
