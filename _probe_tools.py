"""Temporary probe - checks whether @tool functions are directly callable. Deleted after use."""
import agent_tools as at

print("=== tool object types ===")
for name in ["extract_requirements", "compare_candidates", "generate_interview_questions",
             "analyze_candidate_profile", "semantic_search_resumes", "find_job_matches",
             "parse_job_description", "read_file_tool"]:
    obj = getattr(at, name, None)
    print(f"{name:32} {type(obj).__name__:20} callable={callable(obj)}")

print("\n=== direct-call test (this is how agent_workflow.py + screening_pipeline.py call them) ===")
try:
    r = at.extract_requirements("We need 5 years Python. Required: Django. Preferred: AWS.")
    print("direct call OK ->", type(r).__name__)
except Exception as e:
    print("direct call FAILED ->", type(e).__name__, str(e)[:200])

print("\n=== .invoke() test ===")
try:
    r = at.extract_requirements.invoke({"job_description": "We need 5 years Python. Required: Django."})
    print(".invoke OK ->", type(r).__name__, "keys:", list(r.keys())[:6] if isinstance(r, dict) else "")
except Exception as e:
    print(".invoke FAILED ->", type(e).__name__, str(e)[:300])

print("\n=== .func attribute (undecorated original) ===")
for name in ["extract_requirements", "analyze_candidate_profile"]:
    obj = getattr(at, name)
    print(f"{name}: has .func = {hasattr(obj, 'func')}")

print("\n=== screening_pipeline import check ===")
try:
    from screening_pipeline import MultiRoundScreeningPipeline
    import inspect
    p = MultiRoundScreeningPipeline
    print("methods:", [m for m in dir(p) if not m.startswith('__')])
    print("execute_screening_round:", inspect.signature(p.execute_screening_round))
except Exception as e:
    print("ERR", type(e).__name__, e)
