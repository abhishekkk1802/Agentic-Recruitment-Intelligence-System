"""Temporary API probe - determines installed LangGraph API surface. Deleted after use."""
import importlib.metadata as md
import inspect

print("=== VERSIONS ===")
for p in ['langgraph', 'langgraph-checkpoint', 'langgraph-checkpoint-sqlite',
          'langchain-core', 'langchain-openai', 'langchain-community',
          'chromadb', 'sentence-transformers', 'openai', 'streamlit', 'gradio']:
    try:
        print(f"{p:34} {md.version(p)}")
    except Exception:
        print(f"{p:34} NOT INSTALLED")

print("\n=== langgraph.types exports ===")
try:
    import langgraph.types as t
    print([x for x in dir(t) if not x.startswith('_')])
except Exception as e:
    print("ERR", e)

print("\n=== interrupt + Command importable? ===")
try:
    from langgraph.types import interrupt, Command
    print("OK: both available")
except Exception as e:
    print("ERR", type(e).__name__, e)

print("\n=== SqliteSaver ===")
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    print("from_conn_string:", inspect.signature(SqliteSaver.from_conn_string))
    doc = (SqliteSaver.from_conn_string.__doc__ or "")[:400]
    print("doc:", " ".join(doc.split()))
    r = SqliteSaver.from_conn_string(":memory:")
    print("returns type:", type(r).__name__)
    print("is contextmanager:", hasattr(r, "__enter__"))
except Exception as e:
    print("ERR", type(e).__name__, e)

print("\n=== MemorySaver ===")
try:
    from langgraph.checkpoint.memory import MemorySaver
    ms = MemorySaver()
    print("OK:", type(ms).__name__)
except Exception as e:
    print("ERR", type(e).__name__, e)

print("\n=== draw methods on graph object ===")
try:
    import langchain_core.runnables.graph as g
    print([x for x in dir(g.Graph) if 'draw' in x])
except Exception as e:
    print("ERR", e)

print("\n=== ToolNode / tools_condition ===")
try:
    from langgraph.prebuilt import ToolNode, tools_condition
    print("OK: langgraph.prebuilt has ToolNode + tools_condition")
except Exception as e:
    print("ERR", type(e).__name__, e)

print("\n=== StateGraph.compile signature ===")
try:
    from langgraph.graph import StateGraph
    print(inspect.signature(StateGraph.compile))
except Exception as e:
    print("ERR", e)
