"""
MCP Test Scenarios
==================

Six test scenarios demonstrating MCP resource usage and the agent workflow
operating through the MCP server. Each scenario starts the filesystem MCP
server, connects the client, exercises a capability, and verifies the result.

Run all scenarios::

    python mcp_test_scenarios.py

Run a single scenario by index::

    python mcp_test_scenarios.py --scenario 3

Scenarios:
    1. Tool discovery and catalogue
    2. File read via MCP tool
    3. Directory listing and filtering via MCP
    4. watch_directory — detect new files between calls
    5. batch_process — read and search multiple files in one round-trip
    6. Agent end-to-end through MCP — file operations routed via MCP server
"""

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_client import MCPClient

# Resume directory used by the scenarios
RESUME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "resumes")
SAMPLE_RESUME = os.path.join(RESUME_DIR, "john_doe_resume.txt")
SECOND_RESUME = os.path.join(RESUME_DIR, "sarah_smith_resume.txt")


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_result(label: str, value: Any, indent: int = 2) -> None:
    pad = " " * indent
    if isinstance(value, dict):
        print(f"{pad}{label}:")
        for k, v in value.items():
            if isinstance(v, (list, dict)) and len(str(v)) > 80:
                print(f"{pad}  {k}: [{type(v).__name__} with {len(v)} items]")
            else:
                print(f"{pad}  {k}: {v}")
    elif isinstance(value, list):
        print(f"{pad}{label}: [{len(value)} items]")
        for item in value[:3]:
            print(f"{pad}  - {item}")
        if len(value) > 3:
            print(f"{pad}  ... and {len(value) - 3} more")
    else:
        print(f"{pad}{label}: {value}")


# =============================================================================
# Scenario 1: Tool discovery
# =============================================================================

def scenario_1_discovery(client: MCPClient) -> bool:
    """Verify the client discovers all 6 tools with correct names."""
    _print_header("SCENARIO 1 — Tool Discovery")
    tools = client.list_tools()
    tool_names = [t["name"] for t in tools]
    expected = {"read_file", "list_files", "write_file",
                "search_in_file", "watch_directory", "batch_process"}

    print(f"Discovered {len(tools)} tools:")
    for t in tools:
        print(f"  - {t['name']}: {t['description'][:70]}")

    missing = expected - set(tool_names)
    assert not missing, f"Missing tools: {missing}"
    assert len(tools) == 6, f"Expected 6 tools, got {len(tools)}"
    print("\n✅ All 6 tools discovered correctly")
    return True


# =============================================================================
# Scenario 2: File read via MCP
# =============================================================================

def scenario_2_read_file(client: MCPClient) -> bool:
    """Read a resume through the MCP server and verify content."""
    _print_header("SCENARIO 2 — File Read via MCP")
    result = client.call_tool("read_file", {"filepath": SAMPLE_RESUME})

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result.get("success") is True, f"read_file failed: {result.get('error')}"
    assert result.get("content"), "Content should not be empty"
    assert "JOHN DOE" in result["content"], "Should contain candidate name"

    _print_result("success", result.get("success"))
    _print_result("filename", result.get("metadata", {}).get("filename"))
    _print_result("size_bytes", result.get("metadata", {}).get("size_bytes"))
    print(f"  content preview: {result['content'][:100]}...")

    print("\n✅ File read through MCP returned correct content")
    return True


# =============================================================================
# Scenario 3: Directory listing and filtering
# =============================================================================

def scenario_3_list_files(client: MCPClient) -> bool:
    """List files via MCP and verify extension filtering works."""
    _print_header("SCENARIO 3 — Directory Listing & Filtering")

    # All files
    all_files = client.call_tool("list_files", {"directory": RESUME_DIR})
    assert isinstance(all_files, list), f"Expected list, got {type(all_files)}"
    assert len(all_files) == 30, f"Expected 30 resumes, got {len(all_files)}"
    print(f"  All files: {len(all_files)}")

    # Filtered to .txt
    txt_files = client.call_tool("list_files", {
        "directory": RESUME_DIR, "extension": ".txt"
    })
    assert isinstance(txt_files, list)
    assert len(txt_files) == 30, f"Expected 30 .txt files, got {len(txt_files)}"
    assert all(f["name"].endswith(".txt") for f in txt_files)
    print(f"  .txt files: {len(txt_files)}")

    # Filtered to .pdf (should be 0)
    pdf_files = client.call_tool("list_files", {
        "directory": RESUME_DIR, "extension": ".pdf"
    })
    assert len(pdf_files) == 0
    print(f"  .pdf files: {len(pdf_files)}")

    print("\n✅ Directory listing and extension filtering work correctly")
    return True


# =============================================================================
# Scenario 4: watch_directory — detect changes between calls
# =============================================================================

def scenario_4_watch_directory(client: MCPClient) -> bool:
    """Verify watch_directory detects new and modified files."""
    _print_header("SCENARIO 4 — watch_directory (change detection)")

    # First call: all existing files appear as "added"
    result1 = client.call_tool("watch_directory", {
        "directory": RESUME_DIR, "extensions": [".txt"]
    })
    assert isinstance(result1, dict)
    assert result1["total_files"] == 30
    assert len(result1["added"]) == 30
    assert len(result1["modified"]) == 0
    assert len(result1["removed"]) == 0
    print(f"  First watch: added={len(result1['added'])}, "
          f"modified={len(result1['modified'])}, removed={len(result1['removed'])}")

    # Second call: no changes
    result2 = client.call_tool("watch_directory", {
        "directory": RESUME_DIR, "extensions": [".txt"]
    })
    assert result2["total_files"] == 30
    assert len(result2["added"]) == 0
    assert len(result2["modified"]) == 0
    assert len(result2["removed"]) == 0
    print(f"  Second watch: added={len(result2['added'])}, "
          f"modified={len(result2['modified'])}, removed={len(result2['removed'])}")

    # Create a temp file to trigger detection
    test_file = os.path.join(RESUME_DIR, "_mcp_watch_test.txt")
    try:
        with open(test_file, "w") as f:
            f.write("temporary file for watch_directory test")

        result3 = client.call_tool("watch_directory", {
            "directory": RESUME_DIR, "extensions": [".txt"]
        })
        assert result3["total_files"] == 31
        assert "_mcp_watch_test.txt" in result3["added"]
        print(f"  After adding file: added={result3['added']}, total={result3['total_files']}")

        # Remove the file and verify detection
        os.remove(test_file)
        result4 = client.call_tool("watch_directory", {
            "directory": RESUME_DIR, "extensions": [".txt"]
        })
        assert "_mcp_watch_test.txt" in result4["removed"]
        print(f"  After removing file: removed={result4['removed']}, total={result4['total_files']}")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

    print("\n✅ watch_directory correctly detects added and removed files")
    return True


# =============================================================================
# Scenario 5: batch_process — multiple files in one call
# =============================================================================

def scenario_5_batch_process(client: MCPClient) -> bool:
    """Verify batch_process reads, searches, and stats multiple files."""
    _print_header("SCENARIO 5 — batch_process (multi-file operations)")

    files = [
        SAMPLE_RESUME,
        SECOND_RESUME,
        os.path.join(RESUME_DIR, "nonexistent_resume.txt"),  # should fail gracefully
    ]

    # --- Read operation ---
    read_result = client.call_tool("batch_process", {
        "filepaths": files, "operation": "read"
    })
    assert isinstance(read_result, dict)
    assert read_result["operation"] == "read"
    assert read_result["total_files"] == 3
    assert read_result["successful"] == 2
    assert read_result["failed"] == 1
    print(f"  Read: {read_result['successful']} ok, {read_result['failed']} failed, "
          f"{read_result['processing_time']}s")

    # Verify the failed file has an error
    failed = [r for r in read_result["results"] if not r["success"]]
    assert len(failed) == 1
    assert "nonexistent" in failed[0]["filepath"]
    print(f"  Failed file error: {failed[0]['error']}")

    # --- Search operation ---
    search_result = client.call_tool("batch_process", {
        "filepaths": [SAMPLE_RESUME, SECOND_RESUME],
        "operation": "search",
        "keyword": "Python"
    })
    assert search_result["operation"] == "search"
    assert search_result["successful"] == 2
    for r in search_result["results"]:
        print(f"  Search '{r['filepath'][-30:]}': {r['total_matches']} matches")
    assert all(r["success"] for r in search_result["results"])

    # --- Stats operation ---
    stats_result = client.call_tool("batch_process", {
        "filepaths": [SAMPLE_RESUME, SECOND_RESUME],
        "operation": "stats"
    })
    assert stats_result["operation"] == "stats"
    assert stats_result["successful"] == 2
    for r in stats_result["results"]:
        print(f"  Stats {r['name']}: {r['size_bytes']} bytes, {r['extension']}")
    assert all(r["success"] for r in stats_result["results"])

    # --- Error handling: unknown operation ---
    err_result = client.call_tool("batch_process", {
        "filepaths": [SAMPLE_RESUME], "operation": "delete"
    })
    assert "error" in err_result
    print(f"  Unknown operation error: {err_result['error']}")

    print("\n✅ batch_process handles read, search, stats, and error cases")
    return True


# =============================================================================
# Scenario 6: Agent end-to-end through MCP
# =============================================================================

def scenario_6_agent_via_mcp(client: MCPClient) -> bool:
    """
    Verify the LangGraph agent uses MCP-mediated file tools.

    This scenario exercises the agent_workflow's file_tools node through MCP.
    Since the full agent pipeline needs an LLM, we verify the tool wiring
    directly: the MCP-backed StructuredTools are callable and return the same
    shapes as the direct tools.
    """
    _print_header("SCENARIO 6 — Agent file tools via MCP")

    # Get LangChain tools from the MCP client
    lc_tools = client.get_tools()
    assert len(lc_tools) == 6, f"Expected 6 LangChain tools, got {len(lc_tools)}"

    tool_map = {t.name: t for t in lc_tools}
    print(f"  LangChain tools: {list(tool_map.keys())}")

    # --- read_file via LangChain tool ---
    read_tool = tool_map["read_file"]
    result = read_tool.invoke({"filepath": SAMPLE_RESUME})
    if isinstance(result, str):
        result = json.loads(result)
    assert isinstance(result, dict)
    assert result.get("success") is True
    assert "JOHN DOE" in result.get("content", "")
    print(f"  read_file.invoke: success={result['success']}, "
          f"content starts with: {result['content'][:40]}...")

    # --- list_files via LangChain tool ---
    list_tool = tool_map["list_files"]
    result = list_tool.invoke({"directory": RESUME_DIR, "extension": ".txt"})
    if isinstance(result, str):
        result = json.loads(result)
    assert isinstance(result, list)
    assert len(result) == 30
    print(f"  list_files.invoke: {len(result)} files returned")

    # --- search_in_file via LangChain tool ---
    search_tool = tool_map["search_in_file"]
    result = search_tool.invoke({"filepath": SAMPLE_RESUME, "keyword": "Python"})
    if isinstance(result, str):
        result = json.loads(result)
    assert isinstance(result, dict)
    assert result.get("total_matches", 0) > 0
    print(f"  search_in_file.invoke: {result['total_matches']} matches for 'Python'")

    # --- watch_directory via LangChain tool ---
    watch_tool = tool_map["watch_directory"]
    result = watch_tool.invoke({"directory": RESUME_DIR})
    if isinstance(result, str):
        result = json.loads(result)
    assert isinstance(result, dict)
    assert result.get("total_files") == 30
    print(f"  watch_directory.invoke: {result['total_files']} files, "
          f"{len(result.get('added', []))} new")

    # --- batch_process via LangChain tool ---
    batch_tool = tool_map["batch_process"]
    result = batch_tool.invoke({
        "filepaths": [SAMPLE_RESUME, SECOND_RESUME],
        "operation": "read"
    })
    if isinstance(result, str):
        result = json.loads(result)
    assert isinstance(result, dict)
    assert result["successful"] == 2
    print(f"  batch_process.invoke: {result['successful']}/{result['total_files']} ok")

    print("\n✅ All MCP-backed LangChain tools callable and return correct results")
    return True


# =============================================================================
# Runner
# =============================================================================

SCENARIOS = [
    ("Tool discovery and catalogue", scenario_1_discovery),
    ("File read via MCP tool", scenario_2_read_file),
    ("Directory listing and filtering", scenario_3_list_files),
    ("watch_directory — change detection", scenario_4_watch_directory),
    ("batch_process — multi-file operations", scenario_5_batch_process),
    ("Agent file tools via MCP (LangChain bridge)", scenario_6_agent_via_mcp),
]


def run_all(scenario_indices: List[int] | None = None) -> int:
    """Run the specified scenarios (or all if none given)."""
    indices = scenario_indices or list(range(1, len(SCENARIOS) + 1))
    results: Dict[int, bool] = {}

    client = MCPClient()
    try:
        print("\n" + "=" * 60)
        print("  MCP Test Scenarios — Starting filesystem MCP server")
        print("=" * 60)
        client.start()
        print(f"  Connected. {len(client.list_tools())} tools available.\n")

        for idx in indices:
            if idx < 1 or idx > len(SCENARIOS):
                print(f"\n⚠️  Scenario {idx} does not exist (1-{len(SCENARIOS)})")
                continue
            name, func = SCENARIOS[idx - 1]
            try:
                ok = func(client)
                results[idx] = ok
            except Exception as exc:
                print(f"\n❌ Scenario {idx} ({name}) failed: {exc}")
                import traceback
                traceback.print_exc()
                results[idx] = False
    finally:
        client.stop()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    for idx, ok in sorted(results.items()):
        name = SCENARIOS[idx - 1][0]
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  Scenario {idx}: {status} — {name}")
    print(f"\n  {passed} passed, {failed} failed out of {len(results)}")
    print(f"{'=' * 60}")

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="MCP Test Scenarios")
    parser.add_argument(
        "--scenario", type=int, action="append",
        help="Run a specific scenario by number (1-6). Repeat for multiple. "
             "Omit to run all."
    )
    args = parser.parse_args()
    sys.exit(run_all(args.scenario))


if __name__ == "__main__":
    main()
