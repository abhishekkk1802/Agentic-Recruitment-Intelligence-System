"""
Filesystem MCP Server
====================

Exposes the Milestone 1 file-system tools through the Model Context Protocol
so any MCP-compatible client — including the LangGraph matching agent — can
discover and invoke them without importing the Python functions directly.

Protocol: JSON-RPC 2.0 over stdio (the MCP SDK handles framing and dispatch).
Transport: stdio by default; ``run(transport='streamable-http')`` for network.

Tools exposed (discovered via ``tools/list``):
    read_file, list_files, write_file, search_in_file,
    watch_directory (new), batch_process (new)

Resources exposed (discovered via ``resources/list``):
    file://<path>  —  raw text content of any readable file

The server is a standard Python script run as a subprocess by the MCP client
in ``mcp_client.py``. It can also be tested standalone:

    python filesystem_mcp_server.py            # run as stdio server
    python filesystem_mcp_server.py --inspect    # print tool/resource catalogue
"""

import asyncio
import json
import os
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from mcp.types import TextContent, Tool

# Reuse the Milestone 1 implementations rather than duplicating logic.
from fs_tools import (
    read_file as _read_file,
    list_files as _list_files,
    write_file as _write_file,
    search_in_file as _search_in_file,
)

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("filesystem_mcp_server")

# =============================================================================
# Create the MCP server instance
# =============================================================================

mcp = MCPServer(
    name="filesystem-mcp-server",
    version="1.0.0",
    description=(
        "File system operations for the LLM-Powered File System Assistant. "
        "Exposes read, list, write, search, directory-watch and batch-process "
        "capabilities via the Model Context Protocol (JSON-RPC 2.0)."
    ),
)

# =============================================================================
# State for watch_directory
# =============================================================================
# The watch state is kept in-process. Each call to watch_directory returns the
# delta since the last call for that directory, then updates the snapshot.
# This is deliberately simple — no background polling thread — because MCP tool
# calls are synchronous request/response, and a client that wants continuous
# monitoring calls watch_directory on a timer.

_watch_state: Dict[str, Dict[str, float]] = {}  # directory -> {filename: mtime}


def _file_signature(path: Path) -> float:
    """Return the mtime for a file, or 0 if it cannot be stat'd."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _compute_watch_delta(
    directory: str,
    extensions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare the current directory listing against the stored snapshot.

    Returns added, modified, removed lists plus counts.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return {
            "error": f"Directory not found: {directory}",
            "added": [], "modified": [], "removed": [],
        }

    ext_set = {e.lower() for e in extensions} if extensions else None
    current: Dict[str, float] = {}
    for entry in dir_path.iterdir():
        if not entry.is_file():
            continue
        if ext_set and entry.suffix.lower() not in ext_set:
            continue
        current[entry.name] = _file_signature(entry)

    previous = _watch_state.get(directory, {})
    _watch_state[directory] = current

    added = sorted(n for n in current if n not in previous)
    modified = sorted(
        n for n in current
        if n in previous and current[n] != previous[n]
    )
    removed = sorted(n for n in previous if n not in current)

    return {
        "directory": str(dir_path.absolute()),
        "added": added,
        "modified": modified,
        "removed": removed,
        "total_files": len(current),
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# MCP Tools
# =============================================================================
# Each tool is registered with @mcp.tool(). The MCP SDK inspects the function
# signature and docstring to build the JSON schema that clients discover via
# tools/list. Return values are wrapped in TextContent automatically; when a
# structured dict is returned we serialise it to JSON first so the client gets
# parseable data.

@mcp.tool()
def read_file(filepath: str) -> str:
    """Read and extract text content from a file (PDF, TXT, DOCX).

    Args:
        filepath: Path to the file to read.
    """
    result = _read_file(filepath)
    return json.dumps(result, indent=2)


@mcp.tool()
def list_files(directory: str, extension: str = "") -> str:
    """List all files in a directory with optional extension filtering.

    Args:
        directory: Directory path to list files from.
        extension: Optional file extension filter (e.g. '.txt', '.pdf'). Omit for all files.
    """
    ext = extension or None
    result = _list_files(directory, ext)
    return json.dumps(result, indent=2)


@mcp.tool()
def write_file(filepath: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed.

    Args:
        filepath: Path where to write the file.
        content: Text content to write.
    """
    result = _write_file(filepath, content)
    return json.dumps(result, indent=2)


@mcp.tool()
def search_in_file(filepath: str, keyword: str) -> str:
    """Search for a keyword in file content and return matches with surrounding context.

    Args:
        filepath: Path to the file to search in.
        keyword: Keyword to search for (case-insensitive).
    """
    result = _search_in_file(filepath, keyword)
    return json.dumps(result, indent=2)


@mcp.tool()
def watch_directory(directory: str, extensions: list[str] | None = None) -> str:
    """Monitor a directory for new, modified, or removed files since the last check.

    Maintains an in-process snapshot per directory. Each call returns the delta
    (added / modified / removed) since the previous call, then updates the
    snapshot. Call repeatedly to track changes over time.

    Args:
        directory: Directory path to monitor.
        extensions: Optional list of extensions to filter (e.g. ['.txt', '.pdf']). Omit for all files.
    """
    result = _compute_watch_delta(directory, extensions)
    return json.dumps(result, indent=2)


@mcp.tool()
def batch_process(
    filepaths: list[str],
    operation: str = "read",
    keyword: str = "",
) -> str:
    """Process multiple files in a single call for efficiency.

    Supported operations:
    - ``read``: Read content from each file. Returns a list of {filepath, success, content, error}.
    - ``search``: Search each file for ``keyword``. Returns a list of {filepath, total_matches}.
    - ``stats``: Return file metadata (size, modified date) without reading content.

    Args:
        filepaths: List of file paths to process.
        operation: One of 'read', 'search', or 'stats'.
        keyword: Required when operation is 'search'.
    """
    if not filepaths:
        return json.dumps({"error": "No file paths provided", "results": []})

    operation = operation.lower()
    if operation not in ("read", "search", "stats"):
        return json.dumps({
            "error": f"Unknown operation '{operation}'. Use 'read', 'search', or 'stats'.",
            "results": [],
        })

    if operation == "search" and not keyword:
        return json.dumps({"error": "search operation requires a keyword", "results": []})

    results: List[Dict[str, Any]] = []
    start_time = time.time()

    for filepath in filepaths:
        try:
            if operation == "read":
                r = _read_file(filepath)
                results.append({
                    "filepath": filepath,
                    "success": r.get("success", False),
                    "content": r.get("content", "")[:2000] if r.get("success") else None,
                    "content_truncated": len(r.get("content", "")) > 2000 if r.get("success") else False,
                    "error": r.get("error"),
                })
            elif operation == "search":
                r = _search_in_file(filepath, keyword)
                results.append({
                    "filepath": filepath,
                    "success": r.get("success", False),
                    "total_matches": r.get("total_matches", 0),
                    "first_match_line": (
                        r["matches"][0]["line_number"] if r.get("matches") else None
                    ),
                    "error": r.get("error"),
                })
            elif operation == "stats":
                p = Path(filepath)
                if p.exists():
                    stat = p.stat()
                    results.append({
                        "filepath": filepath,
                        "success": True,
                        "name": p.name,
                        "size_bytes": stat.st_size,
                        "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "extension": p.suffix.lower(),
                    })
                else:
                    results.append({
                        "filepath": filepath,
                        "success": False,
                        "error": "File not found",
                    })
        except Exception as exc:
            results.append({
                "filepath": filepath,
                "success": False,
                "error": str(exc),
            })

    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r.get("success"))

    return json.dumps({
        "operation": operation,
        "total_files": len(filepaths),
        "successful": success_count,
        "failed": len(filepaths) - success_count,
        "processing_time": round(elapsed, 3),
        "results": results,
    }, indent=2)


# =============================================================================
# MCP Resources
# =============================================================================
# Resources are addressable by URI. The MCP SDK exposes them via
# resources/list and resources/read, giving clients a standardised way to
# access file content without calling a tool.

@mcp.resource("file://{filepath}")
def read_file_resource(filepath: str) -> str:
    """Read any file by its path via the file:// resource URI scheme."""
    result = _read_file(filepath)
    if result.get("success"):
        return result.get("content", "")
    return f"Error reading {filepath}: {result.get('error', 'unknown')}"


# =============================================================================
# Resource templates for resume directories
# =============================================================================

@mcp.resource("dir://{directory}")
def list_directory_resource(directory: str) -> str:
    """List files in a directory via the dir:// resource URI scheme."""
    result = _list_files(directory)
    return json.dumps(result, indent=2)


# =============================================================================
# Inspect / catalogue mode (for testing without running the full server)
# =============================================================================

def get_tool_catalogue() -> List[Dict[str, Any]]:
    """Return a static description of every tool this server exposes."""
    return [
        {
            "name": "read_file",
            "description": "Read and extract text content from a file (PDF, TXT, DOCX)",
            "parameters": {"filepath": {"type": "string", "required": True}},
        },
        {
            "name": "list_files",
            "description": "List all files in a directory with optional extension filtering",
            "parameters": {
                "directory": {"type": "string", "required": True},
                "extension": {"type": "string", "required": False, "default": ""},
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file, creating parent directories if needed",
            "parameters": {
                "filepath": {"type": "string", "required": True},
                "content": {"type": "string", "required": True},
            },
        },
        {
            "name": "search_in_file",
            "description": "Search for a keyword in file content and return matches with context",
            "parameters": {
                "filepath": {"type": "string", "required": True},
                "keyword": {"type": "string", "required": True},
            },
        },
        {
            "name": "watch_directory",
            "description": "Monitor a directory for new, modified, or removed files since last check",
            "parameters": {
                "directory": {"type": "string", "required": True},
                "extensions": {"type": "array", "required": False, "items": {"type": "string"}},
            },
        },
        {
            "name": "batch_process",
            "description": "Process multiple files in one call (read, search, or stats)",
            "parameters": {
                "filepaths": {"type": "array", "required": True, "items": {"type": "string"}},
                "operation": {"type": "string", "required": False, "default": "read"},
                "keyword": {"type": "string", "required": False, "default": ""},
            },
        },
    ]


def get_resource_catalogue() -> List[Dict[str, Any]]:
    """Return a static description of every resource this server exposes."""
    return [
        {
            "uri": "file://{filepath}",
            "description": "Read any file by path via the file:// URI scheme",
        },
        {
            "uri": "dir://{directory}",
            "description": "List files in a directory via the dir:// URI scheme",
        },
    ]


# =============================================================================
# Entry point
# =============================================================================

def main():
    """Run the MCP server or print its catalogue."""
    import sys

    if "--inspect" in sys.argv:
        print("=== Filesystem MCP Server — Tool Catalogue ===\n")
        for tool in get_tool_catalogue():
            print(f"  Tool: {tool['name']}")
            print(f"    {tool['description']}")
            params = tool["parameters"]
            for pname, pinfo in params.items():
                req = "required" if pinfo.get("required") else "optional"
                default = f" default={pinfo.get('default')!r}" if "default" in pinfo else ""
                print(f"    - {pname} ({pinfo['type']}, {req}){default}")
            print()

        print("=== Resource Catalogue ===\n")
        for res in get_resource_catalogue():
            print(f"  {res['uri']}")
            print(f"    {res['description']}")
            print()

        return

    logger.info("Starting filesystem MCP server on stdio")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
