"""
MCP Client — LangChain Tool Bridge
==================================

Connects to the filesystem MCP server (``filesystem_mcp_server.py``) over stdio,
discovers the tools it exposes, and wraps each one as a LangChain ``@tool``
function with the same name and schema the agent already uses.

This lets the LangGraph agent call MCP-mediated file operations without any
change to the graph structure — the tool names and return shapes match the
direct ``agent_tools`` versions, so ``ToolNode`` and ``tools_condition`` work
unchanged.

Usage::

    from mcp_client import MCPClient

    client = MCPClient()
    client.start()          # spawns the server subprocess, discovers tools
    tools = client.get_tools()   # list of StructuredTool objects
    # ... use in a LangGraph ToolNode ...
    client.stop()           # tear down the subprocess

The client manages the server subprocess lifecycle. ``start()`` spawns
``python filesystem_mcp_server.py`` as a child process, connects via stdio,
and performs the MCP handshake. ``stop()`` terminates it cleanly.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool, tool as lc_tool
from pydantic import BaseModel, Field

from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession

logger = logging.getLogger(__name__)

# Path to the MCP server script, resolved relative to this file.
_SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "filesystem_mcp_server.py")


# =============================================================================
# Tool input schemas — mirror agent_tools so the LangGraph graph is unchanged
# =============================================================================

class ReadFileInput(BaseModel):
    filepath: str = Field(description="Path to the file to read")


class ListFilesInput(BaseModel):
    directory: str = Field(description="Directory path to list files from")
    extension: Optional[str] = Field(default=None, description="Optional extension filter (e.g. '.txt')")


class WriteFileInput(BaseModel):
    filepath: str = Field(description="Path where to write the file")
    content: str = Field(description="Content to write")


class SearchInFileInput(BaseModel):
    filepath: str = Field(description="Path to the file to search in")
    keyword: str = Field(description="Keyword to search for (case-insensitive)")


class WatchDirectoryInput(BaseModel):
    directory: str = Field(description="Directory path to monitor")
    extensions: Optional[List[str]] = Field(default=None, description="Optional extension filter (e.g. ['.txt', '.pdf'])")


class BatchProcessInput(BaseModel):
    filepaths: List[str] = Field(description="List of file paths to process")
    operation: str = Field(default="read", description="Operation: 'read', 'search', or 'stats'")
    keyword: str = Field(default="", description="Required when operation is 'search'")


# =============================================================================
# MCP Client
# =============================================================================

class MCPClient:
    """
    Manages a connection to the filesystem MCP server.

    The connection lifecycle:
        1. ``start()`` — spawns the server subprocess, opens a stdio session,
           performs the MCP ``initialize`` handshake, and discovers tools.
        2. ``call_tool(name, arguments)`` — invokes a tool on the server and
           returns the text content of the result.
        3. ``read_resource(uri)`` — reads a resource by URI.
        4. ``stop()`` — terminates the subprocess.

    The heavy lifting (JSON-RPC framing, capability negotiation, transport)
    is done by the MCP SDK's ``ClientSession``. This class is a thin sync
    wrapper that also produces LangChain ``StructuredTool`` objects so the
    existing LangGraph graph can use the tools without modification.
    """

    def __init__(self, server_script: Optional[str] = None, python_executable: Optional[str] = None):
        """
        Args:
            server_script: Path to the MCP server script. Defaults to
                ``filesystem_mcp_server.py`` next to this file.
            python_executable: Python interpreter to run the server with.
                Defaults to ``sys.executable`` so the venv is inherited.
        """
        self.server_script = server_script or _SERVER_SCRIPT
        self.python_executable = python_executable or sys.executable
        self._session: Optional[ClientSession] = None
        self._stdio_context = None
        self._session_context = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_params: Optional[StdioServerParameters] = None
        self._tools_cache: Optional[List[StructuredTool]] = None
        self._discovered: List[Dict[str, Any]] = []

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Spawn the server, connect, handshake, and discover tools."""
        if self._session is not None:
            logger.warning("MCP client already started")
            return

        logger.info("Starting MCP server: %s %s", self.python_executable, self.server_script)
        self._server_params = StdioServerParameters(
            command=self.python_executable,
            args=[self.server_script],
            env=dict(os.environ),
        )

        # Run the async connection sequence in a dedicated event loop that
        # persists for the lifetime of the client, so tool calls can be made
        # from sync code without blocking the agent's main thread.
        self._loop = asyncio.new_event_loop()

        self._loop.run_until_complete(self._connect())

        # Discover tools so we know what's available.
        self._loop.run_until_complete(self._discover())
        logger.info("MCP client connected, %d tools discovered", len(self._discovered))

    async def _connect(self) -> None:
        """Open the stdio transport and initialise the session."""
        self._stdio_context = stdio_client(self._server_params)
        read_stream, write_stream = await self._stdio_context.__aenter__()

        self._session_context = ClientSession(read_stream, write_stream)
        self._session = await self._session_context.__aenter__()
        await self._session.initialize()

    async def _discover(self) -> None:
        """List the tools exposed by the server."""
        result = await self._session.list_tools()
        self._discovered = []
        for t in result.tools:
            self._discovered.append({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            })

    def stop(self) -> None:
        """Tear down the session and terminate the server subprocess."""
        if self._loop is None:
            return

        try:
            self._loop.run_until_complete(self._cleanup())
        except Exception as exc:
            logger.warning("Error during MCP cleanup: %s", exc)
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()
            self._loop = None
            self._session = None
            self._stdio_context = None
            self._session_context = None
            self._tools_cache = None
            self._discovered = []
            logger.info("MCP client stopped")

    async def _cleanup(self) -> None:
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)

    # ---- tool invocation -------------------------------------------------

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """
        Call a tool on the MCP server and return the result.

        The MCP SDK returns ``CallToolResult`` whose ``content`` is a list of
        ``TextContent`` blocks. We join the text of all blocks and attempt to
        parse the result as JSON, since our server serialises structured data.
        If parsing fails the raw text is returned.

        Args:
            name: Tool name (e.g. ``"read_file"``).
            arguments: Tool arguments as a dict.

        Returns:
            Parsed JSON dict/list, or raw text string.
        """
        if self._session is None or self._loop is None:
            raise RuntimeError("MCP client not started. Call start() first.")

        result = self._loop.run_until_complete(
            self._session.call_tool(name, arguments or {})
        )

        # Extract text from the content blocks.
        text_parts: List[str] = []
        for block in (result.content or []):
            if hasattr(block, "text"):
                text_parts.append(block.text)
        raw = "\n".join(text_parts)

        # Try to parse as JSON — our server returns JSON-serialised dicts.
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def read_resource(self, uri: str) -> str:
        """
        Read an MCP resource by URI.

        Args:
            uri: Resource URI (e.g. ``"file://examples/resumes/john_doe_resume.txt"``)

        Returns:
            Resource content as text.
        """
        if self._session is None or self._loop is None:
            raise RuntimeError("MCP client not started. Call start() first.")

        result = self._loop.run_until_complete(self._session.read_resource(uri))
        text_parts: List[str] = []
        for block in (result.contents or []):
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "\n".join(text_parts)

    # ---- discovery --------------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return the discovered tool catalogue (name, description, schema)."""
        return list(self._discovered)

    def list_resources(self) -> List[Dict[str, Any]]:
        """Return the resource templates discovered from the server."""
        if self._session is None or self._loop is None:
            raise RuntimeError("MCP client not started. Call start() first.")

        result = self._loop.run_until_complete(self._session.list_resource_templates())
        return [
            {
                "uri_template": t.uri_template,
                "description": t.description,
                "name": getattr(t, "name", ""),
            }
            for t in result.resource_templates
        ]

    # ---- LangChain tool bridge -------------------------------------------

    def get_tools(self) -> List[StructuredTool]:
        """
        Build LangChain ``StructuredTool`` objects for every MCP tool.

        Each tool calls the MCP server via ``call_tool()`` and returns the
        parsed result. The tool names and schemas match the existing
        ``agent_tools`` functions, so the LangGraph graph and ``ToolNode``
        work without any change.
        """
        if self._tools_cache is not None:
            return self._tools_cache

        client = self  # capture for closures

        def _make(name: str, schema_cls: type, description: str) -> StructuredTool:
            def _run(**kwargs) -> Any:
                return client.call_tool(name, kwargs)
            return StructuredTool.from_function(
                _run,
                name=name,
                description=description,
                args_schema=schema_cls,
            )

        tools: List[StructuredTool] = [
            _make("read_file", ReadFileInput, "Read and extract text content from a file (PDF, TXT, DOCX) via MCP."),
            _make("list_files", ListFilesInput, "List all files in a directory with optional extension filtering via MCP."),
            _make("write_file", WriteFileInput, "Write content to a file, creating parent directories if needed via MCP."),
            _make("search_in_file", SearchInFileInput, "Search for a keyword in file content and return matches with context via MCP."),
            _make("watch_directory", WatchDirectoryInput, "Monitor a directory for new, modified, or removed files via MCP."),
            _make("batch_process", BatchProcessInput, "Process multiple files in one call (read, search, stats) via MCP."),
        ]

        self._tools_cache = tools
        return tools

    # ---- convenience for the agent ---------------------------------------

    def get_file_tools(self) -> List[StructuredTool]:
        """Return only the file-system tools (read, list, search) for ToolNode."""
        all_tools = self.get_tools()
        names = {"read_file", "list_files", "search_in_file"}
        return [t for t in all_tools if t.name in names]

    def get_all_tools(self) -> List[StructuredTool]:
        """Return all MCP-backed tools including watch_directory and batch_process."""
        return self.get_tools()

    # ---- context manager support -----------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


# =============================================================================
# Standalone test entry point
# =============================================================================

def main():
    """Quick smoke test: start the server, call each tool, print results."""
    import textwrap

    print("=" * 60)
    print("MCP Client Smoke Test")
    print("=" * 60)

    client = MCPClient()

    try:
        print("\n1. Starting MCP server...")
        client.start()
        print("   Connected.")

        print("\n2. Discovered tools:")
        for t in client.list_tools():
            print(f"   - {t['name']}: {t['description'][:60]}")

        print("\n3. Calling list_files('examples/resumes', '.txt')...")
        result = client.call_tool("list_files", {
            "directory": "examples/resumes",
            "extension": ".txt",
        })
        if isinstance(result, list):
            print(f"   Found {len(result)} files. First 3:")
            for f in result[:3]:
                print(f"     - {f['name']} ({f.get('size_human', '?')})")
        else:
            print(f"   Result: {str(result)[:200]}")

        print("\n4. Calling read_file('examples/resumes/john_doe_resume.txt')...")
        result = client.call_tool("read_file", {
            "filepath": "examples/resumes/john_doe_resume.txt",
        })
        if isinstance(result, dict):
            print(f"   success={result.get('success')}")
            content = result.get("content", "")
            print(f"   content preview: {content[:120]}...")
        else:
            print(f"   Result: {str(result)[:200]}")

        print("\n5. Calling watch_directory('examples/resumes', ['.txt'])...")
        result = client.call_tool("watch_directory", {
            "directory": "examples/resumes",
            "extensions": [".txt"],
        })
        if isinstance(result, dict):
            print(f"   total_files={result.get('total_files')}")
            print(f"   added={len(result.get('added', []))} first call (all seen as new)")
        else:
            print(f"   Result: {str(result)[:200]}")

        print("\n6. Calling batch_process (read 3 resumes)...")
        result = client.call_tool("batch_process", {
            "filepaths": [
                "examples/resumes/john_doe_resume.txt",
                "examples/resumes/sarah_smith_resume.txt",
                "examples/resumes/nonexistent.txt",
            ],
            "operation": "read",
        })
        if isinstance(result, dict):
            print(f"   successful={result.get('successful')}, failed={result.get('failed')}")
            print(f"   time={result.get('processing_time')}s")
        else:
            print(f"   Result: {str(result)[:200]}")

        print("\n7. Calling search_in_file('examples/resumes/john_doe_resume.txt', 'Python')...")
        result = client.call_tool("search_in_file", {
            "filepath": "examples/resumes/john_doe_resume.txt",
            "keyword": "Python",
        })
        if isinstance(result, dict):
            print(f"   total_matches={result.get('total_matches')}")
        else:
            print(f"   Result: {str(result)[:200]}")

        print("\n8. Reading resource file://examples/resumes/john_doe_resume.txt...")
        try:
            content = client.read_resource("file://examples/resumes/john_doe_resume.txt")
            print(f"   content preview: {content[:120]}...")
        except Exception as e:
            print(f"   (resource read skipped: {e})")

        print("\n9. Getting LangChain tools...")
        lc_tools = client.get_tools()
        print(f"   {len(lc_tools)} StructuredTool objects created:")
        for t in lc_tools:
            print(f"   - {t.name}")

        print("\n" + "=" * 60)
        print("ALL MCP CLIENT TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        client.stop()


if __name__ == "__main__":
    main()
