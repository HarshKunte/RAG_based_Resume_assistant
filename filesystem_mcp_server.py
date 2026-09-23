"""FastMCP filesystem server for resume ingestion.

FastMCP owns the MCP wire protocol and stdio lifecycle. The project only needs
plain Python tools for filesystem operations, so the legacy hand-written
JSON-RPC handling is no longer part of the runtime.
"""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastmcp import FastMCP


SERVER_NAME = "resume-filesystem"
SERVER_VERSION = "1.0.0"


def _filesystem_tools():
    from fs_tools import list_files, read_file

    return list_files, read_file


def list_files(directory: str, extension: str | None = None) -> list[dict[str, Any]]:
    list_files_tool, _ = _filesystem_tools()
    return list_files_tool(directory, extension)


def read_file(filepath: str) -> dict[str, Any]:
    _, read_file_tool = _filesystem_tools()
    return read_file_tool(filepath)


def watch_directory(
    directory: str,
    known_files: list[str] | None = None,
    timeout: float = 0,
    poll_interval: float = 1,
    extension: str | None = None,
) -> list[dict[str, Any]]:
    """Return files that appear during the optional polling window."""
    known = {os.path.abspath(path) for path in (known_files or [])}
    deadline = time.monotonic() + max(0, timeout)
    while True:
        current = list_files(directory, extension)
        if current and "error" in current[0]:
            return current
        new_files = [
            item
            for item in current
            if os.path.abspath(item.get("filepath", "")) not in known
        ]
        if new_files or timeout <= 0 or time.monotonic() >= deadline:
            return new_files
        time.sleep(max(0.01, poll_interval))


def batch_process(
    filepaths: list[str], max_workers: int = 4
) -> list[dict[str, Any]]:
    """Read several resume files concurrently while preserving input order."""
    if not isinstance(filepaths, list):
        raise ValueError("filepaths must be a list")
    workers = max(1, min(max_workers, len(filepaths) or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(read_file, filepaths))


def build_fastmcp_server(root_directory: str | None = None) -> FastMCP:
    """Build the FastMCP server that exposes the resume filesystem tools."""
    _ = root_directory
    server = FastMCP(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        instructions=(
            "This server exposes resume filesystem tools for listing files, reading resume text, "
            "watching for new resumes, and processing multiple files in parallel."
        ),
    )

    @server.tool(name="list_files")
    def list_files_tool(directory: str, extension: str | None = None) -> list[dict[str, Any]]:
        return list_files(directory, extension)

    @server.tool(name="read_file")
    def read_file_tool(filepath: str) -> dict[str, Any]:
        return read_file(filepath)

    @server.tool(name="watch_directory")
    def watch_directory_tool(
        directory: str,
        known_files: list[str] | None = None,
        timeout: float = 0,
        poll_interval: float = 1,
        extension: str | None = None,
    ) -> list[dict[str, Any]]:
        return watch_directory(directory, known_files, timeout, poll_interval, extension)

    @server.tool(name="batch_process")
    def batch_process_tool(filepaths: list[str], max_workers: int = 4) -> list[dict[str, Any]]:
        return batch_process(filepaths, max_workers)

    return server


if __name__ == "__main__":
    asyncio.run(build_fastmcp_server(os.getenv("MCP_FILESYSTEM_ROOT")).run_stdio_async())