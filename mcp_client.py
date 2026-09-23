"""Simplified MCP client for the local resume filesystem server."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class MCPFilesystemClient:
    """Small synchronous wrapper around the official MCP stdio client."""

    def __init__(
        self,
        server_script: str | None = None,
        root_directory: str | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.server_script = server_script or str(
            Path(__file__).with_name("filesystem_mcp_server.py")
        )
        self.root_directory = root_directory or os.getenv(
            "MCP_FILESYSTEM_ROOT", str(Path(__file__).with_name("resumes"))
        )
        self.python_executable = python_executable or sys.executable
        self._process: Any = None

    def _server_params(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=self.python_executable,
            args=[self.server_script],
            env={**os.environ, "MCP_FILESYSTEM_ROOT": self.root_directory},
            cwd=str(Path(self.server_script).resolve().parent),
        )

    def _run(self, action):
        self._process = self._server_params()

        async def _runner():
            async with stdio_client(self._process) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await action(session)

        try:
            return asyncio.run(_runner())
        finally:
            self._process = None

    def _normalise_tool_result(self, result: Any) -> Any:
        if hasattr(result, "structured_content"):
            payload = result.structured_content
            if isinstance(payload, dict) and "result" in payload:
                return payload["result"]
            return payload

        if isinstance(result, dict):
            structured = result.get("structuredContent")
            if structured is not None:
                if isinstance(structured, dict) and "result" in structured:
                    return structured["result"]
                return structured

            content = result.get("content") or []
            first_item = content[0] if content else None
            if isinstance(first_item, dict):
                if "json" in first_item:
                    return first_item["json"]
                if "text" in first_item:
                    text = first_item["text"]
                    try:
                        return json.loads(text)
                    except (TypeError, ValueError):
                        return text
            return result

        return result

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        params = params or {}

        async def _request(session: ClientSession):
            if method == "tools/list":
                result = await session.list_tools()
                return [tool.model_dump() for tool in result.tools]
            if method == "resources/list":
                result = await session.list_resources()
                return [item.model_dump() for item in result.resources]
            if method == "resources/read":
                result = await session.read_resource(params["uri"])
                return [item.model_dump() for item in result.contents]
            if method == "tools/call":
                response = await session.call_tool(
                    name=params["name"],
                    arguments=params.get("arguments") or {},
                )
                return self._normalise_tool_result(response)
            return await session.send_request(method, params)

        return self._run(_request)

    def call_tool(self, name: str, **arguments: Any) -> Any:
        return self._normalise_tool_result(
            self.request("tools/call", {"name": name, "arguments": arguments})
        )

    def list_files(self, directory: str, extension: str | None = None) -> Any:
        return self.call_tool("list_files", directory=directory, extension=extension)

    def read_file(self, filepath: str) -> Any:
        return self.call_tool("read_file", filepath=filepath)

    def watch_directory(self, directory: str, **options: Any) -> Any:
        return self.call_tool("watch_directory", directory=directory, **options)

    def batch_process(self, filepaths: list[str], max_workers: int = 4) -> Any:
        return self.call_tool("batch_process", filepaths=filepaths, max_workers=max_workers)

    def list_resources(self, directory: str | None = None) -> Any:
        params = {"directory": directory} if directory else {}
        response = self.request("resources/list", params)
        return response if isinstance(response, list) else []

    def close(self) -> None:
        """Clear the lightweight transport state kept for compatibility."""
        self._process = None

    def __enter__(self) -> "MCPFilesystemClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()