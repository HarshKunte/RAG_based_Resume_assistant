import asyncio
import tempfile
import unittest
from pathlib import Path

from filesystem_mcp_server import batch_process, watch_directory
from matching_agent import MCPFilesystemClient


class FilesystemMCPTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.first = self.root / "first.txt"
        self.second = self.root / "second.txt"
        self.first.write_text("Python and React experience", encoding="utf-8")
        self.second.write_text("Data engineering experience", encoding="utf-8")

    def tearDown(self):
        self.directory.cleanup()

    def test_tool_and_resource_discovery(self):
        server = __import__("filesystem_mcp_server").build_fastmcp_server(root_directory=str(self.root))
        tools = asyncio.run(server.list_tools())
        self.assertEqual(
            {tool.name for tool in tools},
            {"list_files", "read_file", "watch_directory", "batch_process"},
        )
        files = asyncio.run(server.call_tool("list_files", {"directory": str(self.root), "extension": ".txt"}))
        self.assertEqual(len(files.structured_content["result"]), 2)

    def test_batch_and_watch(self):
        results = batch_process([str(self.first), str(self.second)])
        self.assertEqual(len(results), 2)
        self.assertEqual(
            {item["filepath"] for item in watch_directory(str(self.root), [str(self.first)])},
            {str(self.second)},
        )

    def test_watch_normalises_known_paths(self):
        discovered = watch_directory(str(self.root), [str(self.first.resolve())])
        self.assertEqual(
            {item["filepath"] for item in discovered},
            {str(self.second)},
        )

    def test_client_uses_mcp_transport(self):
        client = MCPFilesystemClient(root_directory=str(self.root))
        try:
            files = client.list_files(str(self.root), ".txt")
            process = client._process
            self.assertEqual(len(files), 2)
            self.assertTrue(client.read_file(str(self.first))["success"])
            self.assertIs(process, client._process)
        finally:
            client.close()
        self.assertIsNone(client._process)

    def test_fastmcp_server_contract(self):
        from filesystem_mcp_server import build_fastmcp_server

        server = build_fastmcp_server(root_directory=str(self.root))
        tool_names = asyncio.run(server.list_tools())
        self.assertTrue({"list_files", "read_file", "watch_directory", "batch_process"}.issubset({tool.name for tool in tool_names}))


if __name__ == "__main__":
    unittest.main()