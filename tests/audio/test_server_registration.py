"""Verify analyze_audio is registered on the FastMCP server."""

import asyncio
from unittest.mock import MagicMock, patch


def test_analyze_audio_tool_registered() -> None:
    # Patch PlaudClient to avoid Keychain I/O during import
    with patch("plaud_mcp.plaud_client.PlaudClient", return_value=MagicMock()):
        from plaud_mcp.server import mcp  # noqa: PLC0415

    tools = asyncio.run(mcp.list_tools())
    tool_names = [t.name for t in tools]
    assert "analyze_audio" in tool_names


def test_analyze_audio_tool_has_expected_parameters() -> None:
    with patch("plaud_mcp.plaud_client.PlaudClient", return_value=MagicMock()):
        from plaud_mcp.server import mcp  # noqa: PLC0415

    tools = asyncio.run(mcp.list_tools())
    tool = next(t for t in tools if t.name == "analyze_audio")
    schema_props = tool.inputSchema.get("properties", {})
    assert "audio_path" in schema_props
    assert "language" in schema_props
    assert "num_speakers" in schema_props
    assert "return_word_timestamps" in schema_props
