"""Unit tests for SystemOneEngine Stdio MCP Server."""

import json

import pytest

from system_one_engine.server.mcp_server import process_message


def test_mcp_initialize():
    """Verify initialize request returns correct server info and capabilities."""
    msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {},
    }
    resp = process_message(msg)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    result = resp["result"]
    assert result["serverInfo"]["name"] == "system-one-engine"
    assert "tools" in result["capabilities"]


def test_mcp_ping():
    """Verify ping method returns empty dict."""
    msg = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
    resp = process_message(msg)
    assert resp["id"] == 2
    assert resp["result"] == {}


def test_mcp_tools_list():
    """Verify tools/list exposes all decision primitives."""
    msg = {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
    resp = process_message(msg)
    tools = resp["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert "system_one_choice" in tool_names
    assert "system_one_score" in tool_names
    assert "system_one_boolean" in tool_names
    assert "system_one_safety_check" in tool_names
    assert "system_one_shortlist" in tool_names


def test_mcp_tool_call_safety_check():
    """Verify system_one_safety_check tool blocks lethal commands."""
    msg = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "system_one_safety_check",
            "arguments": {"command": "rm -rf /", "safety_mode": "STRICT"},
        },
    }
    resp = process_message(msg)
    assert resp["id"] == 4
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["is_safe"] is False
    assert content["requires_confirmation"] is True


def test_mcp_tool_call_choice():
    """Verify system_one_choice evaluates candidate criteria."""
    msg = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "system_one_choice",
            "arguments": {
                "context": "User wants to check unit tests",
                "instruction": "Select the appropriate terminal command",
                "criteria": {
                    "pytest": "Run automated python test suite",
                    "format": "Reformat code with ruff",
                },
            },
        },
    }
    resp = process_message(msg)
    assert resp["id"] == 5
    assert not resp["result"]["isError"]
    content = json.loads(resp["result"]["content"][0]["text"])
    assert "choice" in content
    assert "confidence" in content
    assert "probabilities" in content
    assert content["choice"] in ("pytest", "format")


def test_mcp_tool_call_shortlist():
    """Verify system_one_shortlist ranks candidates."""
    msg = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "system_one_shortlist",
            "arguments": {
                "query": "unit tests",
                "candidates": ["tests/test_api.py", "docs/DEV_LOG.md", "tests/test_onnx.py", "README.md"],
                "top_k": 2,
            },
        },
    }
    resp = process_message(msg)
    content = json.loads(resp["result"]["content"][0]["text"])
    assert len(content["selected_candidates"]) == 2
    assert all(c in ["tests/test_api.py", "docs/DEV_LOG.md", "tests/test_onnx.py", "README.md"] for c in content["selected_candidates"])


def test_mcp_unknown_method():
    """Verify unknown method returns -32601 error."""
    msg = {"jsonrpc": "2.0", "id": 99, "method": "non_existent_method"}
    resp = process_message(msg)
    assert resp["error"]["code"] == -32601
