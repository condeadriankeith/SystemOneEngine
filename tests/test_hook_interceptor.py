"""Unit tests for Antigravity IDE PreToolUse Lifecycle Hook Interceptor."""

import io
import json
import sys

import pytest

from system_one_engine.agent.hook_interceptor import evaluate_tool_call, main


def test_interceptor_denies_destructive_shell_commands():
    """Verify destructive shell commands are hard blocked with decision: deny."""
    destructive_cmds = [
        "rm -rf /",
        "rm -rf *",
        "format C:",
        "diskpart",
        "shutdown /s /t 0",
        "drop database production;",
    ]
    for cmd in destructive_cmds:
        tool_call = {
            "name": "run_command",
            "args": {"CommandLine": cmd},
        }
        res = evaluate_tool_call(tool_call)
        assert res["decision"] == "deny", f"Failed to deny lethal command: {cmd}"
        assert "SystemOne Safety Guardrail Block" in res["reason"]


def test_interceptor_allows_safe_shell_commands():
    """Verify standard read-only and dev commands pass through with decision: allow."""
    safe_cmds = [
        "git status",
        "git log -n 5 --oneline",
        "uv run pytest",
        "dir",
        "echo testing",
        "python --version",
    ]
    for cmd in safe_cmds:
        tool_call = {
            "name": "run_command",
            "args": {"CommandLine": cmd},
        }
        res = evaluate_tool_call(tool_call)
        assert res["decision"] == "allow", f"Failed to allow safe command: {cmd}"


def test_interceptor_guards_protected_file_paths():
    """Verify modifications to critical paths require explicit user confirmation."""
    protected_files = [
        ".git/config",
        ".env",
        ".env.production",
        "C:\\Windows\\system.ini",
        "/etc/passwd",
    ]
    for p in protected_files:
        tool_call = {
            "name": "write_to_file",
            "args": {"TargetFile": p},
        }
        res = evaluate_tool_call(tool_call)
        assert res["decision"] == "ask", f"Failed to gate protected file: {p}"
        assert "protected path" in res["reason"]


def test_interceptor_allows_normal_project_file_edits():
    """Verify regular workspace source edits pass through freely."""
    tool_call = {
        "name": "replace_file_content",
        "args": {"TargetFile": "src/system_one_engine/core/contracts.py"},
    }
    res = evaluate_tool_call(tool_call)
    assert res["decision"] == "allow"


def test_interceptor_allows_non_target_tools():
    """Verify benign tools like view_file or list_dir pass immediately."""
    tool_call = {
        "name": "view_file",
        "args": {"AbsolutePath": "README.md"},
    }
    res = evaluate_tool_call(tool_call)
    assert res["decision"] == "allow"


def test_interceptor_main_cli_pipeline(monkeypatch):
    """Verify full stdin/stdout CLI pipeline execution."""
    payload = {
        "toolCall": {
            "name": "run_command",
            "args": {"CommandLine": "rm -rf /root"},
        },
        "stepIdx": 10,
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)

    main()

    output = json.loads(out.getvalue())
    assert output["decision"] == "deny"
    assert "SystemOne Safety Guardrail Block" in output["reason"]
