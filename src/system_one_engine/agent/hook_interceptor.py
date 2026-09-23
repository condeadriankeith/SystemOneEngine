"""Antigravity IDE Lifecycle Hook Interceptor.

Accepts Antigravity PreToolUse JSON payload on sys.stdin, performs sub-5ms safety
evaluation via SystemOne SafetyGuardrail, and emits decision JSON on sys.stdout.
"""

import json
import os
import re
import sys
from typing import Any

from system_one_engine.agent.contracts import OSAction, OSActionType, SafetyMode
from system_one_engine.agent.guardrails import SafetyGuardrail

# Path protection regex for file modifications
PROTECTED_PATH_PATTERNS = [
    re.compile(r"(\.git[/\\].*|\.env(\..*)?)$", re.IGNORECASE),
    re.compile(r"^[a-zA-Z]:[/\\](Windows|Program Files|Recovery)[/\\]?", re.IGNORECASE),
    re.compile(r"^[/\\](etc|boot|sys|proc|dev)[/\\]?", re.IGNORECASE),
]


def evaluate_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Evaluate an Antigravity tool call and return a decision dict.

    Args:
        tool_call: Dict containing 'name' and 'args'.

    Returns:
        Decision dict formatted for Antigravity hook runner:
        {"decision": "allow"|"deny"|"ask"|"force_ask", "reason": "..."}
    """
    name = tool_call.get("name", "")
    args = tool_call.get("args", {}) or {}

    guardrail = SafetyGuardrail()

    # 1. Shell commands via run_command
    if name == "run_command":
        command_line = args.get("CommandLine", "").strip()
        if not command_line:
            return {"decision": "allow"}

        action = OSAction(
            action_type=OSActionType.SHELL,
            target=command_line,
            params={"command": command_line},
            rationale="Antigravity IDE run_command execution",
        )

        safety_mode_str = os.environ.get("SYSTEM_ONE_SAFETY_MODE", "BALANCED").upper()
        mode = getattr(SafetyMode, safety_mode_str, SafetyMode.BALANCED)

        result = guardrail.check(action, mode)

        if not result.is_safe:
            return {
                "decision": "deny",
                "reason": f"SystemOne Safety Guardrail Block: {result.reason}",
            }

        if result.requires_confirmation:
            return {
                "decision": "ask",
                "reason": f"SystemOne Pre-Execution Gate: {result.reason}",
            }

        return {
            "decision": "allow",
            "reason": f"Cleared by SystemOne Guardrail (confidence: {result.confidence * 100:.1f}%)",
        }

    # 2. File modifications via write_to_file or replace_file_content
    if name in ("write_to_file", "replace_file_content", "multi_replace_file_content"):
        target_file = args.get("TargetFile", "").strip()
        if target_file:
            norm_path = os.path.normpath(target_file)
            clean_slash = target_file.replace("\\", "/")
            for pattern in PROTECTED_PATH_PATTERNS:
                if pattern.search(norm_path) or pattern.search(clean_slash):
                    return {
                        "decision": "ask",
                        "reason": f"SystemOne File Guardrail: Modifying protected path '{target_file}' requires explicit user confirmation.",
                    }

        return {"decision": "allow"}

    # Default to allow for all other tools
    return {"decision": "allow"}


def main() -> None:
    """CLI entry point reading from stdin and writing to stdout."""
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            # No payload received, allow safely
            print(json.dumps({"decision": "allow"}))
            return

        payload = json.loads(raw_input)
        tool_call = payload.get("toolCall", {})

        decision = evaluate_tool_call(tool_call)
        print(json.dumps(decision))
    except Exception as exc:
        # Fail safe: allow with warning if hook crashes
        fallback = {
            "decision": "allow",
            "reason": f"SystemOne hook fallback on error: {exc}",
        }
        print(json.dumps(fallback))


if __name__ == "__main__":
    main()
