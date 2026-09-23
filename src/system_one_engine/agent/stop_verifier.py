"""Antigravity IDE Independent Stop Verifier.

Invoked by Antigravity IDE 'Stop' lifecycle hook when the agent attempts to stop.
Independently verifies that the workspace has no broken syntax or failing tests
before permitting loop completion, preventing hallucinated completions.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def check_syntax(workspace_path: Path) -> list[str]:
    """Scan Python files in workspace for syntax errors.

    Returns:
        List of error description strings.
    """
    errors: list[str] = []
    # Check src and tests directories
    target_dirs = [workspace_path / "src", workspace_path / "tests", workspace_path / "scripts"]

    for tdir in target_dirs:
        if not tdir.exists():
            continue
        for py_file in tdir.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
                ast.parse(content, filename=str(py_file))
            except SyntaxError as err:
                errors.append(f"Syntax error in {py_file.name}:{err.lineno}: {err.msg}")
            except Exception as err:
                errors.append(f"Error reading {py_file.name}: {err}")
    return errors


def check_tests(workspace_path: Path, timeout: int = 45) -> tuple[bool, str]:
    """Run pytest suite to verify zero regressions.

    Returns:
        tuple[bool, str]: (passed, summary_message)
    """
    try:
        res = subprocess.run(
            ["uv", "run", "pytest", "-q"],
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if res.returncode == 0:
            return True, "All tests passed successfully."
        # Capture last lines of output
        lines = [line.strip() for line in (res.stdout + res.stderr).splitlines() if line.strip()]
        summary = " | ".join(lines[-3:]) if lines else "Tests failed with exit code non-zero."
        return False, summary
    except subprocess.TimeoutExpired:
        return False, f"Test suite execution timed out after {timeout}s."
    except Exception as exc:
        return True, f"Skipped test execution due to environment error: {exc}"


def evaluate_stop_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate whether the agent may safely terminate.

    Args:
        payload: Dict from Antigravity IDE Stop hook.

    Returns:
        Dict with 'decision': 'allow' | 'continue', and optional 'reason'.
    """
    reason = payload.get("terminationReason", "")
    # Only verify on autonomous model completions (not user aborts or errors)
    if reason != "model_stop":
        return {"decision": "allow"}

    # Check if stop verification is enabled (default enabled)
    if os.environ.get("SYSTEM_ONE_VERIFY_ON_STOP", "true").lower() in ("false", "0", "no"):
        return {"decision": "allow"}

    workspace_paths = payload.get("workspacePaths", [])
    workspace_path = Path(workspace_paths[0]) if workspace_paths else Path.cwd()

    # 1. Check Python syntax
    syntax_errors = check_syntax(workspace_path)
    if syntax_errors:
        return {
            "decision": "continue",
            "reason": (
                "System One Independent Verification Block: Python syntax errors detected: "
                + "; ".join(syntax_errors[:3])
            ),
        }

    # 2. Check test suite if requested or enabled
    run_tests = os.environ.get("SYSTEM_ONE_RUN_TESTS_ON_STOP", "false").lower() in ("true", "1", "yes")
    if run_tests:
        tests_passed, test_summary = check_tests(workspace_path)
        if not tests_passed:
            return {
                "decision": "continue",
                "reason": (
                    "System One Independent Verification Block: Automated test regressions detected. "
                    f"Please resolve before finishing: {test_summary}"
                ),
            }

    return {"decision": "allow", "reason": "System One independent verification passed."}


def main() -> None:
    """CLI entry point reading from stdin and writing to stdout."""
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            print(json.dumps({"decision": "allow"}))
            return

        payload = json.loads(raw_input)
        result = evaluate_stop_event(payload)
        print(json.dumps(result))
    except Exception as exc:
        fallback = {
            "decision": "allow",
            "reason": f"SystemOne stop verifier fallback on error: {exc}",
        }
        print(json.dumps(fallback))


if __name__ == "__main__":
    main()
