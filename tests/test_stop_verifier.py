"""Unit tests for Antigravity IDE Stop Lifecycle Hook Verifier."""

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

from system_one_engine.agent.stop_verifier import (
    check_syntax,
    evaluate_stop_event,
    main,
)


def test_stop_verifier_ignores_non_model_stops():
    """Non-model stops (user cancels, timeout) should pass immediately."""
    payload = {"terminationReason": "user_cancelled"}
    res = evaluate_stop_event(payload)
    assert res["decision"] == "allow"


def test_stop_verifier_passes_on_clean_workspace(tmp_path):
    """Clean workspace with valid syntax should be approved."""
    src = tmp_path / "src"
    src.mkdir()
    valid_py = src / "module.py"
    valid_py.write_text("def hello() -> str:\n    return 'world'\n", encoding="utf-8")

    payload = {
        "terminationReason": "model_stop",
        "workspacePaths": [str(tmp_path)],
    }
    res = evaluate_stop_event(payload)
    assert res["decision"] == "allow"


def test_stop_verifier_blocks_on_syntax_error(tmp_path):
    """Broken Python syntax must block termination with decision: continue."""
    src = tmp_path / "src"
    src.mkdir()
    broken_py = src / "broken.py"
    broken_py.write_text("def syntax_error(\n", encoding="utf-8")

    payload = {
        "terminationReason": "model_stop",
        "workspacePaths": [str(tmp_path)],
    }
    res = evaluate_stop_event(payload)
    assert res["decision"] == "continue"
    assert "syntax errors detected" in res["reason"]


def test_stop_verifier_blocks_when_tests_fail(tmp_path, monkeypatch):
    """When SYSTEM_ONE_RUN_TESTS_ON_STOP=true and tests fail, termination is blocked."""
    monkeypatch.setenv("SYSTEM_ONE_RUN_TESTS_ON_STOP", "true")

    with patch("system_one_engine.agent.stop_verifier.check_tests", return_value=(False, "2 failed")):
        payload = {
            "terminationReason": "model_stop",
            "workspacePaths": [str(tmp_path)],
        }
        res = evaluate_stop_event(payload)
        assert res["decision"] == "continue"
        assert "Automated test regressions detected" in res["reason"]


def test_stop_verifier_main_cli(monkeypatch, tmp_path):
    """Verify stop_verifier main() CLI handling."""
    payload = {
        "terminationReason": "model_stop",
        "workspacePaths": [str(tmp_path)],
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)

    main()

    output = json.loads(out.getvalue())
    assert output["decision"] == "allow"
