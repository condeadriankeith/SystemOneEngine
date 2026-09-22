"""Tests for core/presets.py — production question preset functions."""

import pytest
from system_one_engine.core.presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------

VALID_TYPES = {"choice", "score", "boolean", "noul"}


def _assert_question_valid(qid: str, qdef: dict) -> None:
    """Assert that a single question definition is well-formed."""
    assert isinstance(qdef, dict), f"Question {qid!r} must be a dict"
    assert "type" in qdef, f"Question {qid!r} missing 'type'"
    assert qdef["type"] in VALID_TYPES, f"Question {qid!r} has invalid type {qdef['type']!r}"
    assert "instructions" in qdef, f"Question {qid!r} missing 'instructions'"
    assert isinstance(qdef["instructions"], str), f"Question {qid!r} instructions must be str"
    assert qdef["instructions"].strip(), f"Question {qid!r} instructions must not be empty"

    if qdef["type"] == "choice":
        assert "criteria" in qdef, f"Choice question {qid!r} missing 'criteria'"
        assert isinstance(qdef["criteria"], (dict, list)), f"{qid!r} criteria must be dict or list"
        assert qdef["criteria"], f"{qid!r} criteria must not be empty"

    elif qdef["type"] == "score":
        assert "criteria" in qdef, f"Score question {qid!r} missing 'criteria'"
        assert isinstance(qdef["criteria"], list), f"{qid!r} score criteria must be a list"
        assert len(qdef["criteria"]) >= 2, f"{qid!r} score requires at least 2 levels"


def _assert_preset_valid(questions: dict, min_questions: int = 1) -> None:
    """Assert that an entire preset dict is valid."""
    assert isinstance(questions, dict), "Preset must return a dict"
    assert len(questions) >= min_questions, f"Expected >= {min_questions} questions, got {len(questions)}"
    for qid, qdef in questions.items():
        _assert_question_valid(qid, qdef)


# ---------------------------------------------------------------------------
# triage_questions
# ---------------------------------------------------------------------------


def test_triage_questions_structure():
    q = triage_questions()
    _assert_preset_valid(q, min_questions=5)


def test_triage_questions_has_intent_choice():
    q = triage_questions()
    assert "intent" in q
    assert q["intent"]["type"] == "choice"
    assert len(q["intent"]["criteria"]) >= 4


def test_triage_questions_has_booleans():
    q = triage_questions()
    for key in ("is_urgent", "refund_requested", "churn_risk"):
        assert key in q, f"Missing {key!r}"
        assert q[key]["type"] == "boolean"


def test_triage_questions_has_score():
    q = triage_questions()
    assert "frustration" in q
    assert q["frustration"]["type"] == "score"
    assert len(q["frustration"]["criteria"]) >= 2


# ---------------------------------------------------------------------------
# email_questions
# ---------------------------------------------------------------------------


def test_email_questions_structure():
    q = email_questions()
    _assert_preset_valid(q, min_questions=4)


def test_email_questions_custom_categories():
    custom = {"billing": "invoices", "tech": "bugs"}
    q = email_questions(categories=custom)
    assert q["category"]["criteria"] is custom


def test_email_questions_has_spam_and_phishing():
    q = email_questions()
    for key in ("is_spam", "is_phishing"):
        assert key in q
        assert q[key]["type"] == "boolean"


def test_email_questions_has_urgency_score():
    q = email_questions()
    assert "urgency" in q
    assert q["urgency"]["type"] == "score"


# ---------------------------------------------------------------------------
# guard_questions
# ---------------------------------------------------------------------------


def test_guard_questions_structure():
    q = guard_questions()
    _assert_preset_valid(q, min_questions=4)


def test_guard_questions_has_jailbreak_and_injection():
    q = guard_questions()
    for key in ("jailbreak", "prompt_injection", "sensitive_data"):
        assert key in q
        assert q[key]["type"] == "boolean"


def test_guard_questions_has_harm_severity_score():
    q = guard_questions()
    assert "harm_severity" in q
    assert q["harm_severity"]["type"] == "score"
    assert len(q["harm_severity"]["criteria"]) >= 3


def test_guard_questions_has_topic_choice():
    q = guard_questions()
    assert "topic" in q
    assert q["topic"]["type"] == "choice"


# ---------------------------------------------------------------------------
# moderation_questions
# ---------------------------------------------------------------------------


def test_moderation_questions_structure():
    q = moderation_questions()
    _assert_preset_valid(q, min_questions=4)


def test_moderation_questions_booleans():
    q = moderation_questions()
    for key in ("toxic", "harassment", "threat", "spam"):
        assert key in q
        assert q[key]["type"] == "boolean"


def test_moderation_questions_severity_score():
    q = moderation_questions()
    assert "severity" in q
    assert q["severity"]["type"] == "score"


# ---------------------------------------------------------------------------
# router_questions
# ---------------------------------------------------------------------------


def test_router_questions_structure():
    q = router_questions()
    _assert_preset_valid(q, min_questions=3)


def test_router_questions_has_difficulty_score():
    q = router_questions()
    assert "difficulty" in q
    assert q["difficulty"]["type"] == "score"


def test_router_questions_has_domain_choice():
    q = router_questions()
    assert "domain" in q
    assert q["domain"]["type"] == "choice"
    assert len(q["domain"]["criteria"]) >= 4


def test_router_questions_has_needs_tools_boolean():
    q = router_questions()
    assert "needs_tools" in q
    assert q["needs_tools"]["type"] == "boolean"


# ---------------------------------------------------------------------------
# Cross-preset: preset returns a fresh dict each call (not shared singleton)
# ---------------------------------------------------------------------------


def test_triage_questions_returns_fresh_copy():
    q1 = triage_questions()
    q2 = triage_questions()
    q1["__extra__"] = True
    assert "__extra__" not in q2


def test_email_questions_returns_fresh_copy():
    q1 = email_questions()
    q2 = email_questions()
    q1["__extra__"] = True
    assert "__extra__" not in q2
