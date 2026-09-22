"""Tests for core/shortlist.py — embedding shortlist for high-cardinality choices."""

import numpy as np
import pytest

from system_one_engine.core.shortlist import (
    _check_k,
    _cosine,
    _embeddings,
    _rank,
    predict_shortlist,
    shortlist_choice,
)


# ---------------------------------------------------------------------------
# Deterministic embed_fn helpers
# ---------------------------------------------------------------------------


def _identity_embed(texts):
    """Each text → a one-hot unit vector in R^len(texts) space (for cos-sim = 1.0 on self)."""
    n = len(texts)
    out = np.zeros((n, max(n, 1)), dtype=np.float64)
    for i in range(n):
        out[i, i] = 1.0
    return out


def _uniform_embed(texts):
    """All texts get the same embedding → cosine sims all equal."""
    n = len(texts)
    out = np.ones((n, 8), dtype=np.float64)
    return out


def _biased_embed(texts):
    """First text has cos-sim 1.0 with query; rest have 0.0."""
    n = len(texts)
    out = np.zeros((n, n + 1), dtype=np.float64)
    for i in range(n):
        out[i, i + 1] = 1.0
    # Make query (index 0) similar only to option at index 1
    out[0, 1] = 1.0
    return out


# ---------------------------------------------------------------------------
# _check_k
# ---------------------------------------------------------------------------


def test_check_k_valid():
    assert _check_k(5) == 5


def test_check_k_bool_raises():
    with pytest.raises(ValueError):
        _check_k(True)


def test_check_k_zero_raises():
    with pytest.raises(ValueError):
        _check_k(0)


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------


def test_cosine_zero_query():
    q = np.zeros(4)
    docs = np.random.rand(3, 4)
    sims = _cosine(q, docs)
    assert np.all(sims == 0.0)


def test_cosine_self_similarity():
    v = np.array([1.0, 0.0, 0.0])
    docs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sims = _cosine(v, docs)
    assert pytest.approx(sims[0], abs=1e-6) == 1.0
    assert pytest.approx(sims[1], abs=1e-6) == 0.0


# ---------------------------------------------------------------------------
# _embeddings validation
# ---------------------------------------------------------------------------


def test_embeddings_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        _embeddings("not_a_function", ["a", "b"])


def test_embeddings_rejects_wrong_shape():
    def bad_embed(texts):
        return np.zeros((len(texts) + 1, 8))  # wrong row count

    with pytest.raises(ValueError, match="shape"):
        _embeddings(bad_embed, ["a", "b"])


def test_embeddings_accepts_valid():
    arr = _embeddings(_identity_embed, ["a", "b", "c"])
    assert arr.shape == (3, 3)


# ---------------------------------------------------------------------------
# shortlist_choice
# ---------------------------------------------------------------------------


def test_shortlist_choice_passthrough_when_k_gte_n():
    criteria = {"a": None, "b": None, "c": None}
    # k=10 >= 3, so all labels returned without calling embed_fn
    labels = shortlist_choice("state", criteria, embed_fn=_identity_embed, k=10)
    assert set(labels) == {"a", "b", "c"}


def test_shortlist_choice_reduces_k():
    criteria = {"a": None, "b": None, "c": None, "d": None, "e": None}
    labels = shortlist_choice("state", criteria, embed_fn=_identity_embed, k=2)
    assert len(labels) == 2
    assert all(lbl in criteria for lbl in labels)


def test_shortlist_choice_list_criteria():
    criteria = ["alpha", "beta", "gamma"]
    labels = shortlist_choice("state", criteria, embed_fn=_identity_embed, k=2)
    assert len(labels) == 2


def test_shortlist_choice_invalid_criteria_type():
    with pytest.raises(TypeError):
        shortlist_choice("state", "not_a_criteria", embed_fn=_identity_embed)


def test_shortlist_choice_empty_criteria():
    with pytest.raises(ValueError, match="at least one"):
        shortlist_choice("state", {}, embed_fn=_identity_embed)


def test_shortlist_choice_duplicate_labels():
    with pytest.raises(ValueError, match="duplicated"):
        shortlist_choice("state", ["a", "a", "b"], embed_fn=_identity_embed)


# ---------------------------------------------------------------------------
# predict_shortlist
# ---------------------------------------------------------------------------


class _MockAgent:
    """Minimal agent stub that returns a fixed result dict for predict_shortlist tests."""

    def predict(self, state, questions):
        result = {}
        for qid, qdef in questions.items():
            if isinstance(qdef, dict) and qdef.get("type") == "choice":
                crit = qdef.get("criteria", {})
                keys = list(crit.keys()) if isinstance(crit, dict) else list(crit)
                result[qid] = {"choice": keys[0], "probabilities": {k: 1.0 / len(keys) for k in keys}, "confidence": 0.5}
            else:
                result[qid] = {"confirmed": True, "probability": 0.8, "confidence": 0.6}
        return result


def test_predict_shortlist_returns_shortlist_key():
    questions = {
        "intent": {
            "type": "choice",
            "instructions": "pick one",
            "criteria": {f"opt{i}": None for i in range(10)},
        }
    }
    agent = _MockAgent()
    result = predict_shortlist(agent, "state", questions, embed_fn=_identity_embed, k=3)
    assert "shortlist" in result
    assert "intent" in result["shortlist"]


def test_predict_shortlist_passthrough_non_choice():
    questions = {
        "flag": {"type": "boolean", "instructions": "is it?"},
    }
    agent = _MockAgent()
    result = predict_shortlist(agent, "state", questions, embed_fn=_identity_embed, k=3)
    # Shortlist should be empty for non-choice questions
    assert result.get("shortlist", {}).get("flag") is None


def test_predict_shortlist_passthrough_when_k_gte_n():
    questions = {
        "small_choice": {
            "type": "choice",
            "instructions": "pick",
            "criteria": {"a": None, "b": None},
        }
    }
    agent = _MockAgent()
    result = predict_shortlist(agent, "state", questions, embed_fn=_identity_embed, k=10)
    # With k >= n, no embedding needed → passthrough=True in meta
    assert result["shortlist"]["small_choice"]["passthrough"] is True


def test_predict_shortlist_rejects_non_dict_questions():
    agent = _MockAgent()
    with pytest.raises(TypeError, match="dict"):
        predict_shortlist(agent, "state", ["not", "a", "dict"], embed_fn=_identity_embed)


def test_predict_shortlist_choice_missing_criteria_raises():
    agent = _MockAgent()
    questions = {"q": {"type": "choice", "instructions": "?"}}
    with pytest.raises(ValueError, match="no criteria"):
        predict_shortlist(agent, "state", questions, embed_fn=_identity_embed)
