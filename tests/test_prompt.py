"""Tests for core/prompt.py — laya-mlx aligned prompt layout utilities."""

import pytest
from system_one_engine.core.prompt import (
    build_prefix,
    build_sequence,
    render_criterion,
    render_options,
    serialize_state,
    to_internal,
)


# ---------------------------------------------------------------------------
# serialize_state
# ---------------------------------------------------------------------------


def test_serialize_state_string_passthrough():
    assert serialize_state("hello world") == "hello world"


def test_serialize_state_none():
    assert serialize_state(None) == ""


def test_serialize_state_dict():
    result = serialize_state({"key": "val"})
    assert '"key"' in result
    assert '"val"' in result


def test_serialize_state_list():
    result = serialize_state([1, 2, 3])
    assert "1" in result


# ---------------------------------------------------------------------------
# render_criterion
# ---------------------------------------------------------------------------


def test_render_criterion_string():
    assert render_criterion("urgent") == "urgent"


def test_render_criterion_dict():
    result = render_criterion({"severity": "high"})
    assert "severity" in result
    assert "high" in result


def test_render_criterion_number():
    assert render_criterion(42) == "42"


# ---------------------------------------------------------------------------
# render_options
# ---------------------------------------------------------------------------


def test_render_options_choice_no_description():
    q = {"t": "choice", "ins": "", "crit": {"a": None, "b": ""}}
    opts = render_options(q)
    assert opts == ["a", "b"]


def test_render_options_choice_with_description():
    q = {"t": "choice", "ins": "", "crit": {"cat": "a feline", "dog": "a canine"}}
    opts = render_options(q)
    assert opts[0] == "cat: a feline"
    assert opts[1] == "dog: a canine"


def test_render_options_score():
    q = {"t": "score", "ins": "", "crit": ["low", "medium", "high"]}
    opts = render_options(q)
    assert opts[0] == "level 0: low"
    assert opts[1] == "level 1: medium"
    assert opts[2] == "level 2: high"


def test_render_options_noul_defaults():
    q = {"t": "noul", "ins": "", "crit": None}
    opts = render_options(q)
    assert len(opts) == 2
    assert opts[0].startswith("false")
    assert opts[1].startswith("true")


def test_render_options_noul_custom():
    q = {"t": "noul", "ins": "", "crit": {"false": "no risk", "true": "high risk"}}
    opts = render_options(q)
    assert "no risk" in opts[0]
    assert "high risk" in opts[1]


# ---------------------------------------------------------------------------
# to_internal
# ---------------------------------------------------------------------------


def test_to_internal_choice_dict():
    q = to_internal({"type": "choice", "instructions": "pick one", "criteria": {"a": None, "b": "desc"}})
    assert q["t"] == "choice"
    assert q["ins"] == "pick one"
    assert "a" in q["crit"]


def test_to_internal_choice_list():
    q = to_internal({"type": "choice", "instructions": "pick", "criteria": ["x", "y"]})
    assert "x" in q["crit"]


def test_to_internal_boolean_maps_to_noul():
    q = to_internal({"type": "boolean", "instructions": "is it?"})
    assert q["t"] == "noul"


def test_to_internal_score():
    q = to_internal({"type": "score", "instructions": "rate it", "criteria": ["low", "high"]})
    assert q["t"] == "score"
    assert q["crit"] == ["low", "high"]


def test_to_internal_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown question type"):
        to_internal({"type": "unknown", "instructions": "?"})


def test_to_internal_missing_instructions_raises():
    with pytest.raises(ValueError, match="instructions"):
        to_internal({"type": "choice", "criteria": {"a": None}})


# ---------------------------------------------------------------------------
# build_prefix — uses a minimal stub tokenizer
# ---------------------------------------------------------------------------


class _StubTok:
    """Minimal tokenizer stub for prefix/sequence tests."""
    cls_token_id = 0
    sep_token_id = 1
    mask_token_id = 2
    mask_token = "[MASK]"
    pad_token_id = 3

    def __call__(self, text, add_special_tokens=True, **_kwargs):
        # Each character → one token id (mod 100), deterministic
        return {"input_ids": [(ord(c) % 100) + 10 for c in text]}


@pytest.fixture
def stub_tok():
    return _StubTok()


def test_build_prefix_returns_ids_and_markers(stub_tok):
    q = {"t": "choice", "ins": "pick one", "crit": {"yes": None, "no": None}}
    ids, markers = build_prefix(stub_tok, q)
    assert isinstance(ids, list)
    assert len(ids) > 0
    # Two options → two markers
    assert len(markers) == 2
    # Markers are valid positions within ids
    for m in markers:
        assert 0 <= m < len(ids)
        assert ids[m] == stub_tok.mask_token_id


def test_build_prefix_score_markers(stub_tok):
    q = {"t": "score", "ins": "rate this", "crit": ["low", "med", "high"]}
    ids, markers = build_prefix(stub_tok, q)
    assert len(markers) == 3


def test_build_prefix_boolean_markers(stub_tok):
    q = {"t": "noul", "ins": "is it?", "crit": None}
    ids, markers = build_prefix(stub_tok, q)
    assert len(markers) == 2


# ---------------------------------------------------------------------------
# build_sequence
# ---------------------------------------------------------------------------


def test_build_sequence_length_bounded(stub_tok):
    q = {"t": "choice", "ins": "a" * 10, "crit": {"a": None, "b": None, "c": None}}
    ids, markers = build_sequence(stub_tok, "some state text here", q, max_len=64)
    assert len(ids) <= 64
    # All markers inside window
    for m in markers:
        assert m < len(ids)


def test_build_sequence_state_appended(stub_tok):
    q = {"t": "noul", "ins": "?", "crit": None}
    ids_short, _ = build_sequence(stub_tok, "a", q)
    ids_long, _ = build_sequence(stub_tok, "a" * 200, q)
    # Longer state fills more of the budget
    assert len(ids_long) >= len(ids_short)


def test_build_sequence_truncate_left(stub_tok):
    q = {"t": "noul", "ins": "?", "crit": None}
    ids_right, _ = build_sequence(stub_tok, "x" * 300, q, truncate_left=False)
    ids_left, _ = build_sequence(stub_tok, "x" * 300, q, truncate_left=True)
    # Both should hit max_len (512) — lengths should be equal in this case
    assert len(ids_right) == len(ids_left)
