"""Embedding shortlist for high-cardinality choice questions.

Adapted from laya-mlx (Apache-2.0, mizorewww/laya-mlx); see laya-mlx/NOTICE.
MLX dependency removed; uses NumPy only.

Why shortlisting matters:
  Choice options share one ``head_max_len`` token budget (default 192 tokens).
  With 64 options each option gets ~3 tokens — not enough to express semantics.
  ``predict_shortlist`` embeds the state and each option with a caller-supplied
  ``embed_fn``, keeps the top ``k`` by cosine similarity, and runs a single
  ``system_one()`` call on that reduced criteria set.

  This module does NOT change how the model scores options; it only pre-filters
  which options are presented to the model. The coarse→fine pattern is the same
  one described in the upstream Laya README.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from system_one_engine.core.prompt import render_options, serialize_state

DEFAULT_SHORTLIST_K = 20


def shortlist_choice(
    state: Any,
    criteria: Any,
    embed_fn: Callable[[Sequence[str]], Any],
    k: int = DEFAULT_SHORTLIST_K,
    *,
    instructions: Optional[str] = None,
) -> List[Any]:
    """Return the top-``k`` choice labels ranked by cosine similarity to ``state``.

    ``embed_fn`` maps a list of strings to an array of shape ``(len(texts), dim)``.
    It is called once with the query text first and then one string per option.

    When ``k >= len(labels)`` all labels are returned in original order and
    ``embed_fn`` is NOT called.

    Args:
        state: Input state payload — string, dict, list, or None.
        criteria: Choice criteria — a dict (label→description) or a list of labels.
        embed_fn: Callable returning an embedding array of shape (N, dim).
        k: Maximum number of labels to keep.
        instructions: Optional instruction text prepended to the query string.

    Returns:
        List[Any]: Top-k label keys in similarity-ranked order.
    """
    labels, _scores, _passthrough, _n = _rank(state, criteria, embed_fn, k, instructions)
    return labels


def predict_shortlist(
    agent: Any,
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    embed_fn: Callable[[Sequence[str]], Any],
    k: int = DEFAULT_SHORTLIST_K,
    **predict_kwargs: Any,
) -> Dict[str, Any]:
    """Shortlist each choice question then call ``system_one()`` / ``predict()`` once.

    Non-choice questions are forwarded unchanged. A choice whose label count is
    <= k is forwarded unchanged and does not call ``embed_fn``. The caller's
    ``questions`` dict is not mutated.

    The returned dict is the model result plus a ``shortlist`` entry. Probabilities
    on a shortlisted choice are over the kept labels only. ``shortlist[qid]`` holds
    ``labels`` (rank order), ``scores`` (cosine, or None when nothing was dropped),
    ``k``, ``n``, and ``passthrough``.

    Args:
        agent: Any object with a ``predict`` or ``system_one`` method.
        state: Input state payload.
        questions: Dict mapping question IDs to question definition dicts.
        embed_fn: Callable returning an embedding array of shape (N, dim).
        k: Maximum labels to keep per choice question.
        **predict_kwargs: Forwarded to ``predict`` / ``system_one``.

    Returns:
        Dict: Model result dict augmented with a ``"shortlist"`` metadata key.

    Raises:
        TypeError: If questions is not a dict or agent lacks predict/system_one.
        ValueError: If a choice question is missing criteria.
    """
    if not isinstance(questions, dict):
        raise TypeError("questions must be a dict of question id → definition")
    checked_k = _check_k(k)
    reduced: Dict[str, Any] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    for qid, qdef in questions.items():
        if not isinstance(qdef, dict) or qdef.get("type") != "choice":
            reduced[qid] = qdef
            continue
        if "criteria" not in qdef:
            raise ValueError(f"Question {qid!r} is a choice but has no criteria.")

        labels, scores, passthrough, n = _rank(
            state, qdef["criteria"], embed_fn, checked_k, qdef.get("instructions")
        )
        meta[qid] = {
            "labels": list(labels),
            "scores": scores,
            "k": checked_k,
            "n": n,
            "passthrough": passthrough,
        }

        if passthrough:
            reduced[qid] = qdef
        else:
            updated = dict(qdef)
            updated["criteria"] = _subset_criteria(qdef["criteria"], labels)
            reduced[qid] = updated

    result = _call_predict(agent, state, reduced, **predict_kwargs)
    if not isinstance(result, dict):
        raise TypeError(
            f"predict/system_one must return a dict, got {type(result).__name__}"
        )

    out = dict(result)
    out["shortlist"] = meta
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rank(
    state: Any,
    criteria: Any,
    embed_fn: Callable,
    k: int,
    instructions: Optional[str],
) -> tuple:
    """Rank criteria labels by cosine similarity to the state query.

    Returns:
        tuple: (labels, scores_or_None, passthrough_bool, total_n)
    """
    checked_k = _check_k(k)
    items = _criteria_items(criteria)
    n = len(items)
    keys = [key for key, _value in items]

    # If we want all labels, skip embedding entirely
    if checked_k >= n:
        return list(keys), None, True, n

    query = _query_text(state, instructions)
    matrix = _embeddings(embed_fn, [query] + _option_texts(items))
    sims = _cosine(matrix[0], matrix[1:])
    order = np.argsort(-sims, kind="mergesort")[:checked_k]
    labels = [keys[int(i)] for i in order]
    scores = [float(sims[int(i)]) for i in order]
    return labels, scores, False, n


def _check_k(k: int) -> int:
    """Validate k is a positive integer."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return k


def _criteria_items(criteria: Any) -> list:
    """Normalise criteria to a list of (key, value) pairs with uniqueness check."""
    if isinstance(criteria, dict):
        items = list(criteria.items())
    elif isinstance(criteria, list):
        items = [(item, None) for item in criteria]
    else:
        raise TypeError(
            f"choice criteria must be a dict or list, got {type(criteria).__name__}"
        )
    if not items:
        raise ValueError("choice criteria must contain at least one option.")

    seen: set = set()
    for key, _value in items:
        if key in seen:
            raise ValueError(f"choice criteria label {key!r} is duplicated.")
        seen.add(key)
    return items


def _option_texts(items: list) -> List[str]:
    """Render each choice option as a text string for embedding."""
    crit = {key: value for key, value in items}
    rendered = render_options({"t": "choice", "ins": "", "crit": crit})
    texts = [piece if isinstance(piece, str) else str(piece) for piece in rendered]
    if len(texts) != len(items):
        raise ValueError("Could not render every choice option.")
    return texts


def _query_text(state: Any, instructions: Optional[str]) -> str:
    """Build the query text by prepending instructions to the state body."""
    body = serialize_state(state)
    if instructions is None or instructions == "":
        return body
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    return f"{instructions}\n{body}"


def _subset_criteria(criteria: Any, labels: List[Any]) -> Any:
    """Return a subset of criteria containing only the given labels."""
    if isinstance(criteria, dict):
        return {label: criteria[label] for label in labels}
    return list(labels)


def _embeddings(embed_fn: Callable, texts: Sequence[str]) -> np.ndarray:
    """Call ``embed_fn`` and validate the returned embedding array."""
    if not callable(embed_fn):
        raise TypeError("embed_fn must be callable.")
    raw = embed_fn(list(texts))

    # Accept PyTorch tensors gracefully
    if hasattr(raw, "detach"):
        raw = raw.detach().float().cpu().numpy()

    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != len(texts) or arr.shape[1] < 1:
        raise ValueError(
            f"embed_fn must return an array of shape ({len(texts)}, dim), got {tuple(arr.shape)}"
        )
    return np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)


def _cosine(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between ``query`` and each row of ``docs``."""
    qn = float(np.linalg.norm(query))
    dn = np.linalg.norm(docs, axis=1)
    sims = np.zeros(docs.shape[0], dtype=np.float64)
    if qn == 0.0:
        return sims
    denom = dn * qn
    ok = denom > 0.0
    if np.any(ok):
        sims[ok] = docs[ok] @ query / denom[ok]
    return sims


def _call_predict(agent: Any, state: Any, questions: dict, **kwargs: Any) -> dict:
    """Call the agent's predict or system_one method."""
    fn = getattr(agent, "predict", None)
    if fn is None:
        fn = getattr(agent, "system_one", None)
    if fn is None:
        raise TypeError("agent must provide a predict or system_one method.")
    return fn(state, questions, **kwargs)
