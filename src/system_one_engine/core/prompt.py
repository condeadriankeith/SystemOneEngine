"""Canonical prompt layout utilities for System One Decision Engine.

Adapted from laya-mlx (Apache-2.0, mizorewww/laya-mlx); see laya-mlx/NOTICE.

Provides:
- serialize_state: Convert state payloads (str/dict/list) to a single string.
- render_criterion: Render one criterion value as text.
- render_options: Render the ordered option list for any question type.
- build_prefix: Build the question-only token prefix (before state tokens).
- build_sequence: Assemble the full token sequence:
    [CLS] <type> instruction [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]

Why this layout?
  Placing option MASK tokens inside the *head* (before state) bounds their
  positions and allows prefix caching: a question seen with different states
  only needs the state tail re-tokenized. Marker positions (list of int) point
  to the MASK token index for each option; the classifier reads those positions
  and produces one logit per option without any autoregressive generation.
"""

import json
from typing import Any, Dict, List, Optional, Union


# Internal question-type names used in prompt rendering.
# External Pydantic API keeps "boolean"; internal prompt uses "noul" to match
# upstream Laya vocabulary (necessary when using laya checkpoints directly).
QTYPES: Dict[str, int] = {"choice": 0, "score": 1, "noul": 2, "boolean": 2}
QTYPE_NAMES: Dict[int, str] = {0: "choice", 1: "score", 2: "noul"}


def serialize_state(state: Union[str, dict, list, None]) -> str:
    """Convert a state payload to a plain string for tokenization.

    Strings pass through unchanged. Structured payloads (dict, list) are
    serialized as compact JSON so they can be tokenized without a Python repr.

    Args:
        state: Raw state — a string, dictionary, list, or None.

    Returns:
        str: Plain text representation of the state.
    """
    if isinstance(state, str):
        return state
    if state is None:
        return ""
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value: Any) -> str:
    """Render one criterion value as a clean text string.

    Strings pass through. Structured values (dict, list, number) become compact
    JSON to avoid Python repr leaking into prompts (e.g. {'desc': ...}).

    Args:
        value: A criterion value — string, dict, list, or primitive.

    Returns:
        str: Rendered text for the criterion.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: Dict) -> List[str]:
    """Render the ordered option text list for a given internal question dict.

    For choice: labels with optional descriptions ("label: description").
    For score: "level 0: …", "level 1: …", ...
    For noul/boolean: always ["false: …", "true: …"] in that order.

    Args:
        q: Internal question dict with keys "t" (type), "ins" (instructions),
           and optionally "crit" (criteria).

    Returns:
        List[str]: Rendered option texts in label/level order.
    """
    t, crit = q["t"], q.get("crit")

    if t == "choice":
        # Only None/"" means "no description"; 0 and False are valid criteria values.
        return [
            k if v is None or v == "" else f"{k}: {render_criterion(v)}"
            for k, v in crit.items()
        ]

    if t == "score":
        return [f"level {i}: {render_criterion(c)}" for i, c in enumerate(crit)]

    # noul / boolean: always [false, true]
    crit = crit or {}
    false_crit = crit.get("false")
    true_crit = crit.get("true")
    return [
        "false: "
        + (
            render_criterion(false_crit)
            if false_crit not in (None, "")
            else "no, the statement does not hold"
        ),
        "true: "
        + (
            render_criterion(true_crit)
            if true_crit not in (None, "")
            else "yes, the statement holds"
        ),
    ]


def build_prefix(
    tok: Any,
    q: Dict,
    head_max_len: int = 192,
    option_order: Optional[List[int]] = None,
) -> tuple[list[int], list[int]]:
    """Build the question-only token prefix, before state tokens.

    Format: [CLS] <type> instruction [SEP] [MASK] opt0_tokens [MASK] opt1_tokens … [SEP]

    Marker positions returned are the indices (within the prefix id list) of
    each MASK token, one per option. The classifier reads the hidden state at
    each marker position and produces one logit per option.

    Args:
        tok: Tokenizer with attributes cls_token_id, sep_token_id, mask_token_id,
             mask_token, and a callable interface returning {"input_ids": [...]}.
        q: Internal question dict: {"t": type_str, "ins": instruction_str, "crit": ...}.
        head_max_len: Maximum token budget for the question head (default 192).
        option_order: Optional permutation of option indices (for position-bias studies).

    Returns:
        tuple[list[int], list[int]]: (token_ids, marker_positions)
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))

    # Sanitize instruction: strip any mask token to avoid collisions
    ins = str(q["ins"]).replace(mask_tok, " ")

    # Head: "<type> question: <instruction>"
    head_text = f"{q['t']} question: {ins}"
    head_ids: list[int] = tok(head_text, add_special_tokens=False)["input_ids"]

    # Each option: [MASK] + truncated option text tokens (max 48 tokens each)
    opt_ids: list[list[int]] = []
    for i in order:
        opt_text = " " + opts[i].replace(mask_tok, " ")
        opt_token_ids: list[int] = tok(opt_text, add_special_tokens=False)["input_ids"][:48]
        opt_ids.append([tok.mask_token_id] + opt_token_ids)

    # Ensure options fit in budget; truncate per-option if needed
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)

    # Truncate head to remaining budget (min 8 tokens)
    head_ids = head_ids[: max(8, opt_budget)]

    # Assemble: [CLS] head [SEP] [MASK] opt0 [MASK] opt1 … [SEP]
    ids: list[int] = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers: list[int] = []
    for o in opt_ids:
        markers.append(len(ids))   # Position of [MASK] for this option
        ids.extend(o)
    ids.append(tok.sep_token_id)

    return ids, markers


def build_sequence(
    tok: Any,
    state: Union[str, dict, list, None],
    q: Dict,
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Optional[List[int]] = None,
    truncate_left: bool = False,
) -> tuple[list[int], list[int]]:
    """Assemble the full input sequence for one question.

    Format:
        [CLS] <type> instruction [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]

    State tokens fill the remaining budget after the question head. When
    ``truncate_left=True`` the *tail* of the state is kept (useful for
    streaming conversations where recent context is most relevant).

    Args:
        tok: Tokenizer with callable interface returning {"input_ids": [...]}.
        state: Input state — string, dict, list, or None.
        q: Internal question dict.
        max_len: Maximum total sequence length (default 512).
        head_max_len: Maximum token budget for the question prefix (default 192).
        option_order: Optional permutation of option indices.
        truncate_left: If True, keep the right (recent) end of the state.

    Returns:
        tuple[list[int], list[int]]: (token_ids, valid_marker_positions)
            marker_positions contains only those markers that fit within max_len.
    """
    ids, markers = build_prefix(tok, q, head_max_len, option_order)

    # State tokens fill the remaining budget (minus the trailing [SEP])
    room = max(0, max_len - len(ids) - 1)
    state_text = serialize_state(state).replace(tok.mask_token, " ")
    state_ids: list[int] = tok(state_text, add_special_tokens=False)["input_ids"]

    state_ids = state_ids[-room:] if truncate_left else state_ids[:room]
    ids = ids + state_ids + [tok.sep_token_id]

    # Clamp to max_len and filter markers that fell outside the window
    return ids[:max_len], [m for m in markers if m < max_len]


def to_internal(qdef: dict) -> dict:
    """Validate and normalise a public question definition dict to internal form.

    Public form (matches our Pydantic API + laya-mlx API):
        {
            "type": "choice" | "score" | "boolean" | "noul",
            "instructions": "<string>",
            "criteria": <dict for choice, list for score/noul, None for bare noul>
        }

    Internal form used by build_prefix / build_sequence / render_options:
        {"t": <type_str>, "ins": "<instructions>", "crit": <criteria>}

    Args:
        qdef: Public question definition dictionary.

    Returns:
        dict: Internal question dict with keys "t", "ins", "crit".

    Raises:
        ValueError: If the type is unknown or required fields are missing.
    """
    if not isinstance(qdef, dict):
        raise ValueError("Each question must be a dictionary.")

    kind = qdef.get("type")
    # Map "boolean" alias → "noul" for internal prompt rendering
    if kind == "boolean":
        kind = "noul"

    if kind not in ("choice", "score", "noul"):
        raise ValueError(
            f"Unknown question type {kind!r}; expected 'choice', 'score', or 'boolean'."
        )

    if "instructions" not in qdef:
        raise ValueError("Question is missing required 'instructions' field.")

    criteria = qdef.get("criteria")
    instructions = qdef["instructions"]
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions)

    # Type-specific criteria validation
    if kind == "choice":
        if isinstance(criteria, list):
            if not all(isinstance(c, str) for c in criteria):
                raise ValueError("Choice labels must be strings.")
            if len(set(criteria)) != len(criteria):
                raise ValueError("Choice labels must be unique.")
            criteria = dict.fromkeys(criteria)
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("Choice criteria must be a non-empty dictionary or list.")
        if not all(isinstance(k, str) for k in criteria):
            raise ValueError("Choice labels must be strings.")

    elif kind == "score":
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("Score criteria must be a non-empty list.")

    elif criteria is not None and not isinstance(criteria, dict):
        raise ValueError("Boolean/noul criteria must be a dict with 'false'/'true' keys.")

    return {"t": kind, "ins": instructions, "crit": criteria}
