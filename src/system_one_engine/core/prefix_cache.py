"""Bounded tokenized-prefix reuse for the System One prompt layout.

Adapted from laya-mlx (Apache-2.0, mizorewww/laya-mlx); see laya-mlx/NOTICE.

Why prefix caching is worth it:
  The question head ([CLS] <type> instruction [SEP] [MASK] opts [SEP]) is
  determined entirely by the question definition — it does not change when
  state changes. Tokenizing it once and caching it by a stable key avoids
  redundant tokenization on every call for repeated questions (e.g. the
  same guard_questions() applied to every incoming prompt).

  A cold tokenization of a 192-token head costs ~0.5 ms on CPU. At 100 q/s
  this becomes 50 ms/s in tokenization alone. PrefixCache eliminates that.

The cache operates over plain tokenized ids only — it does NOT cache encoder
hidden states or any model output. Predictions are always fresh.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from system_one_engine.core.prompt import QTYPES, build_prefix, render_options, serialize_state


@dataclass(frozen=True)
class _PreparedPrefix:
    """Immutable, hashable representation of a cached question prefix.

    Attributes:
        ids: Token id tuple for the question head (CLS, instructions, SEP, MASKs, opts, SEP).
        markers: Tuple of int positions pointing to each option's MASK token.
    """

    ids: tuple
    markers: tuple


class PrefixCache:
    """LRU-bounded cache of tokenized question prefixes.

    Keyed on a tuple derived from the tokenizer identity, the question type,
    instructions, and rendered option texts. Questions that appear in different
    ``system_one()`` calls with the same schema hit the cache; only the state
    tail is re-tokenized.

    The capacity defaults to 128 entries, which covers all built-in presets
    (24 questions across 5 preset functions) many times over while consuming
    < 1 MiB of memory at typical token-id array sizes.

    Attributes:
        capacity: Maximum number of cached prefixes before LRU eviction.
    """

    def __init__(self, capacity: int = 128) -> None:
        """Initialize the prefix cache.

        Args:
            capacity: Maximum cached entries (default 128).
        """
        if capacity < 1:
            raise ValueError(f"PrefixCache capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._entries: OrderedDict[tuple, _PreparedPrefix] = OrderedDict()

    def prepare(
        self,
        tok: Any,
        state: Any,
        questions: dict,
        max_len: int = 512,
        head_max_len: int = 192,
    ) -> tuple[list[dict], list[dict]]:
        """Tokenize state+questions, reusing cached prefixes for known questions.

        Args:
            tok: Tokenizer instance with callable interface returning {"input_ids": [...]}.
            state: Input state payload (str, dict, list, or None).
            questions: Dict mapping question IDs to question definition dicts.
            max_len: Maximum sequence length (default 512).
            head_max_len: Maximum token budget for the question head (default 192).

        Returns:
            tuple[list[dict], list[dict]]:
                - items: List of dicts with keys "ids", "markers", "qtype" for batching.
                - internal: List of internal question dicts {"t", "ins", "crit"}.

        Raises:
            ValueError: If questions is not a dict or a question has too many options.
        """
        if not isinstance(questions, dict):
            raise ValueError("questions must be a dictionary keyed by question id.")
        if not questions:
            return [], []

        # Tokenize the state once — shared across all questions
        state_text = serialize_state(state).replace(tok.mask_token, " ")
        state_ids: list[int] = tok(state_text, add_special_tokens=False)["input_ids"]

        items: list[dict] = []
        internal: list[dict] = []

        for qid, definition in questions.items():
            from system_one_engine.core.prompt import to_internal  # avoid circular at module level

            q = to_internal(definition)
            options = render_options(q)

            # Build a stable cache key that uniquely identifies this prefix
            cache_key = (
                id(tok),                    # tokenizer identity
                tok.cls_token_id,
                tok.sep_token_id,
                tok.mask_token_id,
                tok.mask_token,
                head_max_len,
                q["t"],
                q["ins"],
                tuple(options),
            )

            if cache_key not in self._entries:
                prefix_ids, markers = build_prefix(tok, q, head_max_len)
                self._entries[cache_key] = _PreparedPrefix(
                    ids=tuple(prefix_ids), markers=tuple(markers)
                )
                # Evict oldest entry if over capacity
                if len(self._entries) > self.capacity:
                    self._entries.popitem(last=False)

            # Promote to most-recently-used
            self._entries.move_to_end(cache_key)
            prefix = self._entries[cache_key]

            # Append state tail to cached prefix
            room = max(0, max_len - len(prefix.ids) - 1)
            full_ids = list(prefix.ids) + state_ids[:room] + [tok.sep_token_id]
            full_ids = full_ids[:max_len]

            # Drop markers that fell outside the window (shouldn't happen under normal budgets)
            valid_markers = [m for m in prefix.markers if m < max_len]

            if len(valid_markers) != len(options):
                raise ValueError(
                    f"Question {qid!r} has too many options for the token budget "
                    f"(max_len={max_len}, head_max_len={head_max_len})."
                )

            qtype_int = QTYPES.get(q["t"], 2)
            items.append({"ids": full_ids, "markers": valid_markers, "qtype": qtype_int})
            internal.append(q)

        return items, internal

    def clear(self) -> None:
        """Evict all cached prefixes."""
        self._entries.clear()

    @property
    def size(self) -> int:
        """Current number of cached prefixes."""
        return len(self._entries)
