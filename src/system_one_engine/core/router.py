"""Lightweight model router for System One Decision Engine.

Adapted from laya-mlx (Apache-2.0, mizorewww/laya-mlx); see laya-mlx/NOTICE.

Why a router?
  laya-mlx's ``Router`` class manages a registry of loaded checkpoints, routes
  requests to the correct model based on language and question type, and
  lazy-loads models on first access with an LRU eviction policy.

  Our version adapts this for System One's two adapter backends (ONNX for
  CPU-fast inference, Ollama for zero-shot / offline fallback). The router:
    - Holds an LRU-bounded dict of named adapter instances.
    - Resolves the ``model`` keyword to the correct adapter.
    - Provides ``predict()`` / ``system_one()`` methods compatible with the
      laya-mlx agent API so that ``shortlist.predict_shortlist()`` works without
      modification.

  The router does NOT perform language detection routing (laya's Latin vs.
  non-Latin checkpoint selection) because our ONNX checkpoint is multilingual
  by design. Language routing can be added in a future iteration.
"""

import logging
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Canonical model name constants matching laya-mlx DEFAULT_MODELS pattern
MODEL_ONNX = "onnx"
MODEL_OLLAMA = "ollama"
DEFAULT_MODEL = MODEL_ONNX


class Router:
    """LRU-managed adapter registry with laya-mlx compatible predict/system_one API.

    Lazily initializes adapter instances on first use and evicts the
    least-recently-used adapter when the registry exceeds ``max_loaded``.

    Usage (matches laya-mlx Router API):
        from system_one_engine import Router, triage_questions
        router = Router()
        result = router.predict(state, triage_questions())
        result = router.predict(state, questions, model="ollama")

    Attributes:
        max_loaded: Maximum number of concurrently loaded adapter instances.
        default_model: Default adapter name ("onnx" or "ollama").
    """

    def __init__(
        self,
        adapters: Optional[dict[str, Any]] = None,
        max_loaded: int = 4,
        default_model: str = DEFAULT_MODEL,
    ) -> None:
        """Initialize the Router.

        Args:
            adapters: Optional pre-built adapter dict mapping name → adapter instance.
                      If not provided, adapters are created lazily on first use.
            max_loaded: Maximum concurrently loaded adapters (default: 4).
            default_model: Default model key to use when ``model`` kwarg is omitted.
        """
        if max_loaded < 1:
            raise ValueError(f"max_loaded must be >= 1, got {max_loaded}")
        self.max_loaded = max_loaded
        self.default_model = default_model
        # LRU-ordered registry: key → adapter instance
        self._registry: OrderedDict[str, Any] = OrderedDict()

        if adapters is not None:
            for name, adapter in adapters.items():
                self._register(name, adapter)

    # -----------------------------------------------------------------------
    # Public API — mirrors laya-mlx Router interface
    # -----------------------------------------------------------------------

    def predict(self, state: Any, questions: dict, model: Optional[str] = None, **kwargs: Any) -> dict:
        """Route questions to the named adapter and call system_one().

        This is the primary laya-mlx compatible entry point. ``predict`` is
        an alias of ``system_one`` on the router itself (matching laya-mlx's
        ``agent.predict = agent.system_one`` pattern).

        Args:
            state: Input state payload.
            questions: Dict of question definitions (laya-mlx schema).
            model: Adapter name to use (default: ``self.default_model``).
            **kwargs: Forwarded to the adapter's system_one() method.

        Returns:
            dict: Result dict from the selected adapter.
        """
        adapter = self._get_adapter(model or self.default_model)
        fn = getattr(adapter, "system_one", None) or getattr(adapter, "predict", None)
        if fn is None:
            raise TypeError(
                f"Adapter {type(adapter).__name__!r} must provide system_one or predict."
            )
        return fn(state, questions, **kwargs)

    # laya-mlx API alias: router.system_one(state, questions)
    system_one = predict

    # -----------------------------------------------------------------------
    # Registry management
    # -----------------------------------------------------------------------

    def register(self, name: str, adapter: Any) -> None:
        """Register a named adapter instance.

        Args:
            name: Logical model name (e.g. "onnx", "ollama", "custom").
            adapter: Adapter instance with a system_one or predict method.
        """
        self._register(name, adapter)

    def _register(self, name: str, adapter: Any) -> None:
        """Internal: add/update an adapter in the LRU registry."""
        if name in self._registry:
            self._registry.move_to_end(name)
            self._registry[name] = adapter
        else:
            self._registry[name] = adapter
            if len(self._registry) > self.max_loaded:
                evicted_name, _ = self._registry.popitem(last=False)
                logger.info("Router evicted adapter %r from registry.", evicted_name)

    def _get_adapter(self, name: str) -> Any:
        """Retrieve (and promote) a named adapter, creating it lazily if absent.

        Lazy creation uses the _build_adapter() factory. If adapter cannot
        be built (e.g. missing ONNX model files), falls back to the Ollama
        mock adapter automatically.

        Args:
            name: Adapter name string.

        Returns:
            Any: Loaded adapter instance.
        """
        if name in self._registry:
            self._registry.move_to_end(name)
            return self._registry[name]

        # Lazy build
        adapter = self._build_adapter(name)
        self._register(name, adapter)
        return adapter

    def _build_adapter(self, name: str) -> Any:
        """Factory method: construct an adapter by name.

        Tries to build the requested adapter. If construction fails (e.g. ONNX
        model files not present), falls back to OllamaSystemOneAdapter in mock
        mode and logs a warning.

        Args:
            name: Adapter name.

        Returns:
            Any: Constructed adapter instance.
        """
        from pathlib import Path

        if name == MODEL_ONNX:
            onnx_dir = Path("models/onnx")
            enc_path = onnx_dir / "encoder_int8.onnx"
            heads_path = onnx_dir / "heads_int8.onnx"
            ch_path = onnx_dir / "choice_head_int8.onnx"

            if enc_path.exists() and heads_path.exists() and ch_path.exists():
                # Deferred import to keep router importable without onnxruntime
                from system_one_engine.adapters.onnx_runtime_adapter import ONNXRuntimeAdapter
                from system_one_engine.training.dataset import SimpleVocabTokenizer

                tokenizer = SimpleVocabTokenizer(vocab_size=500, max_len=16)
                logger.info("Router: building ONNX adapter from %s", onnx_dir)
                return ONNXRuntimeAdapter(
                    encoder_path=str(enc_path),
                    heads_path=str(heads_path),
                    choice_head_path=str(ch_path),
                    tokenizer=tokenizer,
                    num_threads=4,
                )

            logger.warning(
                "Router: ONNX model files not found at %s; falling back to Ollama mock adapter.",
                onnx_dir,
            )

        # Ollama adapter (or fallback)
        from system_one_engine.adapters.ollama_adapter import OllamaSystemOneAdapter

        if name == MODEL_OLLAMA:
            logger.info("Router: building Ollama adapter.")
            return OllamaSystemOneAdapter()

        # Unknown model name — fall back to mock
        logger.warning(
            "Router: unknown model name %r; falling back to Ollama mock adapter.", name
        )
        return OllamaSystemOneAdapter(mock_mode=True)

    def loaded_models(self) -> list[str]:
        """Return the names of currently loaded adapter instances (most-recent-first).

        Returns:
            list[str]: Names of loaded adapters, newest first.
        """
        return list(reversed(list(self._registry.keys())))
