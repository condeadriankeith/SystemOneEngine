"""High-Performance Stdio MCP Server for SystemOneEngine.

Exposes System One non-autoregressive decision primitives (Choice, Score, Boolean,
Decide, Shortlist, and Safety Guardrail) over the Model Context Protocol (MCP) using
standard JSON-RPC 2.0 via sys.stdin / sys.stdout.
"""

import json
import logging
import sys
import time
from typing import Any, Optional

from system_one_engine.agent.contracts import OSAction, OSActionType, SafetyMode
from system_one_engine.agent.guardrails import SafetyGuardrail
from system_one_engine.client import SystemOneClient
from system_one_engine.core.shortlist import shortlist_choice

# Configure logging to stderr so stdout remains purely JSON-RPC messages
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("system-one-mcp")

# In-process SystemOneClient singleton
_CLIENT: Optional[SystemOneClient] = None
_GUARDRAIL = SafetyGuardrail()


def get_client() -> SystemOneClient:
    """Lazy initialization of SystemOneClient with in-process adapter."""
    global _CLIENT
    if _CLIENT is None:
        try:
            from system_one_engine import load, MODEL_ONNX
            adapter = load(MODEL_ONNX)
            _CLIENT = SystemOneClient(adapter=adapter)
            logger.info("SystemOneClient initialized with in-process ONNX adapter.")
        except Exception as exc:
            logger.warning("Falling back to in-process mock Ollama adapter: %s", exc)
            from system_one_engine.adapters.ollama_adapter import OllamaSystemOneAdapter
            adapter = OllamaSystemOneAdapter(mock_mode=True)
            _CLIENT = SystemOneClient(adapter=adapter)
    return _CLIENT


# Tool Definitions for tools/list
TOOLS = [
    {
        "name": "system_one_choice",
        "description": "Sub-3ms categorical choice evaluation from 2 to 64 candidate options with calibrated confidence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {"type": "string", "description": "Context or situation description"},
                "instruction": {"type": "string", "description": "Decision question or instruction"},
                "criteria": {
                    "type": "object",
                    "description": "Map of candidate keys to their description/rationale",
                    "additionalProperties": {"type": "string"},
                },
                "temperature": {"type": "number", "description": "Softmax temperature (default: 0.8)", "default": 0.8},
            },
            "required": ["context", "instruction", "criteria"],
        },
    },
    {
        "name": "system_one_score",
        "description": "Continuous score expectation E[S] over an ordinal scale (2 to 10 levels) with calibrated confidence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {"type": "string", "description": "Context or item description to evaluate"},
                "instruction": {"type": "string", "description": "Scoring rubric or grading criteria"},
                "levels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered scale level labels (e.g. ['low', 'medium', 'high', 'critical'])",
                },
                "temperature": {"type": "number", "description": "Softmax temperature (default: 0.8)", "default": 0.8},
            },
            "required": ["context", "instruction", "levels"],
        },
    },
    {
        "name": "system_one_boolean",
        "description": "Sub-millisecond calibrated binary assertion under uncertainty (True/False with confidence).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {"type": "string", "description": "Context or premise"},
                "statement": {"type": "string", "description": "Statement or assertion to evaluate"},
                "temperature": {"type": "number", "description": "Softmax temperature (default: 0.8)", "default": 0.8},
            },
            "required": ["context", "statement"],
        },
    },
    {
        "name": "system_one_safety_check",
        "description": "Sub-5ms pre-execution safety audit for shell commands, intercepting destructive actions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to audit"},
                "safety_mode": {
                    "type": "string",
                    "enum": ["STRICT", "BALANCED", "AUTONOMOUS"],
                    "description": "Safety policy mode (default: BALANCED)",
                    "default": "BALANCED",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "system_one_shortlist",
        "description": "Top-K semantic candidate shortlisting before LLM context filling, saving token costs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query or target intent"},
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of candidate strings to filter/rank",
                },
                "top_k": {"type": "integer", "description": "Number of top candidates to return (default: 5)", "default": 5},
            },
            "required": ["query", "candidates"],
        },
    },
]


def handle_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Route tool call to corresponding System One primitive."""
    client = get_client()

    if name == "system_one_choice":
        t0 = time.perf_counter()
        resp = client.choice(
            context=arguments["context"],
            instruction=arguments["instruction"],
            criteria=arguments["criteria"],
            temperature=float(arguments.get("temperature", 0.8)),
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "choice": resp.choice,
            "confidence": resp.confidence,
            "probabilities": resp.probabilities,
            "latency_ms": round(dt_ms, 2),
        }

    if name == "system_one_score":
        t0 = time.perf_counter()
        resp = client.score(
            context=arguments["context"],
            instruction=arguments["instruction"],
            criteria=arguments["levels"],
            temperature=float(arguments.get("temperature", 0.8)),
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "expected_score": resp.score,
            "confidence": resp.confidence,
            "distribution": resp.level_distribution,
            "latency_ms": round(dt_ms, 2),
        }

    if name == "system_one_boolean":
        t0 = time.perf_counter()
        resp = client.boolean(
            context=arguments["context"],
            instruction=arguments["statement"],
            temperature=float(arguments.get("temperature", 0.8)),
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "confirmed": resp.confirmed,
            "confidence": resp.confidence,
            "probability": resp.probability,
            "latency_ms": round(dt_ms, 2),
        }

    if name == "system_one_safety_check":
        t0 = time.perf_counter()
        cmd = arguments["command"]
        mode_str = arguments.get("safety_mode", "BALANCED").upper()
        mode = getattr(SafetyMode, mode_str, SafetyMode.BALANCED)
        action = OSAction(
            action_type=OSActionType.SHELL,
            target=cmd,
            params={"command": cmd},
            rationale="MCP safety check",
        )
        result = _GUARDRAIL.check(action, mode)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "is_safe": result.is_safe,
            "confidence": result.confidence,
            "reason": result.reason,
            "requires_confirmation": result.requires_confirmation,
            "latency_ms": round(dt_ms, 2),
        }

    if name == "system_one_shortlist":
        t0 = time.perf_counter()
        query = arguments["query"]
        candidates = arguments["candidates"]
        top_k = int(arguments.get("top_k", 5))

        def _bag_embed(texts):
            import hashlib
            import numpy as np

            dim = 32
            out = np.zeros((len(texts), dim), dtype=np.float64)
            for idx, text in enumerate(texts):
                for word in str(text).lower().split():
                    val = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dim
                    out[idx, val] += 1.0
                norm = np.linalg.norm(out[idx])
                if norm > 0:
                    out[idx] /= norm
            return out

        top_candidates = shortlist_choice(query, candidates, embed_fn=_bag_embed, k=top_k)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "selected_candidates": top_candidates,
            "latency_ms": round(dt_ms, 2),
        }

    raise ValueError(f"Unknown tool: {name}")


def process_message(message: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Process a single JSON-RPC 2.0 message."""
    msg_id = message.get("id")
    method = message.get("method")

    # Notifications without id
    if msg_id is None:
        if method == "notifications/initialized":
            logger.info("Client completed initialization handshake.")
        return None

    # Request handlers
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False},
                },
                "serverInfo": {
                    "name": "system-one-engine",
                    "version": "0.3.0",
                },
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        params = message.get("params", {})
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        try:
            result = handle_tool_call(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2),
                        }
                    ],
                    "isError": False,
                },
            }
        except Exception as exc:
            logger.error("Error executing tool '%s': %s", tool_name, exc)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error executing tool '{tool_name}': {exc}",
                        }
                    ],
                    "isError": True,
                },
            }

    # Unknown method error
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {
            "code": -32601,
            "message": f"Method '{method}' not found",
        },
    }


def run_stdio_server() -> None:
    """Main JSON-RPC stdio event loop."""
    logger.info("System One Stdio MCP Server started. Listening on sys.stdin...")

    for line in sys.stdin:
        line_clean = line.strip()
        if not line_clean:
            continue

        try:
            message = json.loads(line_clean)
            response = process_message(message)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except json.JSONDecodeError as err:
            logger.error("Invalid JSON received: %s", err)
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {err}"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            logger.error("Unhandled error in MCP server: %s", exc)


if __name__ == "__main__":
    run_stdio_server()
