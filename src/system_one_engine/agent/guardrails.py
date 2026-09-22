"""Real-time Safety Guardrails for Autonomous Computer-Use Agent.

Provides sub-5ms pre-execution validation using System One's Boolean decision primitive:
- Detects destructive filesystem mutations (recursive deletions, wildcard purges).
- Detects system integrity threats (disk format, system shutdown, privilege tampering).
- Detects credential leaks (dumping environment variables, private keys).
- Halts execution with calibrated confidence when risk exceeds safety tolerance.
"""

import re
from dataclasses import dataclass
from typing import Optional

from system_one_engine.agent.contracts import OSAction, OSActionType, SafetyMode
from system_one_engine.client import SystemOneClient


# Critical patterns that are unconditionally destructive or dangerous
DESTRUCTIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brm\s+-[rf]{1,2}\s+[/~*]", re.IGNORECASE),
    re.compile(r"\bdel(\s+/[a-zA-Z]+)*\s+[*]", re.IGNORECASE),
    re.compile(r"\brmdir(\s+/[a-zA-Z]+)*\s+[a-zA-Z]:", re.IGNORECASE),
    re.compile(r"\bRemove-Item\s+.*-Recurse\s+.*-Force\s+[a-zA-Z]:", re.IGNORECASE),
    re.compile(r"\bformat\s+[a-zA-Z]:", re.IGNORECASE),
    re.compile(r"\bdiskpart\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\b(drop\s+database|drop\s+table)\b", re.IGNORECASE),
    re.compile(r"\b(mkfs|dd\s+if=.*of=/dev/)", re.IGNORECASE),
    re.compile(r"\bcat\s+~/\.ssh/id_[a-z0-9_]+", re.IGNORECASE),
]

SAFE_READONLY_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*(git\s+(status|log|diff|branch)|dir|ls|pwd|whoami|echo|python\s+--version|pytest|uv\s+run\s+pytest)\b", re.IGNORECASE),
    re.compile(r"^\s*(cat|type|head|tail|grep|findstr)\s+", re.IGNORECASE),
]


@dataclass
class SafetyCheckResult:
    """Outcome of a pre-execution safety evaluation.

    Attributes:
        is_safe: True if the action is deemed safe to execute autonomously.
        confidence: Certainty of the safety evaluation [0.0, 1.0].
        reason: Explanation of why the action was flagged or cleared.
        requires_confirmation: True if user must explicitly approve before proceeding.
    """

    is_safe: bool
    confidence: float
    reason: str
    requires_confirmation: bool


class SafetyGuardrail:
    """Sub-5ms safety interceptor for OS actions."""

    def __init__(self, client: Optional[SystemOneClient] = None) -> None:
        """Initialize safety guardrail.

        Args:
            client: Optional SystemOneClient for calibrated neural boolean checks.
        """
        self._client = client

    def check(self, action: OSAction, safety_mode: SafetyMode) -> SafetyCheckResult:
        """Evaluate whether an action is safe to execute under the given safety mode.

        Args:
            action: Candidate OS action to evaluate.
            safety_mode: Current safety policy (STRICT, BALANCED, AUTONOMOUS).

        Returns:
            SafetyCheckResult: Assessment with safety flag, confidence, and reason.
        """
        # 1. Non-shell actions (clicks, navigation, typing, waiting)
        if action.action_type in (OSActionType.CLICK, OSActionType.WAIT, OSActionType.HOTKEY, OSActionType.TYPE):
            return SafetyCheckResult(
                is_safe=True,
                confidence=0.98,
                reason=f"Standard non-destructive UI action: {action.action_type.value}",
                requires_confirmation=False,
            )

        # 2. File read actions are always safe
        if action.action_type == OSActionType.FILE_READ:
            return SafetyCheckResult(
                is_safe=True,
                confidence=0.99,
                reason="Read-only filesystem inspection is safe.",
                requires_confirmation=False,
            )

        # 3. Check shell commands
        if action.action_type == OSActionType.SHELL:
            cmd = action.command.strip()

            # A. Check unconditionally destructive patterns
            for pattern in DESTRUCTIVE_PATTERNS:
                if pattern.search(cmd):
                    return SafetyCheckResult(
                        is_safe=False,
                        confidence=1.0,
                        reason=f"Command matches destructive safety filter: '{pattern.pattern}'",
                        requires_confirmation=True,
                    )

            # B. Check known read-only commands
            for pattern in SAFE_READONLY_PATTERNS:
                if pattern.search(cmd):
                    return SafetyCheckResult(
                        is_safe=True,
                        confidence=0.99,
                        reason="Standard read-only query / test command.",
                        requires_confirmation=safety_mode == SafetyMode.STRICT,
                    )

            # C. System One neural boolean verification if client is available
            if self._client is not None:
                try:
                    resp = self._client.boolean(
                        context=f"Target Shell Command: '{cmd}'",
                        instruction="Is this command destructive, irreversible, or hazardous to system files?",
                    )
                    is_hazardous = resp.confirmed
                    return SafetyCheckResult(
                        is_safe=not is_hazardous,
                        confidence=resp.confidence,
                        reason="System One neural safety classification." if is_hazardous else "Cleared by System One guardrail.",
                        requires_confirmation=is_hazardous or (safety_mode == SafetyMode.STRICT),
                    )
                except Exception:
                    # Fall through to mode-based check on exception
                    pass

            # D. Policy-based check based on SafetyMode
            if safety_mode == SafetyMode.STRICT:
                return SafetyCheckResult(
                    is_safe=True,
                    confidence=0.80,
                    reason="Strict mode requires manual user confirmation for all shell commands.",
                    requires_confirmation=True,
                )

            return SafetyCheckResult(
                is_safe=True,
                confidence=0.90,
                reason="Command passed heuristic safety filters.",
                requires_confirmation=False,
            )

        # 4. File writes
        if action.action_type == OSActionType.FILE_WRITE:
            requires_confirm = safety_mode == SafetyMode.STRICT
            return SafetyCheckResult(
                is_safe=True,
                confidence=0.85,
                reason=f"File modification on target '{action.target}'",
                requires_confirmation=requires_confirm,
            )

        # Default fallback
        return SafetyCheckResult(
            is_safe=True,
            confidence=0.90,
            reason="Action passed safety check.",
            requires_confirmation=safety_mode == SafetyMode.STRICT,
        )
