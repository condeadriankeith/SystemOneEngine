"""System One Autonomous Computer-Use and OS Task Automation Agent.

Implements the dual-process execution loop:
1. Perception: Captures current desktop / terminal state (OSObservation).
2. Action Selection (System 1): Evaluates candidate actions in <15ms with calibrated confidence.
3. Dual-Process Escalation: If confidence < threshold (0.70), flags for System 2 deliberation.
4. Pre-Execution Guardrails: Verifies action safety in <5ms using System One Boolean primitives.
5. Execution: Dispatches action to driver (SimulatedOSDriver or LocalOSDriver).
"""

import logging
import os
import time
from typing import Optional

from system_one_engine.agent.contracts import (
    ActionStatus,
    AgentTask,
    OSAction,
    OSActionResult,
    OSActionType,
    OSObservation,
    SafetyMode,
)
from system_one_engine.agent.driver import BaseOSDriver, LocalOSDriver, SimulatedOSDriver
from system_one_engine.agent.guardrails import SafetyGuardrail
from system_one_engine.client import SystemOneClient

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.70


class SystemOneOSAgent:
    """Autonomous desktop and terminal task agent powered by System One Engine."""

    def __init__(
        self,
        client: Optional[SystemOneClient] = None,
        driver: Optional[BaseOSDriver] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """Initialize System One OS Agent.

        Args:
            client: SystemOneClient instance for sub-20ms decisions & safety guardrails.
            driver: OS driver backend (SimulatedOSDriver or LocalOSDriver).
            confidence_threshold: Minimum confidence [0.0, 1.0] before escalating to System 2.
        """
        self.client = client
        self.driver = driver or LocalOSDriver()
        self.guardrail = SafetyGuardrail(client=client)
        self.confidence_threshold = confidence_threshold
        self.history: list[OSActionResult] = []

    def _generate_candidate_actions(self, task: AgentTask, obs: OSObservation, step: int) -> dict[str, OSAction]:
        """Formulate plausible next actions tailored to the task and current environment."""
        goal_lower = task.goal.lower()
        candidates: dict[str, OSAction] = {}

        # 1. Inspect repository or filesystem
        if "git" in goal_lower or "repo" in goal_lower or "status" in goal_lower:
            candidates["git_status"] = OSAction(
                action_type=OSActionType.SHELL,
                target="git status",
                params={"command": "git status"},
                rationale="Query git repository state and modified files.",
            )
            candidates["git_log"] = OSAction(
                action_type=OSActionType.SHELL,
                target="git log -n 3",
                params={"command": "git log -n 3 --oneline"},
                rationale="Review recent commit history.",
            )

        # 2. Run automated tests / verification
        if "test" in goal_lower or "pytest" in goal_lower or "verify" in goal_lower:
            candidates["run_tests"] = OSAction(
                action_type=OSActionType.SHELL,
                target="uv run pytest",
                params={"command": "uv run pytest"},
                rationale="Execute project unit tests.",
            )

        # 3. Directory listing and exploration
        candidates["list_dir"] = OSAction(
            action_type=OSActionType.SHELL,
            target="dir" if os.name == "nt" else "ls -la",
            params={"command": "dir" if os.name == "nt" else "ls -la"},
            rationale="Inspect current directory contents.",
        )

        # 4. System diagnostics / environment
        if "cpu" in goal_lower or "memory" in goal_lower or "system" in goal_lower:
            cmd = "wmic cpu get loadpercentage" if os.name == "nt" else "top -b -n 1 | head -n 10"
            candidates["system_info"] = OSAction(
                action_type=OSActionType.SHELL,
                target=cmd,
                params={"command": cmd},
                rationale="Check system CPU and performance metrics.",
            )

        # 5. File read if target file mentioned
        for word in task.goal.split():
            if "." in word and len(word) > 3 and not word.startswith("http"):
                clean_path = word.strip("'\":,")
                candidates[f"read_{clean_path}"] = OSAction(
                    action_type=OSActionType.FILE_READ,
                    target=clean_path,
                    params={},
                    rationale=f"Inspect target file '{clean_path}'.",
                )

        # Filter out already executed shell actions so the agent progresses
        executed_commands = {
            res.action.command for res in self.history if res.action.action_type == OSActionType.SHELL
        }
        filtered_candidates = {
            k: act for k, act in candidates.items()
            if act.action_type != OSActionType.SHELL or act.command not in executed_commands
        }

        # 6. Task completion option
        if step > 0 or not filtered_candidates or "done" in goal_lower:
            filtered_candidates["complete_task"] = OSAction(
                action_type=OSActionType.COMPLETE,
                target="",
                params={},
                rationale="Goal requirements have been satisfied.",
            )

        # Ensure at least 2 candidates for categorical Choice evaluation
        if len(filtered_candidates) < 2:
            filtered_candidates["wait"] = OSAction(
                action_type=OSActionType.WAIT,
                target="1",
                params={"seconds": 1},
                rationale="Pause to await background process completion.",
            )
            filtered_candidates["complete_task"] = OSAction(
                action_type=OSActionType.COMPLETE,
                target="",
                params={},
                rationale="Finalize and complete task.",
            )
        return filtered_candidates

    def decide_action(
        self, task: AgentTask, obs: OSObservation, step: int
    ) -> tuple[OSAction, bool]:
        """Evaluate current observation and select next OSAction using System One.

        Returns:
            tuple[OSAction, bool]: (selected_action, is_escalated_flag)
        """
        candidates = self._generate_candidate_actions(task, obs, step)

        # If System One Client is available, run categorical Choice primitive
        if self.client is not None:
            criteria = {k: act.rationale for k, act in candidates.items()}
            context = f"Goal: {task.goal} | {obs.summary()} | Step: {step}/{task.max_steps}"
            try:
                t0 = time.perf_counter()
                choice_resp = self.client.choice(
                    context=context,
                    instruction="Select the most effective next action to achieve the goal.",
                    criteria=criteria,
                    temperature=0.8,
                )
                dt_ms = (time.perf_counter() - t0) * 1000.0

                winning_key = choice_resp.choice
                selected_action = candidates[winning_key]
                selected_action.confidence = choice_resp.confidence

                # Check if decision confidence satisfies threshold
                is_escalated = choice_resp.confidence < self.confidence_threshold
                logger.debug(
                    "System 1 decision: %s (confidence: %.1f%%, latency: %.2fms, escalated: %s)",
                    winning_key,
                    choice_resp.confidence * 100,
                    dt_ms,
                    is_escalated,
                )
                return selected_action, is_escalated
            except Exception as exc:
                logger.warning("System 1 inference fallback due to: %s", exc)

        # Heuristic fallback if client not configured
        first_key = next(iter(candidates))
        action = candidates[first_key]
        action.confidence = 0.95
        return action, False

    def step(self, task: AgentTask, step: int) -> OSActionResult:
        """Execute a single perception-decision-safety-action cycle."""
        t0 = time.perf_counter()

        # 1. Perception
        obs = self.driver.observe()

        # 2. System 1 Action Selection
        action, escalated = self.decide_action(task, obs, step)

        # 3. Real-Time Pre-Execution Safety Guardrail (<5ms)
        safety = self.guardrail.check(action, task.safety_mode)
        if not safety.is_safe or safety.requires_confirmation:
            if task.safety_mode == SafetyMode.STRICT or not safety.is_safe:
                return OSActionResult(
                    action=action,
                    status=ActionStatus.BLOCKED_BY_GUARDRAIL,
                    output=f"Safety Guardrail Intercept: {safety.reason}",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    observation_after=obs,
                )

        # 4. Dry-run intercept
        if task.dry_run:
            return OSActionResult(
                action=action,
                status=ActionStatus.SUCCESS,
                output=f"[DRY-RUN] Simulated execution of {action.action_type.value}: '{action.target}'",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                observation_after=obs,
            )

        # 5. Live / Driver Execution
        result = self.driver.execute(action)
        self.history.append(result)
        return result

    def run(self, task: AgentTask) -> list[OSActionResult]:
        """Run full autonomous agent task loop until completion or max steps."""
        results: list[OSActionResult] = []

        for step_idx in range(task.max_steps):
            result = self.step(task, step_idx)
            results.append(result)

            # Stop on task completion, unrecoverable failure, or guardrail block
            if result.action.action_type == OSActionType.COMPLETE:
                break
            if result.status == ActionStatus.BLOCKED_BY_GUARDRAIL:
                break

        return results
