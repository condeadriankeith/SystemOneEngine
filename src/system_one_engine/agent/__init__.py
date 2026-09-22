"""System One Computer-Use & OS Task Automation Agent module.

Provides:
- SystemOneOSAgent: Autonomous perception-decision-safety-action agent loop.
- AgentTask: Task definition schema.
- OSObservation: Desktop & terminal environmental observation.
- OSAction: Executable OS action primitive.
- OSActionType: Action primitive enumeration.
- OSActionResult: Execution outcome payload.
- SafetyMode: Safety policy profile (STRICT, BALANCED, AUTONOMOUS).
- SafetyGuardrail: Real-time pre-execution safety interceptor.
- BaseOSDriver, SimulatedOSDriver, LocalOSDriver: Driver backends.
"""

from system_one_engine.agent.agent import SystemOneOSAgent
from system_one_engine.agent.contracts import (
    ActionStatus,
    AgentTask,
    OSAction,
    OSActionResult,
    OSActionType,
    OSObservation,
    SafetyMode,
)
from system_one_engine.agent.driver import (
    BaseOSDriver,
    LocalOSDriver,
    SimulatedOSDriver,
)
from system_one_engine.agent.guardrails import (
    SafetyCheckResult,
    SafetyGuardrail,
)

__all__ = [
    "SystemOneOSAgent",
    "AgentTask",
    "OSObservation",
    "OSAction",
    "OSActionType",
    "OSActionResult",
    "ActionStatus",
    "SafetyMode",
    "SafetyGuardrail",
    "SafetyCheckResult",
    "BaseOSDriver",
    "SimulatedOSDriver",
    "LocalOSDriver",
]
