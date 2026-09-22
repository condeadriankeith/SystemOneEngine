"""Pydantic v2 data contracts for System One Computer-Use and OS Task Automation Agent.

Defines schemas for:
- OSActionType: Enumeration of primitive OS actions (SHELL, LAUNCH_APP, CLICK, TYPE, etc.)
- OSObservation: Structured environmental state of the user's desktop/terminal
- OSAction: Executable action payload with parameters and estimated safety level
- OSActionResult: Execution outcome with status, output, and latency metrics
- AgentTask: User task specification with goal, constraints, and safety mode
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class OSActionType(str, Enum):
    """Supported low-level OS and UI automation primitives."""

    SHELL = "shell"              # Execute terminal command
    LAUNCH_APP = "launch_app"    # Launch or bring application to focus
    FILE_READ = "file_read"      # Read file content safely
    FILE_WRITE = "file_write"    # Write/edit file content
    CLICK = "click"              # Mouse click at coordinates or UI target
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"                # Keystroke input
    HOTKEY = "hotkey"            # Keyboard shortcuts (e.g. ['ctrl', 'c'])
    WAIT = "wait"                # Sleep / pause for UI update
    COMPLETE = "complete"        # Signal task completion with final summary
    ASK_USER = "ask_user"        # Prompt user for clarification or confirmation


class SafetyMode(str, Enum):
    """Execution safety profile for the agent."""

    STRICT = "strict"          # Confirm all state-changing or shell actions
    BALANCED = "balanced"      # Confirm destructive/risky actions; auto-execute safe queries
    AUTONOMOUS = "autonomous"  # Auto-execute all actions unless blocked by critical guardrail


class OSObservation(BaseModel):
    """Observed desktop and terminal state at a given decision step.

    Attributes:
        active_window: Title of the currently focused desktop application window.
        cwd: Current working directory for shell interactions.
        stdout_tail: Recent stdout output lines from the terminal.
        stderr_tail: Recent stderr output lines from the terminal.
        screen_width: Display horizontal resolution in pixels.
        screen_height: Display vertical resolution in pixels.
        cursor_x: Current mouse cursor X coordinate.
        cursor_y: Current mouse cursor Y coordinate.
        clipboard: Current text clipboard content, if available.
        open_windows: List of visible application window titles.
        extra: Additional environmental metadata.
    """

    active_window: str = Field(default="", description="Active window title.")
    cwd: str = Field(default=".", description="Current working directory.")
    stdout_tail: str = Field(default="", description="Recent stdout lines.")
    stderr_tail: str = Field(default="", description="Recent stderr lines.")
    screen_width: int = Field(default=1920, ge=1, description="Screen width in pixels.")
    screen_height: int = Field(default=1080, ge=1, description="Screen height in pixels.")
    cursor_x: int = Field(default=0, ge=0, description="Mouse X position.")
    cursor_y: int = Field(default=0, ge=0, description="Mouse Y position.")
    clipboard: Optional[str] = Field(default=None, description="Current clipboard text.")
    open_windows: list[str] = Field(default_factory=list, description="List of open window titles.")
    extra: dict[str, Any] = Field(default_factory=dict, description="Custom metadata.")

    def summary(self) -> str:
        """Render a concise single-paragraph state summary for System One context."""
        parts = [
            f"ActiveWindow: '{self.active_window or 'None'}'",
            f"CWD: '{self.cwd}'",
        ]
        if self.open_windows:
            win_str = ", ".join(self.open_windows[:5])
            parts.append(f"OpenWindows: [{win_str}]")
        if self.stdout_tail.strip():
            tail_snippet = self.stdout_tail.strip().splitlines()[-1][:120]
            parts.append(f"LastStdout: '{tail_snippet}'")
        if self.stderr_tail.strip():
            err_snippet = self.stderr_tail.strip().splitlines()[-1][:120]
            parts.append(f"LastStderr: '{err_snippet}'")
        return " | ".join(parts)


class OSAction(BaseModel):
    """Discrete action selected by System One or System Two for execution.

    Attributes:
        action_type: Primitive operation type.
        target: Target identifier, UI element, application name, or path.
        params: Key-value parameters (e.g. command string, text to type, coordinates).
        rationale: Brief explanation of why this action was selected.
        confidence: System One calibrated decision certainty in [0.0, 1.0].
    """

    action_type: OSActionType = Field(..., description="Action primitive.")
    target: str = Field(default="", description="Target application, file, or coordinate string.")
    params: dict[str, Any] = Field(default_factory=dict, description="Action-specific parameters.")
    rationale: str = Field(default="", description="Decision justification.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score [0.0, 1.0].")

    @property
    def command(self) -> str:
        """Convenience property for shell command string."""
        return str(self.params.get("command", self.target))

    @property
    def text(self) -> str:
        """Convenience property for text payload."""
        return str(self.params.get("text", self.target))


class ActionStatus(str, Enum):
    """Outcome status of an executed OS action."""

    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED_BY_GUARDRAIL = "blocked_by_guardrail"
    ESCALATED = "escalated"
    PAUSED_FOR_USER = "paused_for_user"


class OSActionResult(BaseModel):
    """Execution feedback returned after an OSAction is applied.

    Attributes:
        action: The original action that was executed.
        status: Execution status enum.
        output: Text output, exit code string, or error traceback.
        latency_ms: Execution duration in milliseconds.
        observation_after: Updated environmental state after action completion.
    """

    action: OSAction = Field(..., description="Executed action specification.")
    status: ActionStatus = Field(..., description="Execution outcome.")
    output: str = Field(default="", description="Command stdout, UI feedback, or error.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Action execution latency in ms.")
    observation_after: Optional[OSObservation] = Field(default=None, description="Resulting OS state.")


class AgentTask(BaseModel):
    """High-level user task specification for the autonomous agent.

    Attributes:
        goal: Clear natural language description of what task to accomplish.
        max_steps: Maximum allowable execution steps before terminating.
        safety_mode: Policy profile governing confirmation requirements.
        dry_run: If True, simulate actions without modifying live OS state.
    """

    goal: str = Field(..., min_length=1, description="Task goal prompt.")
    max_steps: int = Field(default=20, ge=1, le=200, description="Max execution steps.")
    safety_mode: SafetyMode = Field(default=SafetyMode.BALANCED, description="Safety mode.")
    dry_run: bool = Field(default=False, description="If True, log without mutating OS.")
