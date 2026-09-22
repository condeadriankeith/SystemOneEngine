"""OS Drivers for System One Computer-Use and Automation Agent.

Provides:
- BaseOSDriver: Abstract interface defining perception and execution protocols.
- SimulatedOSDriver: In-memory mock desktop environment for fast, sandboxed unit testing.
- LocalOSDriver: Real-world OS execution backend supporting Windows, macOS, and Linux.
"""

from abc import ABC, abstractmethod
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Optional

from system_one_engine.agent.contracts import (
    ActionStatus,
    OSAction,
    OSActionResult,
    OSActionType,
    OSObservation,
)


class BaseOSDriver(ABC):
    """Abstract driver interface defining perception and execution protocols."""

    @abstractmethod
    def observe(self) -> OSObservation:
        """Capture the current state of the desktop and terminal environment."""
        ...

    @abstractmethod
    def execute(self, action: OSAction) -> OSActionResult:
        """Execute a discrete OSAction and return an OSActionResult."""
        ...


class SimulatedOSDriver(BaseOSDriver):
    """Deterministic in-memory mock OS environment for sandboxed unit testing.

    Simulates:
    - In-memory virtual filesystem.
    - Simulated shell commands (e.g. git status, ls, python scripts).
    - Simulated active windows and clipboard.
    """

    def __init__(self, initial_cwd: str = "/workspace") -> None:
        self.cwd = initial_cwd
        self.active_window = "Visual Studio Code"
        self.open_windows = ["Visual Studio Code", "Windows Terminal", "Chrome"]
        self.clipboard = "https://github.com/condeadriankeith/SystemOneEngine"
        self.virtual_fs: dict[str, str] = {
            "/workspace/README.md": "# SystemOneEngine\nAutonomous Decision Engine",
            "/workspace/pyproject.toml": '[project]\nname = "system-one-engine"',
        }
        self.last_stdout = ""
        self.last_stderr = ""
        self.action_history: list[OSAction] = []

    def observe(self) -> OSObservation:
        """Return the current simulated environment state."""
        return OSObservation(
            active_window=self.active_window,
            cwd=self.cwd,
            stdout_tail=self.last_stdout,
            stderr_tail=self.last_stderr,
            screen_width=1920,
            screen_height=1080,
            cursor_x=450,
            cursor_y=300,
            clipboard=self.clipboard,
            open_windows=list(self.open_windows),
        )

    def execute(self, action: OSAction) -> OSActionResult:
        """Execute action within the virtual simulation."""
        t0 = time.perf_counter()
        self.action_history.append(action)

        if action.action_type == OSActionType.SHELL:
            cmd = action.command.strip()
            if cmd.startswith("git status"):
                out = "On branch main\nYour branch is up to date with 'origin/main'.\nnothing to commit, working tree clean"
                status = ActionStatus.SUCCESS
            elif cmd.startswith("ls") or cmd.startswith("dir"):
                out = "\n".join(Path(p).name for p in self.virtual_fs if p.startswith(self.cwd))
                status = ActionStatus.SUCCESS
            elif cmd.startswith("echo"):
                out = cmd[5:].strip().strip('"').strip("'")
                status = ActionStatus.SUCCESS
            else:
                out = f"[simulated stdout] Executed command: '{cmd}'"
                status = ActionStatus.SUCCESS
            self.last_stdout = out
            self.last_stderr = ""

        elif action.action_type == OSActionType.FILE_READ:
            target_path = action.target
            if target_path in self.virtual_fs:
                out = self.virtual_fs[target_path]
                status = ActionStatus.SUCCESS
            else:
                out = f"FileNotFoundError: '{target_path}'"
                status = ActionStatus.FAILED

        elif action.action_type == OSActionType.FILE_WRITE:
            target_path = action.target
            content = action.params.get("content", "")
            self.virtual_fs[target_path] = content
            out = f"Wrote {len(content)} characters to '{target_path}'"
            status = ActionStatus.SUCCESS

        elif action.action_type == OSActionType.LAUNCH_APP:
            app_name = action.target
            self.active_window = app_name
            if app_name not in self.open_windows:
                self.open_windows.append(app_name)
            out = f"Focused application '{app_name}'"
            status = ActionStatus.SUCCESS

        elif action.action_type == OSActionType.COMPLETE:
            out = f"Task completed: {action.rationale or 'Done.'}"
            status = ActionStatus.SUCCESS

        else:
            out = f"Simulated execution for action {action.action_type.value}"
            status = ActionStatus.SUCCESS

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return OSActionResult(
            action=action,
            status=status,
            output=out,
            latency_ms=latency_ms,
            observation_after=self.observe(),
        )


class LocalOSDriver(BaseOSDriver):
    """Live OS execution driver for Windows, macOS, and Linux.

    Supports:
    - Real subprocess terminal execution with safe timeouts.
    - Real file reading and writing.
    - Application focus/launching.
    - Active window detection via native platform APIs (ctypes user32 on Windows).
    """

    def __init__(self, default_cwd: Optional[str] = None, command_timeout: float = 15.0) -> None:
        self.cwd = default_cwd or os.getcwd()
        self.command_timeout = command_timeout
        self.last_stdout = ""
        self.last_stderr = ""

    def _get_active_window_title(self) -> str:
        """Query currently active foreground window title."""
        if platform.system() == "Windows":
            try:
                import ctypes
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                    return buf.value
            except Exception:
                pass
        return "Terminal / Desktop"

    def observe(self) -> OSObservation:
        """Capture real-world OS state."""
        active_title = self._get_active_window_title()
        return OSObservation(
            active_window=active_title,
            cwd=self.cwd,
            stdout_tail=self.last_stdout,
            stderr_tail=self.last_stderr,
            screen_width=1920,
            screen_height=1080,
            open_windows=[active_title] if active_title else [],
        )

    def execute(self, action: OSAction) -> OSActionResult:
        """Execute action directly on the host operating system."""
        t0 = time.perf_counter()

        if action.action_type == OSActionType.SHELL:
            cmd = action.command.strip()
            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=self.cwd,
                    capture_output=True,
                    text=True,
                    timeout=self.command_timeout,
                    encoding="utf-8",
                    errors="replace",
                )
                out = proc.stdout if proc.returncode == 0 else f"{proc.stdout}\n{proc.stderr}".strip()
                status = ActionStatus.SUCCESS if proc.returncode == 0 else ActionStatus.FAILED
                self.last_stdout = proc.stdout
                self.last_stderr = proc.stderr
            except subprocess.TimeoutExpired:
                out = f"Command timed out after {self.command_timeout} seconds."
                status = ActionStatus.FAILED
            except Exception as exc:
                out = f"Execution error: {exc}"
                status = ActionStatus.FAILED

        elif action.action_type == OSActionType.FILE_READ:
            path = Path(action.target)
            if not path.is_absolute():
                path = Path(self.cwd) / path
            try:
                out = path.read_text(encoding="utf-8", errors="replace")[:4000]
                status = ActionStatus.SUCCESS
            except Exception as exc:
                out = f"Failed to read file '{path}': {exc}"
                status = ActionStatus.FAILED

        elif action.action_type == OSActionType.FILE_WRITE:
            path = Path(action.target)
            if not path.is_absolute():
                path = Path(self.cwd) / path
            try:
                content = action.params.get("content", "")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                out = f"Successfully wrote {len(content)} bytes to '{path}'"
                status = ActionStatus.SUCCESS
            except Exception as exc:
                out = f"Failed to write file '{path}': {exc}"
                status = ActionStatus.FAILED

        elif action.action_type == OSActionType.LAUNCH_APP:
            target = action.target
            try:
                if platform.system() == "Windows":
                    os.startfile(target)
                elif platform.system() == "Darwin":
                    subprocess.Popen(["open", target])
                else:
                    subprocess.Popen(["xdg-open", target])
                out = f"Launched application / target '{target}'"
                status = ActionStatus.SUCCESS
            except Exception as exc:
                out = f"Failed to launch '{target}': {exc}"
                status = ActionStatus.FAILED

        elif action.action_type == OSActionType.COMPLETE:
            out = f"Task completed: {action.rationale or 'Done.'}"
            status = ActionStatus.SUCCESS

        else:
            out = f"Executed {action.action_type.value}: {action.target}"
            status = ActionStatus.SUCCESS

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return OSActionResult(
            action=action,
            status=status,
            output=out,
            latency_ms=latency_ms,
            observation_after=self.observe(),
        )
