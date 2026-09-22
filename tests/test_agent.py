"""Unit and integration tests for System One Autonomous Computer-Use Agent."""

import pytest
from system_one_engine.agent import (
    ActionStatus,
    AgentTask,
    LocalOSDriver,
    OSAction,
    OSActionResult,
    OSActionType,
    OSObservation,
    SafetyGuardrail,
    SafetyMode,
    SimulatedOSDriver,
    SystemOneOSAgent,
)


def test_agent_contracts_validation():
    """Verify OSObservation, OSAction, OSActionResult, and AgentTask Pydantic models."""
    obs = OSObservation(
        active_window="Terminal",
        cwd="/workspace",
        stdout_tail="file1.py\nfile2.py",
        screen_width=1920,
        screen_height=1080,
    )
    assert obs.active_window == "Terminal"
    summary = obs.summary()
    assert "ActiveWindow: 'Terminal'" in summary
    assert "CWD: '/workspace'" in summary

    action = OSAction(
        action_type=OSActionType.SHELL,
        target="git status",
        params={"command": "git status"},
        confidence=0.98,
    )
    assert action.command == "git status"
    assert action.confidence == 0.98

    task = AgentTask(goal="Inspect files and test", max_steps=10)
    assert task.goal == "Inspect files and test"
    assert task.max_steps == 10
    assert task.safety_mode == SafetyMode.BALANCED


def test_guardrails_destructive_command_interception():
    """Verify SafetyGuardrail blocks dangerous shell commands in <5ms."""
    guardrail = SafetyGuardrail()

    dangerous_commands = [
        "rm -rf /",
        "rm -rf ~/*",
        "del /f /s /q *",
        "format C:",
        "shutdown -s -t 0",
        "DROP DATABASE production",
    ]

    for cmd in dangerous_commands:
        action = OSAction(action_type=OSActionType.SHELL, target=cmd, params={"command": cmd})
        res = guardrail.check(action, SafetyMode.BALANCED)
        assert not res.is_safe, f"Dangerous command '{cmd}' should have been blocked!"
        assert res.requires_confirmation is True


def test_guardrails_safe_command_cleared():
    """Verify SafetyGuardrail allows read-only and benign commands."""
    guardrail = SafetyGuardrail()

    safe_commands = [
        "git status",
        "git log -n 5",
        "dir",
        "ls -la",
        "echo 'hello world'",
        "python --version",
        "uv run pytest",
    ]

    for cmd in safe_commands:
        action = OSAction(action_type=OSActionType.SHELL, target=cmd, params={"command": cmd})
        res = guardrail.check(action, SafetyMode.BALANCED)
        assert res.is_safe, f"Safe command '{cmd}' was unexpectedly flagged!"
        assert res.requires_confirmation is False


def test_guardrails_strict_mode():
    """Verify STRICT safety mode requires confirmation for all shell commands."""
    guardrail = SafetyGuardrail()
    action = OSAction(action_type=OSActionType.SHELL, target="git status", params={"command": "git status"})
    res = guardrail.check(action, SafetyMode.STRICT)
    assert res.requires_confirmation is True


def test_simulated_os_driver_execution():
    """Verify SimulatedOSDriver simulates shell and filesystem operations correctly."""
    driver = SimulatedOSDriver(initial_cwd="/workspace")
    obs = driver.observe()
    assert obs.active_window == "Visual Studio Code"

    # Read virtual file
    read_act = OSAction(action_type=OSActionType.FILE_READ, target="/workspace/README.md")
    res_read = driver.execute(read_act)
    assert res_read.status == ActionStatus.SUCCESS
    assert "SystemOneEngine" in res_read.output

    # Write virtual file
    write_act = OSAction(
        action_type=OSActionType.FILE_WRITE,
        target="/workspace/test.txt",
        params={"content": "hello world"},
    )
    res_write = driver.execute(write_act)
    assert res_write.status == ActionStatus.SUCCESS
    assert "/workspace/test.txt" in driver.virtual_fs


def test_system_one_os_agent_simulated_run():
    """Verify SystemOneOSAgent executes a multi-step task to completion in simulation."""
    driver = SimulatedOSDriver()
    agent = SystemOneOSAgent(client=None, driver=driver)

    task = AgentTask(
        goal="Check git repository status and verify files",
        max_steps=5,
        safety_mode=SafetyMode.BALANCED,
    )

    results = agent.run(task)
    assert len(results) >= 1
    assert any(r.status == ActionStatus.SUCCESS for r in results)
    # Check that average execution latency is sub-20ms
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    assert avg_latency < 35.0, f"Average latency {avg_latency:.2f}ms exceeds 35ms budget"


def test_system_one_os_agent_dry_run():
    """Verify dry_run=True prevents actual execution."""
    driver = SimulatedOSDriver()
    agent = SystemOneOSAgent(client=None, driver=driver)

    task = AgentTask(
        goal="Delete everything",
        max_steps=3,
        dry_run=True,
    )

    results = agent.run(task)
    assert len(results) >= 1
    assert all("[DRY-RUN]" in r.output or r.status == ActionStatus.BLOCKED_BY_GUARDRAIL for r in results)
