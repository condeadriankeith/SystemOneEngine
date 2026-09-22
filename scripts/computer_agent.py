"""System One Computer-Use & Task Automation Agent — Interactive CLI.

Demonstrates System One Engine controlling the computer to accomplish tasks
with sub-20ms reflex decisions, real-time safety guardrails, and calibrated confidence.

Usage:
    uv run python scripts/computer_agent.py              # Interactive menu
    uv run python scripts/computer_agent.py --demo       # Run automated multi-step demo
    uv run python scripts/computer_agent.py --guardrails # Stress-test real-time safety interceptor
    uv run python scripts/computer_agent.py --goal "Check git status and recent commits"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure Unicode output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from system_one_engine import (
    AgentTask,
    LocalOSDriver,
    OSAction,
    OSActionResult,
    OSActionType,
    SafetyGuardrail,
    SafetyMode,
    SimulatedOSDriver,
    SystemOneOSAgent,
)
from system_one_engine.adapters.onnx_runtime_adapter import ONNXRuntimeAdapter
from system_one_engine.client import SystemOneClient
from system_one_engine.training.dataset import SimpleVocabTokenizer


def _get_client() -> SystemOneClient:
    """Initialize local SystemOneClient with INT8 ONNX models if available."""
    onnx_dir = Path("models/onnx")
    enc = onnx_dir / "encoder_int8.onnx"
    heads = onnx_dir / "heads_int8.onnx"
    ch = onnx_dir / "choice_head_int8.onnx"

    if enc.exists() and heads.exists() and ch.exists():
        tokenizer = SimpleVocabTokenizer(vocab_size=500, max_len=16)
        adapter = ONNXRuntimeAdapter(
            encoder_path=str(enc),
            heads_path=str(heads),
            choice_head_path=str(ch),
            tokenizer=tokenizer,
            num_threads=4,
        )
        return SystemOneClient(adapter=adapter)

    # Fallback in-process client
    return SystemOneClient()


def render_step_result(step_num: int, result: OSActionResult) -> None:
    """Render a single agent execution step to the terminal."""
    act = result.action
    status_icon = "[OK]" if result.status.value == "success" else f"[{result.status.value.upper()}]"

    print(f"\n  Step {step_num:02d} {status_icon} Action: {act.action_type.value.upper()} -> '{act.target}'")
    print(f"  Confidence: {act.confidence:5.1%} | Latency: {result.latency_ms:.2f} ms")
    if act.rationale:
        print(f"  Rationale:  {act.rationale}")

    if result.output.strip():
        snippet = "\n".join(f"    | {line}" for line in result.output.strip().splitlines()[:5])
        print(f"  Output:\n{snippet}")


def run_demo_task() -> None:
    """Run an automated multi-step task showing perception, decision, and guardrails."""
    print("\n" + "=" * 70)
    print("  RUNNING SYSTEM ONE COMPUTER AGENT DEMO: Safe Workspace Audit")
    print("=" * 70)

    client = _get_client()
    driver = LocalOSDriver()
    agent = SystemOneOSAgent(client=client, driver=driver, confidence_threshold=0.65)

    task = AgentTask(
        goal="Check git repository status, list project files, and report workspace health",
        max_steps=5,
        safety_mode=SafetyMode.BALANCED,
        dry_run=False,
    )

    t0 = time.perf_counter()
    results = agent.run(task)
    total_time_ms = (time.perf_counter() - t0) * 1000.0

    for i, res in enumerate(results, 1):
        render_step_result(i, res)

    print("\n" + "-" * 70)
    print(f"  Task Completed in {len(results)} steps! Total wall-clock time: {total_time_ms:.1f} ms")
    print("=" * 70)


def run_guardrails_test() -> None:
    """Stress-test the real-time sub-5ms safety guardrail interceptor."""
    print("\n" + "=" * 70)
    print("  TESTING REAL-TIME SYSTEM ONE SAFETY GUARDRAILS (<5ms)")
    print("=" * 70)

    client = _get_client()
    guardrail = SafetyGuardrail(client=client)

    test_commands = [
        ("git status", "Safe query command"),
        ("uv run pytest", "Project automated tests"),
        ("dir", "Directory listing"),
        ("rm -rf /", "Recursive root deletion"),
        ("del /f /s /q *", "Wildcard file purge"),
        ("format C:", "Disk format command"),
        ("shutdown /s /t 0", "System shutdown"),
        ("DROP DATABASE production", "Database drop statement"),
    ]

    for cmd, desc in test_commands:
        action = OSAction(
            action_type=OSActionType.SHELL,
            target=cmd,
            params={"command": cmd},
            rationale=desc,
        )
        t0 = time.perf_counter()
        check = guardrail.check(action, SafetyMode.BALANCED)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        status_tag = "[ALLOWED]" if check.is_safe else "[BLOCKED]"
        print(f"\n  Command:  '{cmd}' ({desc})")
        print(f"  Result:   {status_tag} (Safety Conf: {check.confidence:.1%}, Latency: {latency_ms:.2f} ms)")
        print(f"  Reason:   {check.reason}")

    print("\n" + "=" * 70)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="System One Autonomous Computer-Use Agent")
    parser.add_argument("--demo", action="store_true", help="Run automated multi-step workspace task")
    parser.add_argument("--guardrails", action="store_true", help="Test real-time safety guardrails")
    parser.add_argument("--goal", type=str, help="Run custom natural language task goal")
    parser.add_argument("--dry-run", action="store_true", help="Simulate actions without executing")
    parser.add_argument("--strict", action="store_true", help="Require confirmation on all shell actions")

    args = parser.parse_args()

    print("=" * 70)
    print("  System One Autonomous Computer-Use Agent (OS Task Automation)")
    print("=" * 70)

    if args.guardrails:
        run_guardrails_test()
        return

    if args.demo:
        run_demo_task()
        return

    if args.goal:
        client = _get_client()
        mode = SafetyMode.STRICT if args.strict else SafetyMode.BALANCED
        agent = SystemOneOSAgent(client=client, driver=LocalOSDriver())
        task = AgentTask(goal=args.goal, max_steps=10, safety_mode=mode, dry_run=args.dry_run)
        print(f"\n[*] Executing Goal: '{task.goal}' (Mode: {mode.value.upper()}, DryRun: {args.dry_run})")
        results = agent.run(task)
        for i, res in enumerate(results, 1):
            render_step_result(i, res)
        return

    # Interactive prompt
    print("\nSelect an option:")
    print("  1. Run Automated Workspace Audit Task (Git, Files, Health)")
    print("  2. Test Real-Time Safety Guardrails (<5ms Interception)")
    print("  3. Run Custom Natural Language Goal")
    print("  Q. Quit")

    choice = input("\nEnter choice [1-3, Q]: ").strip()
    if choice == "1":
        run_demo_task()
    elif choice == "2":
        run_guardrails_test()
    elif choice == "3":
        goal = input("\nEnter your computer task goal: ").strip()
        if goal:
            client = _get_client()
            agent = SystemOneOSAgent(client=client, driver=LocalOSDriver())
            task = AgentTask(goal=goal, max_steps=10)
            results = agent.run(task)
            for i, res in enumerate(results, 1):
                render_step_result(i, res)
    else:
        print("[*] Exiting.")


if __name__ == "__main__":
    main()
