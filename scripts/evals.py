"""
Eval framework for the booking intake agent.

Runs a fixed set of test cases against run_intake() with all tool side-effects
(DB writes, email sends, Gingr writes) patched out. Only the LLM reasoning and
tool selection are exercised.

Usage:
    python scripts/evals.py            # run all cases
    python scripts/evals.py -v         # verbose: show agent trace for each case

Exit code 1 if any case fails (CI-friendly).
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

os.environ["DRY_RUN"] = "1"
os.environ.setdefault("OWNER_EMAIL", "test@example.com")
os.environ.setdefault("GINGR_API_KEY", "fake")
os.environ["DATABASE_URL"] = "postgresql://booking:booking@localhost:5432/booking"

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set in .env")
    sys.exit(1)

import app.agent as agent_module  # noqa: E402 — after env setup
from app.agent import run_intake   # noqa: E402


# ---------------------------------------------------------------------------
# Helpers for date assertions
# ---------------------------------------------------------------------------

def _parse(date_str: str) -> date:
    return date.fromisoformat(date_str)

def _start_of_next_week() -> date:
    """Return the Monday that starts the following calendar week."""
    today = date.today()
    days_to_monday = (7 - today.weekday()) % 7 or 7
    return today + timedelta(days=days_to_monday)

def _next_weekday(weekday: int) -> date:
    """Return the date of `weekday` (Mon=0 … Sun=6) in the following calendar week."""
    return _start_of_next_week() + timedelta(days=weekday)

def _this_weekday(weekday: int) -> date:
    """Return the upcoming occurrence of `weekday` within the current week (may be today)."""
    today = date.today()
    days_ahead = (weekday - today.weekday()) % 7
    return today + timedelta(days=days_ahead)


# ---------------------------------------------------------------------------
# EvalCase
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    name: str
    message: str
    sender_email: Optional[str]
    expected_tool: str                         # first tool Claude should call
    expected_args: dict = field(default_factory=dict)  # subset match
    date_check: Optional[Callable[[str], tuple[bool, str]]] = None
    # date_check returns (passed, reason_if_failed)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES: list[EvalCase] = [
    # --- Happy path / extraction ---
    EvalCase(
        name="full booking",
        message="Book grooming for Max on June 20th, my email is will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="no name has email",
        message="Grooming for Biscuit on June 20th, sarah@example.com",
        sender_email="sarah@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Biscuit", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="no name no email",
        message="Grooming for Rex on June 20th",
        sender_email=None,
        expected_tool="create_draft_booking",  # all required fields present; name/email optional
        expected_args={"pet_name": "Rex", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="with time",
        message="Grooming for Max on June 20th at 2pm, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": "2026-06-20"},
        date_check=lambda d: (True, ""),  # just check tool; time assertion below
    ),
    EvalCase(
        name="service is boarding",
        message="Book boarding for Luna on June 20th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Luna", "service": "boarding", "requested_date": "2026-06-20"},
    ),

    # --- Date resolution ---
    EvalCase(
        name="tomorrow",
        message="Grooming for Rex tomorrow, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex"},
        date_check=lambda d: (
            _parse(d) == date.today() + timedelta(days=1),
            f"expected {date.today() + timedelta(days=1)}, got {d}",
        ),
    ),
    EvalCase(
        # Use tomorrow's weekday name to avoid ambiguity when today == that weekday.
        name="this [tomorrow]",
        message=f"Grooming for Rex this {(date.today() + timedelta(days=1)).strftime('%A')}, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex"},
        date_check=lambda d: (
            _parse(d) == date.today() + timedelta(days=1),
            f"expected {date.today() + timedelta(days=1)}, got {d}",
        ),
    ),
    EvalCase(
        name="next Wednesday",
        message="Grooming for Luna next Wednesday, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Luna"},
        date_check=lambda d: (
            _parse(d) == _next_weekday(2),
            f"expected {_next_weekday(2)} (next week's Wednesday), got {d}",
        ),
    ),
    EvalCase(
        name="next Monday",
        message="Grooming for Luna next Monday, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Luna"},
        date_check=lambda d: (
            _parse(d) == _next_weekday(0),
            f"expected {_next_weekday(0)} (next week's Monday), got {d}",
        ),
    ),

    # --- Clarification required ---
    EvalCase(
        name="missing pet",
        message="I need grooming on June 20th, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="missing date",
        message="Grooming for Max please, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="missing service",
        message="Book something for Max on June 20th, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="vague date",
        message="Boarding for Luna sometime next week, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

FAKE_RETURNS = {
    "create_draft_booking": "Booking created successfully. booking_id=99",
    "notify_owners": "Owner notified about booking #99.",
    "send_clarification_email": "Clarification email sent.",
}


async def run_case(case: EvalCase, verbose: bool) -> tuple[bool, str]:
    """
    Run a single eval case. Returns (passed, failure_reason).
    Patches all tools to recorders, restores them after.
    """
    recorded_calls: list[dict] = []

    # Patch
    original_funcs = {}
    for t in agent_module.TOOLS:
        original_funcs[t.name] = t.func
        def make_recorder(name):
            def recorder(**kwargs):
                recorded_calls.append({"tool": name, "args": kwargs})
                return FAKE_RETURNS[name]
            return recorder
        t.func = make_recorder(t.name)

    # Suppress verbose agent trace unless -v
    if not verbose:
        import logging
        logging.disable(logging.CRITICAL)

    try:
        await run_intake(
            message_body=case.message,
            sender_email=case.sender_email,
            source_channel="email",
        )
    except Exception as exc:
        return False, f"run_intake raised: {exc}"
    finally:
        # Restore
        for t in agent_module.TOOLS:
            t.func = original_funcs[t.name]
        if not verbose:
            logging.disable(logging.NOTSET)

    if not recorded_calls:
        return False, "no tools were called"

    first_call = recorded_calls[0]
    actual_tool = first_call["tool"]
    actual_args = first_call["args"]

    # 1. Correct tool?
    if actual_tool != case.expected_tool:
        return False, f"expected tool={case.expected_tool}, got tool={actual_tool}"

    # 2. Expected args subset match (case-insensitive for string values)?
    for key, expected_val in case.expected_args.items():
        actual_val = actual_args.get(key)
        if str(actual_val).lower() != str(expected_val).lower():
            return False, f"args[{key}]: expected {expected_val!r}, got {actual_val!r}"

    # 3. Date check?
    if case.date_check and "requested_date" in actual_args:
        passed, reason = case.date_check(actual_args["requested_date"])
        if not passed:
            return False, f"date_check failed: {reason}"

    # 4. "with time" case: verify requested_time was extracted
    if case.name == "with time":
        if not actual_args.get("requested_time"):
            return False, f"expected requested_time to be set, got {actual_args.get('requested_time')!r}"

    return True, ""


async def main(verbose: bool):
    print(f"\nRunning {len(CASES)} eval cases (today = {date.today().isoformat()} {date.today().strftime('%A')})\n")

    results = []
    for case in CASES:
        passed, reason = await run_case(case, verbose)
        results.append((case.name, passed, reason))
        status = "PASS" if passed else "FAIL"
        line = f"  [ {status} ]  {case.name}"
        if not passed:
            line += f"\n           → {reason}"
        print(line)

    n_passed = sum(1 for _, p, _ in results if p)
    n_total = len(results)
    print(f"\nResults: {n_passed}/{n_total} passed\n")

    if n_passed < n_total:
        sys.exit(1)


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    asyncio.run(main(verbose))
