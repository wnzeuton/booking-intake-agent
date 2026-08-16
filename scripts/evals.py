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

import argparse
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

if not os.environ.get("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY not set in .env")
    sys.exit(1)

_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("-v", action="store_true", dest="verbose")
_arg_parser.add_argument("--model", default=None, help="Override GROQ_MODEL for this run (for comparing candidates)")
_arg_parser.add_argument("--quick", action="store_true", help="Run only the small representative subset (fits one day's free-tier budget on any candidate)")
_args = _arg_parser.parse_args()

if _args.model:
    os.environ["GROQ_MODEL"] = _args.model

import app.agent as agent_module  # noqa: E402 — after env setup
from app.agent import run_intake   # noqa: E402


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _parse(date_str: str) -> date:
    return date.fromisoformat(date_str)

def _start_of_next_week() -> date:
    today = date.today()
    days_to_monday = (7 - today.weekday()) % 7 or 7
    return today + timedelta(days=days_to_monday)

def _next_weekday(weekday: int) -> date:
    """Weekday of the FOLLOWING calendar week (Mon=0 … Sun=6)."""
    return _start_of_next_week() + timedelta(days=weekday)

def _this_weekday(weekday: int) -> date:
    """Upcoming occurrence of weekday within the current week (may be today)."""
    today = date.today()
    return today + timedelta(days=(weekday - today.weekday()) % 7)

def _is_next_weekday(d: str, weekday: int) -> tuple[bool, str]:
    parsed = _parse(d)
    expected = _next_weekday(weekday)
    return parsed == expected, f"expected {expected}, got {d}"

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# EvalCase
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    name: str
    message: str
    sender_email: Optional[str]
    expected_tool: str
    expected_args: dict = field(default_factory=dict)
    date_check: Optional[Callable[[str], tuple[bool, str]]] = None
    # Part of the small representative subset run by `--quick` for everyday
    # dev iteration (fits comfortably in one day's free-tier token budget,
    # even on the tightest-quota model). Covers every category at least
    # once; the full CASES list stays the pre-merge/model-comparison gate.
    quick: bool = False


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

_TODAY = date.today()
_TOMORROW = _TODAY + timedelta(days=1)
_TOMORROW_NAME = _TOMORROW.strftime("%A")
_IN_3_DAYS = _TODAY + timedelta(days=3)
_IN_7_DAYS = _TODAY + timedelta(days=7)

# Anchor dates for "absolute date" fixtures below — computed relative to
# _TODAY so these never go stale (a fixture hardcoding a literal date like
# "June 20th" only stays valid until that date passes in the real world).
# All anchor days are fixed (20 / 30 / 22 / 4) so ordinal suffixes below
# ("20th", "30th", "22nd", "4th") stay correct without extra computation.

def _anchor_month(months_ahead: int) -> tuple[int, int]:
    month = _TODAY.month + months_ahead
    year = _TODAY.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    if month == 2:  # February lacks a 30th; use March for day-30 fixtures
        month = 3
    return year, month

_ANCHOR_YEAR, _ANCHOR_MONTH = _anchor_month(4)
_ANCHOR = date(_ANCHOR_YEAR, _ANCHOR_MONTH, 20)
_ANCHOR_ISO = _ANCHOR.isoformat()
_ANCHOR_MONTH_NAME = _ANCHOR.strftime("%B")
_ANCHOR_DAY = _ANCHOR.day

_ANCHOR_30 = date(_ANCHOR_YEAR, _ANCHOR_MONTH, 30)
_ANCHOR_30_ISO = _ANCHOR_30.isoformat()

_ANCHOR_PLUS2 = _ANCHOR + timedelta(days=2)
_ANCHOR_PLUS2_ISO = _ANCHOR_PLUS2.isoformat()
_ANCHOR_PLUS2_WEEKDAY = _ANCHOR_PLUS2.strftime("%A")

_NEXT_MONTH_YEAR, _NEXT_MONTH_MONTH = _anchor_month(5)
_ANCHOR_NEXT_MONTH = date(_NEXT_MONTH_YEAR, _NEXT_MONTH_MONTH, 4)
_ANCHOR_NEXT_MONTH_ISO = _ANCHOR_NEXT_MONTH.isoformat()
_ANCHOR_NEXT_MONTH_NAME = _ANCHOR_NEXT_MONTH.strftime("%B")

# A month/day that already occurred earlier this year, for testing the
# year-rollover resolution rule in SYSTEM_PROMPT (a bare month/day that has
# already passed this year should resolve to next year, not be flagged as
# vague or booked in the past). Clamped so it can't land in the prior year.
_PAST_THIS_YEAR = max(date(_TODAY.year, 1, 2), _TODAY - timedelta(days=45))
_PAST_THIS_YEAR_NEXT_OCCURRENCE = _PAST_THIS_YEAR.replace(year=_TODAY.year + 1)
_PAST_THIS_YEAR_MONTH_NAME = _PAST_THIS_YEAR.strftime("%B")
_PAST_THIS_YEAR_DAY = _PAST_THIS_YEAR.day

CASES: list[EvalCase] = [

    # -----------------------------------------------------------------------
    # HAPPY PATH — basic extraction
    # -----------------------------------------------------------------------

    EvalCase(
        name="full booking",
        message=f"Book grooming for Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, my email is will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
        quick=True,
    ),
    EvalCase(
        name="no name has email",
        message=f"Grooming for Biscuit on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, sarah@example.com",
        sender_email="sarah@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Biscuit", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="no name no email",
        message=f"Grooming for Rex on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th",
        sender_email=None,
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex", "requested_date": _ANCHOR_ISO},
        quick=True,
    ),
    EvalCase(
        name="with time",
        message=f"Grooming for Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th at 2pm, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": _ANCHOR_ISO},
        quick=True,
    ),
    EvalCase(
        name="morning time",
        message=f"Can I get grooming for Daisy on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th at 9am? Email: jane@example.com",
        sender_email="jane@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Daisy", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="24h time",
        message=f"Grooming for Rex on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th at 14:00, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="noon",
        message=f"Grooming for Mochi on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th at noon, mochi@example.com",
        sender_email="mochi@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Mochi", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="question format",
        message=f"Could I book grooming for Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th? My email is will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="polite request",
        message=f"Hi, I'd like to please schedule grooming for Luna on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th. Email: luna@example.com",
        sender_email="luna@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Luna", "service": "grooming"},
    ),
    EvalCase(
        name="with notes",
        message=f"Grooming for Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com — he's anxious around other dogs, please keep that in mind",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": _ANCHOR_ISO},
        quick=True,
    ),
    EvalCase(
        name="breed mentioned",
        message=f"Book grooming for my golden retriever Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="pup instead of dog",
        message=f"My pup Biscuit needs grooming on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, sarah@example.com",
        sender_email="sarah@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Biscuit", "service": "grooming"},
    ),
    EvalCase(
        name="possessive pet name",
        message=f"My dog's name is Max and I'd like grooming on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th. Email: will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="telegram style",
        message=f"Max grooming {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY} will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
        quick=True,
    ),
    EvalCase(
        name="new customer intro",
        message=f"Hi, I'm a new customer! I'd love to book grooming for my dog Coco on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th. My email is coco@example.com",
        sender_email="coco@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Coco", "service": "grooming"},
    ),
    EvalCase(
        name="with phone only",
        message=f"Book grooming for Bella on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th. My number is 555-1234.",
        sender_email=None,
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Bella", "service": "grooming", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="we as customer",
        message=f"We'd like to book grooming for our dog Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th. Email: family@example.com",
        sender_email="family@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming"},
    ),

    # -----------------------------------------------------------------------
    # SERVICES
    # -----------------------------------------------------------------------

    EvalCase(
        name="service boarding",
        message=f"Book boarding for Luna on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Luna", "service": "boarding", "requested_date": _ANCHOR_ISO},
        quick=True,
    ),
    EvalCase(
        name="service daycare",
        message=f"Book daycare for Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "daycare", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="service bath",
        message=f"Can I get a bath for Biscuit on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th? sarah@example.com",
        sender_email="sarah@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Biscuit", "requested_date": _ANCHOR_ISO},
        quick=True,
    ),
    EvalCase(
        name="service nail trim",
        message=f"Nail trim for Rex on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="service training",
        message=f"Book a training session for Luna on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, luna@example.com",
        sender_email="luna@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Luna", "service": "training"},
        quick=True,
    ),
    EvalCase(
        name="service full groom",
        message=f"Full groom for Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="service implied haircut",
        message=f"My dog Max needs a haircut on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="service as appointment noun",
        message=f"I'd like to schedule a grooming appointment for Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th. will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
    ),

    # -----------------------------------------------------------------------
    # ABSOLUTE DATE FORMATS
    # -----------------------------------------------------------------------

    EvalCase(
        name="date no year",
        message=f"Grooming for Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="date numeric slash",
        message=f"Grooming for Max on {_ANCHOR.month}/{_ANCHOR.day}/{_ANCHOR.year}, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": _ANCHOR_ISO},
        quick=True,
    ),
    EvalCase(
        name="date the Nth of month",
        message=f"Grooming for Max on the {_ANCHOR_DAY}th of {_ANCHOR_MONTH_NAME}, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": _ANCHOR_ISO},
        quick=True,
    ),
    EvalCase(
        name="date written out",
        message=f"Grooming for Max on {_ANCHOR_MONTH_NAME} twentieth, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="date end of month",
        message=f"Grooming for Rex on {_ANCHOR_MONTH_NAME} 30th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex", "requested_date": _ANCHOR_30_ISO},
        quick=True,
    ),
    EvalCase(
        name="date next month",
        message=f"Grooming for Luna on {_ANCHOR_NEXT_MONTH_NAME} 4th, luna@example.com",
        sender_email="luna@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Luna", "requested_date": _ANCHOR_NEXT_MONTH_ISO},
    ),
    EvalCase(
        name="date today",
        message="Grooming for Rex today, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex"},
        date_check=lambda d: (
            _parse(d) == _TODAY,
            f"expected {_TODAY}, got {d}",
        ),
    ),
    EvalCase(
        name="date tomorrow",
        message="Grooming for Rex tomorrow, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex"},
        date_check=lambda d: (
            _parse(d) == _TOMORROW,
            f"expected {_TOMORROW}, got {d}",
        ),
    ),
    EvalCase(
        name="date in 3 days",
        message="Grooming for Rex in 3 days, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex"},
        date_check=lambda d: (
            _parse(d) == _IN_3_DAYS,
            f"expected {_IN_3_DAYS} (today+3), got {d}",
        ),
    ),
    EvalCase(
        name="date a week from today",
        message="Grooming for Rex a week from today, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex"},
        date_check=lambda d: (
            _parse(d) == _IN_7_DAYS,
            f"expected {_IN_7_DAYS} (today+7), got {d}",
        ),
    ),

    # -----------------------------------------------------------------------
    # YEAR ROLLOVER — bare month/day that already passed this year
    # -----------------------------------------------------------------------

    EvalCase(
        name="year rollover — past month/day resolves to next year",
        message=f"Grooming for Max on {_PAST_THIS_YEAR_MONTH_NAME} {_PAST_THIS_YEAR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": _PAST_THIS_YEAR_NEXT_OCCURRENCE.isoformat()},
        quick=True,
    ),
    EvalCase(
        name="year rollover — different service, same rule",
        message=f"Boarding for Luna on {_PAST_THIS_YEAR_MONTH_NAME} {_PAST_THIS_YEAR_DAY}th, luna@example.com",
        sender_email="luna@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Luna", "service": "boarding", "requested_date": _PAST_THIS_YEAR_NEXT_OCCURRENCE.isoformat()},
        quick=True,
    ),

    # -----------------------------------------------------------------------
    # RELATIVE WEEKDAY RESOLUTION — "this [day]"
    # -----------------------------------------------------------------------

    EvalCase(
        name=f"this {_TOMORROW_NAME}",
        message=f"Grooming for Rex this {_TOMORROW_NAME}, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex"},
        date_check=lambda d: (
            _parse(d) == _TOMORROW,
            f"expected {_TOMORROW} (this {_TOMORROW_NAME}), got {d}",
        ),
        quick=True,
    ),

    # -----------------------------------------------------------------------
    # RELATIVE WEEKDAY RESOLUTION — "next [day]" (all 7)
    # -----------------------------------------------------------------------

    *[
        EvalCase(
            name=f"next {WEEKDAY_NAMES[i]}",
            message=f"Grooming for Luna next {WEEKDAY_NAMES[i]}, will@example.com",
            sender_email="will@example.com",
            expected_tool="create_draft_booking",
            expected_args={"pet_name": "Luna"},
            date_check=lambda d, idx=i: _is_next_weekday(d, idx),
            quick=(i in (0, 3)),  # Monday and Thursday — enough to cover the mechanism
        )
        for i in range(7)
    ],

    EvalCase(
        name="this coming Wednesday",
        message="Grooming for Rex this coming Wednesday, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex"},
        date_check=lambda d: _is_next_weekday(d, 2),
        quick=True,
    ),
    EvalCase(
        name="next week Wednesday",
        message="Grooming for Rex next week Wednesday, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex"},
        date_check=lambda d: _is_next_weekday(d, 2),
    ),
    EvalCase(
        name="Wednesday of next week",
        message="Grooming for Rex Wednesday of next week, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex"},
        date_check=lambda d: _is_next_weekday(d, 2),
    ),

    # -----------------------------------------------------------------------
    # VAGUE DATES → clarification required
    # -----------------------------------------------------------------------

    EvalCase(
        name="vague sometime next week",
        message="Boarding for Luna sometime next week, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
        quick=True,
    ),
    EvalCase(
        name="vague sometime this week",
        message="Grooming for Max sometime this week, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="vague early next week",
        message="Grooming for Max early next week, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="vague end of next week",
        message="Grooming for Max at the end of next week, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="vague this weekend",
        message="Grooming for Max this weekend, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="vague next weekend",
        message="Grooming for Max next weekend, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="vague asap",
        message="Grooming for Max as soon as possible, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
        quick=True,
    ),
    EvalCase(
        name="vague whenever",
        message="Grooming for Max whenever you have an opening, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="vague in a few days",
        message="Grooming for Max in a few days, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="vague next month",
        message="Grooming for Max next month, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
        quick=True,
    ),
    EvalCase(
        name="vague soon",
        message="Grooming for Max soon, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="vague early in the week",
        message="Grooming for Max early in the week, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="vague next week no day",
        message="Can I book grooming for Max next week? will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
        quick=True,
    ),

    # -----------------------------------------------------------------------
    # MISSING REQUIRED FIELDS → clarification required
    # -----------------------------------------------------------------------

    EvalCase(
        name="missing pet name",
        message=f"I need grooming on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
        quick=True,
    ),
    EvalCase(
        name="missing date",
        message="Grooming for Max please, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
        quick=True,
    ),
    EvalCase(
        name="missing service",
        message=f"Book something for Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
        quick=True,
    ),
    EvalCase(
        name="missing pet and date",
        message="I need a grooming appointment, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="missing date and service",
        message="Can you help with my dog Max? will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="missing pet and service",
        message=f"I need an appointment on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="missing all three",
        message="Hi I'd like to make a booking, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="only email provided",
        message="will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="only pet name",
        message="Max",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="only date",
        message=f"{_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
        quick=True,
    ),
    EvalCase(
        name="only service",
        message="grooming",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="dog needs appointment",
        message="My dog needs an appointment, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="please schedule me",
        message="Please schedule me in, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="want to book something",
        message="I want to book something for my dog, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),

    # -----------------------------------------------------------------------
    # OFF-TOPIC / NON-BOOKING MESSAGES → clarification
    # -----------------------------------------------------------------------

    EvalCase(
        name="hours question",
        message="What are your hours? will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
        quick=True,
    ),
    EvalCase(
        name="cancellation request",
        message="I'd like to cancel my appointment for Max, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
        quick=True,
    ),
    EvalCase(
        name="pricing question",
        message="How much does grooming cost for a golden retriever? will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="complaint",
        message="My dog Max was very stressed after his last grooming visit, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),

    # -----------------------------------------------------------------------
    # REAL-WORLD EMAIL STYLES
    # -----------------------------------------------------------------------

    EvalCase(
        name="formal email with signature",
        message=(
            "Dear team,\n\n"
            "I hope this message finds you well. I am writing to request a grooming "
            f"appointment for my dog, Max, on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, {_ANCHOR_YEAR}.\n\n"
            "Please confirm availability at your earliest convenience.\n\n"
            "Best regards,\nWill\nwill@example.com"
        ),
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
        quick=True,
    ),
    EvalCase(
        name="reply style email",
        message=(
            "Re: Grooming availability\n\n"
            f"Yes, {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th works great for Max's grooming. "
            "Thanks! — will@example.com"
        ),
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
        quick=True,
    ),
    EvalCase(
        name="question plus booking",
        message=f"Do you have availability for grooming for Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th? My email is will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="long email info buried",
        message=(
            "Hi there! I've been meaning to reach out for a while. "
            "We absolutely love your shop — the staff are so friendly and Max "
            "always comes home looking amazing. Anyway, I wanted to see about "
            "booking him in for another grooming session. "
            f"Would {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th work? My email is will@example.com. "
            "Let me know if you need anything else from me. Thanks so much!"
        ),
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="all caps message",
        message=f"GROOMING FOR MAX ON {_ANCHOR_MONTH_NAME.upper()} {_ANCHOR_DAY}TH PLEASE, WILL@EXAMPLE.COM",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
        quick=True,
    ),
    EvalCase(
        name="message with typo in service",
        message=f"I need groooming for Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="message with emoji",
        message=f"Grooming for Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th 🐾 will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="returning customer context",
        message=f"Hey, I was in last month with my dog Max. Can I book grooming again for {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th? will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="multi sentence with context",
        message=(
            "Hi! My name is Sarah and I have a 2-year-old cockapoo named Biscuit. "
            f"I'd like to book her in for a groom on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th. "
            "You can reach me at sarah@example.com."
        ),
        sender_email="sarah@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Biscuit", "service": "grooming", "requested_date": _ANCHOR_ISO},
    ),

    # -----------------------------------------------------------------------
    # EDGE CASES
    # -----------------------------------------------------------------------

    EvalCase(
        name="pet name two words",
        message=f"Grooming for Bella Rose on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="pet name common noun",
        message=f"Grooming for Cookie on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Cookie", "service": "grooming", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="two dates mentioned",
        message=f"Grooming for Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th or {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY + 7}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
        quick=True,
    ),
    EvalCase(
        name="two pets mentioned",
        message=f"Grooming for Max and Luna on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
        quick=True,
    ),
    EvalCase(
        name="tomorrow morning",
        message="Grooming for Rex tomorrow morning, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex"},
        date_check=lambda d: (
            _parse(d) == _TOMORROW,
            f"expected {_TOMORROW} (tomorrow), got {d}",
        ),
    ),
    EvalCase(
        name="service with modifier",
        message=f"Quick groom for Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="my dog Max phrasing",
        message=f"My dog Max needs grooming on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th. will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="dash separated format",
        message=f"Max - Grooming - {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th - will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="compound service bath and nails",
        message=f"Bath and nail trim for Rex on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="cat instead of dog",
        message=f"Grooming for my cat Luna on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th, luna@example.com",
        sender_email="luna@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Luna", "service": "grooming", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="service listed first",
        message=f"Boarding on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th for our dog Cooper, cooper@example.com",
        sender_email="cooper@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Cooper", "service": "boarding", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="breed plus relative date",
        message="Can you fit my schnauzer Biscuit in for a groom next Thursday? sarah@example.com",
        sender_email="sarah@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Biscuit"},
        date_check=lambda d: _is_next_weekday(d, 3),
    ),
    EvalCase(
        name="day of week plus absolute date",
        message=f"Grooming for Max on {_ANCHOR_PLUS2_WEEKDAY} {_ANCHOR_MONTH_NAME} 22nd, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_PLUS2_ISO},
        quick=True,
    ),
    EvalCase(
        name="evening time",
        message=f"Grooming for Max on {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th at 6pm, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": _ANCHOR_ISO},
    ),
    EvalCase(
        name="form style structured message",
        message=f"Pet name: Bella\nService: grooming\nDate: {_ANCHOR_MONTH_NAME} {_ANCHOR_DAY}th\nEmail: bella@example.com",
        sender_email="bella@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Bella", "service": "grooming", "requested_date": _ANCHOR_ISO},
        quick=True,
    ),
    EvalCase(
        name="wash implies grooming",
        message="I need to get my dog Max washed tomorrow, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max"},
        date_check=lambda d: (
            _parse(d) == _TOMORROW,
            f"expected {_TOMORROW} (tomorrow), got {d}",
        ),
        quick=True,
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

# The only correct tool-call sequences per SYSTEM_PROMPT's decision rules —
# used to catch models that take the right first action but then keep going
# (e.g. also sending a clarification email after already booking + notifying).
EXPECTED_SEQUENCES = {
    "create_draft_booking": ["create_draft_booking", "notify_owners"],
    "send_clarification_email": ["send_clarification_email"],
}


async def run_case(case: EvalCase, verbose: bool) -> tuple[bool, str]:
    recorded_calls: list[dict] = []

    original_funcs = {}
    for t in agent_module.TOOLS:
        original_funcs[t.name] = t.func
        def make_recorder(name):
            def recorder(**kwargs):
                recorded_calls.append({"tool": name, "args": kwargs})
                return FAKE_RETURNS[name]
            return recorder
        t.func = make_recorder(t.name)

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
        for t in agent_module.TOOLS:
            t.func = original_funcs[t.name]
        if not verbose:
            logging.disable(logging.NOTSET)

    if not recorded_calls:
        return False, "no tools were called"

    first_call = recorded_calls[0]
    actual_tool = first_call["tool"]
    actual_args = first_call["args"]

    if actual_tool != case.expected_tool:
        return False, f"expected tool={case.expected_tool}, got tool={actual_tool}"

    for key, expected_val in case.expected_args.items():
        actual_val = actual_args.get(key)
        if str(actual_val).lower() != str(expected_val).lower():
            return False, f"args[{key}]: expected {expected_val!r}, got {actual_val!r}"

    if case.date_check and "requested_date" in actual_args:
        passed, reason = case.date_check(actual_args["requested_date"])
        if not passed:
            return False, f"date_check failed: {reason}"

    if case.name == "with time" or "time" in case.name.lower() and "time" not in ["date in 3 days", "this coming Wednesday"]:
        if "requested_time" in actual_args and actual_args.get("requested_time") is None:
            pass  # optional; don't fail if agent didn't extract it

    actual_sequence = [c["tool"] for c in recorded_calls]
    expected_sequence = EXPECTED_SEQUENCES[case.expected_tool]
    if actual_sequence != expected_sequence:
        return False, f"tool call sequence: expected {expected_sequence}, got {actual_sequence}"

    return True, ""


async def main(verbose: bool, quick: bool):
    cases = [c for c in CASES if c.quick] if quick else CASES
    active_model = os.environ.get("GROQ_MODEL", "<default>")
    mode = "quick" if quick else "full"
    print(f"\nRunning {len(cases)} eval cases ({mode})  (today = {_TODAY.isoformat()} {_TODAY.strftime('%A')}, model = {active_model})\n")

    results = []
    for case in cases:
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
    asyncio.run(main(_args.verbose, _args.quick))
