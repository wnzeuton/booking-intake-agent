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

if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY not set in .env")
    sys.exit(1)

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


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

_TODAY = date.today()
_TOMORROW = _TODAY + timedelta(days=1)
_TOMORROW_NAME = _TOMORROW.strftime("%A")
_IN_3_DAYS = _TODAY + timedelta(days=3)
_IN_7_DAYS = _TODAY + timedelta(days=7)

CASES: list[EvalCase] = [

    # -----------------------------------------------------------------------
    # HAPPY PATH — basic extraction
    # -----------------------------------------------------------------------

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
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="with time",
        message="Grooming for Max on June 20th at 2pm, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="morning time",
        message="Can I get grooming for Daisy on June 20th at 9am? Email: jane@example.com",
        sender_email="jane@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Daisy", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="24h time",
        message="Grooming for Rex on June 20th at 14:00, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="noon",
        message="Grooming for Mochi on June 20th at noon, mochi@example.com",
        sender_email="mochi@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Mochi", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="question format",
        message="Could I book grooming for Max on June 20th? My email is will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="polite request",
        message="Hi, I'd like to please schedule grooming for Luna on June 20th. Email: luna@example.com",
        sender_email="luna@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Luna", "service": "grooming"},
    ),
    EvalCase(
        name="with notes",
        message="Grooming for Max on June 20th, will@example.com — he's anxious around other dogs, please keep that in mind",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="breed mentioned",
        message="Book grooming for my golden retriever Max on June 20th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="pup instead of dog",
        message="My pup Biscuit needs grooming on June 20th, sarah@example.com",
        sender_email="sarah@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Biscuit", "service": "grooming"},
    ),
    EvalCase(
        name="possessive pet name",
        message="My dog's name is Max and I'd like grooming on June 20th. Email: will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="telegram style",
        message="Max grooming June 20 will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="new customer intro",
        message="Hi, I'm a new customer! I'd love to book grooming for my dog Coco on June 20th. My email is coco@example.com",
        sender_email="coco@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Coco", "service": "grooming"},
    ),
    EvalCase(
        name="with phone only",
        message="Book grooming for Bella on June 20th. My number is 555-1234.",
        sender_email=None,
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Bella", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="we as customer",
        message="We'd like to book grooming for our dog Max on June 20th. Email: family@example.com",
        sender_email="family@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming"},
    ),

    # -----------------------------------------------------------------------
    # SERVICES
    # -----------------------------------------------------------------------

    EvalCase(
        name="service boarding",
        message="Book boarding for Luna on June 20th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Luna", "service": "boarding", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="service daycare",
        message="Book daycare for Max on June 20th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "daycare", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="service bath",
        message="Can I get a bath for Biscuit on June 20th? sarah@example.com",
        sender_email="sarah@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Biscuit", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="service nail trim",
        message="Nail trim for Rex on June 20th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="service training",
        message="Book a training session for Luna on June 20th, luna@example.com",
        sender_email="luna@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Luna", "service": "training"},
    ),
    EvalCase(
        name="service full groom",
        message="Full groom for Max on June 20th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="service implied haircut",
        message="My dog Max needs a haircut on June 20th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="service as appointment noun",
        message="I'd like to schedule a grooming appointment for Max on June 20th. will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),

    # -----------------------------------------------------------------------
    # ABSOLUTE DATE FORMATS
    # -----------------------------------------------------------------------

    EvalCase(
        name="date no year",
        message="Grooming for Max on June 20, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="date numeric slash",
        message="Grooming for Max on 6/20/2026, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="date the Nth of month",
        message="Grooming for Max on the 20th of June, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="date written out",
        message="Grooming for Max on June twentieth, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="date end of month",
        message="Grooming for Rex on June 30th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex", "requested_date": "2026-06-30"},
    ),
    EvalCase(
        name="date next month",
        message="Grooming for Luna on July 4th, luna@example.com",
        sender_email="luna@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Luna", "requested_date": "2026-07-04"},
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
    ),

    # -----------------------------------------------------------------------
    # MISSING REQUIRED FIELDS → clarification required
    # -----------------------------------------------------------------------

    EvalCase(
        name="missing pet name",
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
        message="I need an appointment on June 20th, will@example.com",
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
        message="June 20th",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
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
    ),
    EvalCase(
        name="cancellation request",
        message="I'd like to cancel my appointment for Max, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
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
            "appointment for my dog, Max, on June 20th, 2026.\n\n"
            "Please confirm availability at your earliest convenience.\n\n"
            "Best regards,\nWill\nwill@example.com"
        ),
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="reply style email",
        message=(
            "Re: Grooming availability\n\n"
            "Yes, June 20th works great for Max's grooming. "
            "Thanks! — will@example.com"
        ),
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="question plus booking",
        message="Do you have availability for grooming for Max on June 20th? My email is will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="long email info buried",
        message=(
            "Hi there! I've been meaning to reach out for a while. "
            "We absolutely love your shop — the staff are so friendly and Max "
            "always comes home looking amazing. Anyway, I wanted to see about "
            "booking him in for another grooming session. "
            "Would June 20th work? My email is will@example.com. "
            "Let me know if you need anything else from me. Thanks so much!"
        ),
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="all caps message",
        message="GROOMING FOR MAX ON JUNE 20TH PLEASE, WILL@EXAMPLE.COM",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="message with typo in service",
        message="I need groooming for Max on June 20th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="message with emoji",
        message="Grooming for Max on June 20th 🐾 will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="returning customer context",
        message="Hey, I was in last month with my dog Max. Can I book grooming again for June 20th? will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="multi sentence with context",
        message=(
            "Hi! My name is Sarah and I have a 2-year-old cockapoo named Biscuit. "
            "I'd like to book her in for a groom on June 20th. "
            "You can reach me at sarah@example.com."
        ),
        sender_email="sarah@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Biscuit", "service": "grooming", "requested_date": "2026-06-20"},
    ),

    # -----------------------------------------------------------------------
    # EDGE CASES
    # -----------------------------------------------------------------------

    EvalCase(
        name="pet name two words",
        message="Grooming for Bella Rose on June 20th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="pet name common noun",
        message="Grooming for Cookie on June 20th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Cookie", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="two dates mentioned",
        message="Grooming for Max on June 20th or June 27th, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
    ),
    EvalCase(
        name="two pets mentioned",
        message="Grooming for Max and Luna on June 20th, will@example.com",
        sender_email="will@example.com",
        expected_tool="send_clarification_email",
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
        message="Quick groom for Max on June 20th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="my dog Max phrasing",
        message="My dog Max needs grooming on June 20th. will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="dash separated format",
        message="Max - Grooming - June 20th - will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="compound service bath and nails",
        message="Bath and nail trim for Rex on June 20th, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Rex", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="cat instead of dog",
        message="Grooming for my cat Luna on June 20th, luna@example.com",
        sender_email="luna@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Luna", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="service listed first",
        message="Boarding on June 20th for our dog Cooper, cooper@example.com",
        sender_email="cooper@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Cooper", "service": "boarding", "requested_date": "2026-06-20"},
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
        message="Grooming for Max on Monday June 22nd, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-22"},
    ),
    EvalCase(
        name="evening time",
        message="Grooming for Max on June 20th at 6pm, will@example.com",
        sender_email="will@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Max", "service": "grooming", "requested_date": "2026-06-20"},
    ),
    EvalCase(
        name="form style structured message",
        message="Pet name: Bella\nService: grooming\nDate: June 20th\nEmail: bella@example.com",
        sender_email="bella@example.com",
        expected_tool="create_draft_booking",
        expected_args={"pet_name": "Bella", "service": "grooming", "requested_date": "2026-06-20"},
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

    return True, ""


async def main(verbose: bool):
    print(f"\nRunning {len(CASES)} eval cases  (today = {_TODAY.isoformat()} {_TODAY.strftime('%A')})\n")

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
