"""
LangChain agent — booking extraction and intake orchestration.

Runs on every inbound message (email or form). Steps:
  1. Fetch customer history from Gingr (injected into system prompt)
  2. Run the agent to extract a BookingRequest
  3. If date is missing/uncertain, send one clarifying email and stop
  4. Otherwise, write the booking to Postgres and notify owners
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import date
from typing import Optional

import structlog
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import app.db as db
import app.email_client as email_client
import app.gingr as gingr
from app.models import BookingRequest

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# LLM — Claude via Anthropic API
# Defaults to Haiku (fast + cheap) for dev; set ANTHROPIC_MODEL=claude-sonnet-4-6
# in production for better reasoning on edge cases.
# ---------------------------------------------------------------------------

def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=0.0,
        max_tokens=1024,
    )


# ---------------------------------------------------------------------------
# Tools — typed parameters (Claude native tool calling; no JSON parsing needed)
#
# These run inside asyncio.to_thread(), so asyncio.run() is safe here —
# there is no event loop on the worker thread.
# ---------------------------------------------------------------------------

@tool
def create_draft_booking(
    customer_name: str,
    pet_name: str,
    service: str,
    requested_date: str,
    source_channel: str,
    customer_email: Optional[str] = None,
    requested_time: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Persist a pending booking to Postgres.
    - requested_date: ISO 8601 date string (YYYY-MM-DD). Resolve relative dates
      like "tomorrow" or "next Friday" using today's date from the system prompt.
    - source_channel: must be 'email' or 'form'.
    - Only call this when requested_date is known with high confidence.
    Returns: the new booking_id as a string.
    """
    log.info("create_draft_booking",
             customer_name=customer_name, pet_name=pet_name,
             service=service, requested_date=requested_date)
    try:
        request = BookingRequest(
            customer_name=customer_name,
            customer_email=customer_email or None,
            pet_name=pet_name,
            service=service,
            requested_date=requested_date,
            requested_time=requested_time or None,
            source_channel=source_channel,
            notes=notes,
        )
    except Exception as exc:
        missing = (
            [str(e["loc"][0]) for e in exc.errors()]
            if hasattr(exc, "errors")
            else [str(exc)]
        )
        missing_desc = " and ".join(missing).replace("_", " ")
        return (
            f"ERROR: Cannot create booking — {missing_desc} is required but missing. "
            f"Call send_clarification_email instead."
        )

    async def _write():
        import asyncpg as _asyncpg
        conn = await _asyncpg.connect(dsn=os.environ["DATABASE_URL"])
        try:
            customer_id = await db.upsert_customer_conn(
                conn,
                name=request.customer_name,
                email=str(request.customer_email) if request.customer_email else None,
                phone=None,
                channel=request.source_channel,
            )
            pet_id = await db.upsert_pet_conn(
                conn,
                customer_id=customer_id,
                name=request.pet_name,
            )
            booking_id = await db.create_booking_conn(
                conn, request, customer_id=customer_id, pet_id=pet_id
            )
            return booking_id
        finally:
            await conn.close()

    booking_id = asyncio.run(_write())
    log.info("draft_booking_created", booking_id=booking_id)
    return f"Booking created successfully. booking_id={booking_id}"


@tool
def notify_owners(booking_id: int) -> str:
    """
    Email the store owner to approve or reject a booking.
    Call this immediately after create_draft_booking with the returned booking_id.
    Returns 'sent' on success.
    """
    async def _fetch():
        import asyncpg as _asyncpg
        conn = await _asyncpg.connect(dsn=os.environ["DATABASE_URL"])
        try:
            return await db.get_booking_conn(conn, booking_id)
        finally:
            await conn.close()

    booking = asyncio.run(_fetch())
    if not booking:
        return f"Error: booking {booking_id} not found"

    owner_email = os.environ["OWNER_EMAIL"]
    email_client.send_booking_confirmation_request(
        owner_email=owner_email,
        booking_id=booking["id"],
        customer_name=booking.get("customer_name", "Customer"),
        pet_name=booking.get("pet_name", "Pet"),
        service=booking["service"],
        requested_date=str(booking["requested_date"]),
        requested_time=(
            str(booking["requested_time"]) if booking.get("requested_time") else None
        ),
    )
    log.info("owner_notified", booking_id=booking_id)
    return f"Owner notified about booking #{booking_id}."


@tool
def send_clarification_email(
    to: str,
    customer_name: str,
    missing_field: str,
) -> str:
    """
    Send one clarifying email to the customer when a required field is missing.
    Only call this once per intake — do not send multiple clarification emails.
    - to: customer email address (use the sender_email from the context if available)
    - customer_name: the customer's name (or "there" if unknown)
    - missing_field: human-readable description of what's missing
      (e.g. "requested date", "pet name", "your name")
    Returns 'sent' on success.
    """
    email_client.send_clarification_email(
        to=to,
        customer_name=customer_name,
        missing_field=missing_field,
    )
    log.info("clarification_email_sent", to=to, missing_field=missing_field)
    return "Clarification email sent."


TOOLS = [
    create_draft_booking,
    notify_owners,
    send_clarification_email,
]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a booking intake agent for a pet store. Your job is to extract a structured \
booking request from the inbound message and record it for owner approval.

Decision rules (follow in order):
1. If customer_name, pet_name, or requested_date are missing or too vague to resolve \
   confidently (e.g. "next week", "sometime soon", no specific day), call \
   send_clarification_email — then stop. Do NOT create a booking with a guessed date.
2. If all required fields are present and the date is unambiguous, call \
   create_draft_booking, then immediately call notify_owners with the returned booking_id.
3. Never call any tool more than once.

Field notes:
- requested_date must be YYYY-MM-DD. Use today's date (given below) to resolve \
  relative references like "tomorrow" or "next Friday".
- source_channel comes from the context below — use it exactly as given.
- If the customer's email is unknown, pass an empty string for customer_email.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

async def run_intake(
    *,
    message_body: str,
    sender_email: Optional[str],
    source_channel: str,
    raw_message_id: Optional[int] = None,
    memory_state: Optional[dict] = None,
) -> dict:
    """
    Run the booking intake agent on a single inbound message.

    Returns:
      {
        "status": "booked" | "clarification_sent" | "error",
        "booking_id": int | None,
        "output": str,  # agent's final answer
      }
    """
    # Pre-fetch Gingr history to inject into the prompt context.
    customer_history = "No prior history found."
    if sender_email:
        try:
            gingr_customer = await asyncio.to_thread(
                gingr.lookup_customer_by_email, sender_email
            )
        except Exception as exc:
            log.warning("gingr_lookup_failed", email=sender_email, error=str(exc))
            gingr_customer = None
        if gingr_customer:
            pets_str = ", ".join(
                f"{p.name} ({p.preferred_service or 'no preferred service'})"
                for p in gingr_customer.pets
            )
            customer_history = (
                f"Name: {gingr_customer.name}\n"
                f"Pets on file: {pets_str}"
            )

    enriched_input = (
        f"Today's date: {date.today().isoformat()}\n"
        f"Customer email: {sender_email or 'unknown'}\n"
        f"Source channel: {source_channel}\n"
        f"Gingr history: {customer_history}\n\n"
        f"Inbound message:\n{message_body}"
    )

    llm = _get_llm()
    agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=5,
    )

    try:
        # executor.invoke() makes synchronous HTTP calls to the Anthropic API.
        # Run in a thread so it never blocks the FastAPI event loop.
        result = await asyncio.to_thread(
            executor.invoke,
            {"input": enriched_input},
        )
        output = result.get("output", "")
        # Claude via langchain-anthropic returns content as a list of blocks
        # e.g. [{"type": "text", "text": "..."}] — flatten to a plain string.
        if isinstance(output, list):
            output = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in output
            ).strip()
    except Exception as exc:
        log.error("agent_error", error=str(exc))
        return {"status": "error", "booking_id": None, "output": str(exc)}

    # Determine outcome from agent output
    output_lower = output.lower()
    if "clarification" in output_lower or "clarifying" in output_lower:
        status = "clarification_sent"
        booking_id = None
    else:
        status = "booked"
        # Match patterns like: booking_id=10, Booking ID: 10, booking #10
        m = re.search(r'booking[_\s]?id[*_\s]*[=:#*]+[*_\s]*(\d+)', output_lower) \
            or re.search(r'booking\s+#(\d+)', output_lower) \
            or re.search(r'booking[_\s]id\D{0,5}(\d+)', output_lower)
        booking_id = int(m.group(1)) if m else None

    return {"status": status, "booking_id": booking_id, "output": output}
