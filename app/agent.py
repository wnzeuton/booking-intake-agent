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
import json
import os
import threading
from datetime import date
from typing import Optional

import structlog
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_ollama import OllamaLLM

import app.db as db
import app.email_client as email_client
import app.gingr as gingr
from app.models import BookingRequest

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Async helper — run a coroutine from sync tool code, even inside FastAPI's
# running event loop. Spins a fresh event loop on a dedicated thread so it
# never conflicts with the outer loop.
# ---------------------------------------------------------------------------

def _run_async(coro) -> any:
    """
    Run an async coroutine from synchronous code.
    Safe to call from within a running event loop (e.g. FastAPI / uvicorn).
    """
    result = None
    exc = None

    def _target():
        nonlocal result, exc
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(coro)
        except Exception as e:
            exc = e
        finally:
            loop.close()

    t = threading.Thread(target=_target)
    t.start()
    t.join()

    if exc is not None:
        raise exc
    return result


# ---------------------------------------------------------------------------
# LLM — Llama 3 via llama.cpp HTTP server on EC2
# (In local dev, point LLAMA_ENDPOINT at Ollama or a local llama.cpp instance)
# ---------------------------------------------------------------------------

def _get_llm():
    endpoint = os.environ["LLAMA_ENDPOINT"]  # e.g. http://localhost:11434
    # OllamaLLM talks to the Ollama HTTP server (local dev) or a llama.cpp
    # server that exposes an Ollama-compatible API (production on EC2).
    return OllamaLLM(
        model="llama3",
        base_url=endpoint,
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# JSON parsing helper — handles LLM output quirks
# ---------------------------------------------------------------------------

def _parse_tool_json(raw: str) -> dict:
    """
    Parse JSON from a tool input string, handling common LLM output quirks:
    - Missing outer braces
    - Outer single-quotes acting as delimiters
    - Truncated closing quote on last value

    Strategy: try json.loads first, then fall back to regex key-value extraction.
    """
    import re

    s = raw.strip()

    def _clean(d: dict) -> dict:
        """Strip stray single-quotes the model wraps around values."""
        return {k: v.strip("'").strip() if isinstance(v, str) else v for k, v in d.items()}

    # 1. Try direct parse (happy path)
    for candidate in [s, "{" + s + "}", s.strip("'").strip('"')]:
        try:
            return _clean(json.loads(candidate))
        except json.JSONDecodeError:
            pass

    # 2. Regex fallback — extract all "key": "value" or "key": value pairs
    #    Works even if the outer braces/quotes are malformed
    result = {}
    # Match "key": "string value" — strip any inner single quotes from values
    for m in re.finditer(r'"?(\w+)"?\s*:\s*"([^"]*)"?', s):
        result[m.group(1)] = m.group(2).strip("'").strip()
    # Also catch "key": non-string (numbers, booleans, unquoted values)
    for m in re.finditer(r'"?(\w+)"?\s*:\s*([^",}\s][^,}]*)', s):
        if m.group(1) not in result:
            result[m.group(1)] = m.group(2).strip().strip("'").strip('"')

    if result:
        log.info("tool_json_regex_fallback", extracted=result)
        return result

    raise ValueError(f"Could not parse tool input as JSON: {raw!r}")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def lookup_customer(phone_or_email: str) -> str:
    """
    Look up a customer by email in Gingr and the local DB.
    Returns a JSON string with customer name, pets, and their preferred services.
    Use this first before attempting to extract booking details.
    """
    try:
        customer = gingr.lookup_customer_by_email(phone_or_email)
    except Exception as exc:
        log.warning("gingr_lookup_failed", error=str(exc))
        return json.dumps({"found": False, "error": "Gingr unavailable"})
    if not customer:
        return json.dumps({"found": False})
    return json.dumps({
        "found": True,
        "name": customer.name,
        "pets": [
            {
                "name": p.name,
                "breed": p.breed,
                "preferred_service": p.preferred_service,
                "notes": p.notes,
            }
            for p in customer.pets
        ],
    })


@tool
def check_availability(date_str: str, service: str) -> str:
    """
    Check Gingr for existing reservations on a given date and service.
    date_str: ISO 8601 date, e.g. '2025-06-14'
    service: e.g. 'grooming', 'boarding', 'daycare'
    Returns a JSON list of reservation summaries.
    """
    reservations = gingr.check_availability(date_str, service)
    return json.dumps(reservations)


@tool
def create_draft_booking(booking_json: str) -> str:
    """
    Persist a pending booking to Postgres.
    Input must be a JSON object with keys:
      customer_name, customer_email, pet_name, service,
      requested_date (YYYY-MM-DD), source_channel,
      and optionally: requested_time (HH:MM), notes, raw_message_id
    Returns the new booking_id as a string.
    IMPORTANT: Only call this when requested_date is known with high confidence.
    """
    data = _parse_tool_json(booking_json)
    log.info("create_draft_booking_parsed", data=data)
    try:
        request = BookingRequest(**data)
    except Exception as exc:
        missing = [str(e["loc"][0]) for e in exc.errors()] if hasattr(exc, "errors") else [str(exc)]
        to = data.get("customer_email") or data.get("customer_name") or "the customer"
        name = data.get("customer_name") or "there"
        missing_desc = " and ".join(missing).replace("_", " ")
        return (
            f"ERROR: Cannot create booking — {missing_desc} is required but missing. "
            f"You MUST call send_clarification_email now with this exact input: "
            f'{{\"to\": \"{to}\", \"customer_name\": \"{name}\", \"missing_field\": \"{missing_desc}\"}}'
        )

    async def _write():
        # Use a fresh connection per call — asyncpg pools are tied to their
        # event loop and _run_async creates a new loop each time.
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
            booking_id = await db.create_booking_conn(conn, request,
                                                       customer_id=customer_id,
                                                       pet_id=pet_id)
            return booking_id
        finally:
            await conn.close()

    booking_id = _run_async(_write())
    log.info("draft_booking_created", booking_id=booking_id)
    return f"Booking created successfully. booking_id={booking_id}. Now call notify_owners with this booking_id."


@tool
def notify_owners(booking_id: str) -> str:
    """
    Email the store owner to approve or reject a booking.
    Input: booking_id as a string.
    Returns 'sent' on success.
    """
    # Parse booking_id — model may pass "booking_id=3" or just "3"
    import re as _re
    match = _re.search(r'\d+', str(booking_id))
    if not match:
        return f"Error: could not parse booking_id from {booking_id!r}"
    bid = int(match.group())

    async def _fetch():
        import asyncpg as _asyncpg
        conn = await _asyncpg.connect(dsn=os.environ["DATABASE_URL"])
        try:
            return await db.get_booking_conn(conn, bid)
        finally:
            await conn.close()

    booking = _run_async(_fetch())
    if not booking:
        return f"Error: booking {bid} not found"

    owner_email = os.environ["OWNER_EMAIL"]
    email_client.send_booking_confirmation_request(
        owner_email=owner_email,
        booking_id=booking["id"],
        customer_name=booking.get("customer_name", "Customer"),
        pet_name=booking.get("pet_name", "Pet"),
        service=booking["service"],
        requested_date=str(booking["requested_date"]),
        requested_time=str(booking["requested_time"]) if booking.get("requested_time") else None,
    )
    return f"DONE. Owner notified about booking #{bid}. No further actions needed. Output your Final Answer now."


@tool
def send_clarification_email(args_json: str) -> str:
    """
    Send one clarifying email to the customer when a required field is missing.
    Input: JSON with keys 'to' (email), 'customer_name', 'missing_field' (human-readable).
    Only call this once per intake — do not loop.
    Returns 'sent' on success.
    """
    args = _parse_tool_json(args_json)
    email_client.send_clarification_email(
        to=args["to"],
        customer_name=args["customer_name"],
        missing_field=args["missing_field"],
    )
    return "DONE. Clarification email sent. Do not call any more tools. Output your Final Answer now."


TOOLS = [
    create_draft_booking,
    notify_owners,
    send_clarification_email,
]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """\
You are a booking intake agent for a pet store. Your job is to extract a structured booking request from an inbound message and write it to the database for owner approval.

{input}

Available tools: {tool_names}

{tools}

Use this exact format — do not deviate:
Thought: <your reasoning>
Action: <tool name, exactly as listed — never "None" or any other value>
Action Input: <tool input>
Observation: <tool result>
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: <summary of what you did>

IMPORTANT: Every Action must be one of the listed tool names. Never write "Action: None" or "Action: thinking" — put your reasoning in the Thought line only.

Begin!

{agent_scratchpad}"""

PROMPT = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template=SYSTEM_TEMPLATE,
)

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
    # Pre-fetch Gingr history to inject into the prompt.
    # Run in a thread so tenacity retries don't block the event loop.
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
                f"{p.name} ({p.preferred_service or 'no preference'})"
                for p in gingr_customer.pets
            )
            customer_history = (
                f"Name: {gingr_customer.name}\n"
                f"Pets: {pets_str}"
            )

    # Build a single rich input string — LangChain 0.3+ AgentExecutor
    # only accepts one user-defined input key.
    enriched_input = (
        f"Today's date: {date.today().isoformat()}\n"
        f"Customer email: {sender_email or 'unknown'}\n"
        f"Source channel: {source_channel}\n"
        f"Gingr history: {customer_history}\n\n"
        f"Inbound message:\n{message_body}\n\n"
        f"Task: Create a booking from this message. Use ONLY these tools in order:\n"
        f"1. call create_draft_booking — input must be a valid JSON string (double-quoted "
        f"keys) with these fields:\n"
        f"   customer_name, customer_email, pet_name, service, requested_date (YYYY-MM-DD), "
        f"source_channel=\"{source_channel}\"\n"
        f"   Optional: requested_time (HH:MM), notes\n"
        f"   Extract all values from the message above. Resolve relative dates using today.\n"
        f"   If customer_name, pet_name, or requested_date are missing or unknown, do NOT "
        f"pass empty strings — call send_clarification_email instead "
        f"(JSON: {{\"to\": \"<email or '{sender_email or ''}'\", \"customer_name\": \"<name or 'there'>\", "
        f"\"missing_field\": \"<describe what's missing>\"}}) and stop.\n"
        f"2. call notify_owners — input is the booking_id string returned by create_draft_booking.\n"
        f"Do not call any other tools. Do not try to extract data as a tool call — "
        f"read the message and construct the JSON yourself."
    )

    llm = _get_llm()
    agent = create_react_agent(llm=llm, tools=TOOLS, prompt=PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,  # create_draft_booking + notify_owners + Final Answer + slack
    )

    try:
        # executor.invoke() is synchronous (LangChain / Ollama HTTP calls).
        # Run in a thread so it never blocks the FastAPI event loop.
        result = await asyncio.to_thread(
            executor.invoke,
            {"input": enriched_input},
        )
        output = result.get("output", "")
    except Exception as exc:
        log.error("agent_error", error=str(exc))
        return {"status": "error", "booking_id": None, "output": str(exc)}

    # Determine outcome from agent output heuristics
    if "clarification" in output.lower() or "sent" in output.lower() and "clarif" in output.lower():
        status = "clarification_sent"
        booking_id = None
    elif "booking_id" in output.lower() or output.strip().isdigit():
        status = "booked"
        booking_id = int(output.strip()) if output.strip().isdigit() else None
    else:
        status = "booked"
        booking_id = None

    return {"status": status, "booking_id": booking_id, "output": output}
