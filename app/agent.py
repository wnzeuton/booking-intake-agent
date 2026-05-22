"""
LangChain agent — booking extraction and intake orchestration.

Runs on every inbound message (email or form). Steps:
  1. Fetch customer history from Gingr (injected into system prompt)
  2. Run the agent to extract a BookingRequest
  3. If date is missing/uncertain, send one clarifying email and stop
  4. Otherwise, write the booking to Postgres and notify owners
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Optional

import structlog
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_community.llms import LlamaCpp

import app.db as db
import app.email_client as email_client
import app.gingr as gingr
from app.models import BookingRequest

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# LLM — Llama 3 via llama.cpp HTTP server on EC2
# (In local dev, point LLAMA_ENDPOINT at Ollama or a local llama.cpp instance)
# ---------------------------------------------------------------------------

def _get_llm():
    endpoint = os.environ["LLAMA_ENDPOINT"]
    # LlamaCpp community integration talks to a llama.cpp server via HTTP
    return LlamaCpp(
        model_path="",           # unused when using server mode
        base_url=endpoint,
        temperature=0.0,
        max_tokens=1024,
        verbose=False,
    )


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
    customer = gingr.lookup_customer_by_email(phone_or_email)
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
    import asyncio
    data = json.loads(booking_json)
    request = BookingRequest(**data)

    async def _write():
        customer_id = await db.upsert_customer(
            name=request.customer_name,
            email=str(request.customer_email) if request.customer_email else None,
            phone=None,
            channel=request.source_channel,
        )
        pet_id = await db.upsert_pet(
            customer_id=customer_id,
            name=request.pet_name,
        )
        booking_id = await db.create_booking(
            request, customer_id=customer_id, pet_id=pet_id
        )
        return booking_id

    booking_id = asyncio.get_event_loop().run_until_complete(_write())
    log.info("draft_booking_created", booking_id=booking_id)
    return str(booking_id)


@tool
def notify_owners(booking_id: str) -> str:
    """
    Email the store owner to approve or reject a booking.
    Input: booking_id as a string.
    Returns 'sent' on success.
    """
    import asyncio

    async def _fetch():
        return await db.get_booking(int(booking_id))

    booking = asyncio.get_event_loop().run_until_complete(_fetch())
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
        requested_time=str(booking["requested_time"]) if booking.get("requested_time") else None,
    )
    return "sent"


@tool
def send_clarification_email(args_json: str) -> str:
    """
    Send one clarifying email to the customer when a required field is missing.
    Input: JSON with keys 'to' (email), 'customer_name', 'missing_field' (human-readable).
    Only call this once per intake — do not loop.
    Returns 'sent' on success.
    """
    args = json.loads(args_json)
    email_client.send_clarification_email(
        to=args["to"],
        customer_name=args["customer_name"],
        missing_field=args["missing_field"],
    )
    return "sent"


TOOLS = [
    lookup_customer,
    check_availability,
    create_draft_booking,
    notify_owners,
    send_clarification_email,
]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """\
You are a booking intake agent for a pet store. Your job is to extract a structured booking request from an inbound message and write it to the database for owner approval.

Customer history (from Gingr):
{customer_history}

Today's date: {today}

Instructions:
1. Call lookup_customer with the customer's email to get their history and pets.
2. Extract: customer_name, pet_name, service, requested_date, and optionally requested_time.
   - Use customer history to fill gaps (e.g. if they only have one pet, use that pet).
   - requested_date MUST be extracted with high confidence. Relative dates like "Saturday" must be resolved to an absolute date using today's date.
3. If you cannot extract requested_date with confidence, call send_clarification_email once and STOP.
4. Otherwise, call create_draft_booking with the extracted data (source_channel = "{source_channel}").
5. Then call notify_owners with the returned booking_id.
6. Done.

Do not invent information. Do not call create_draft_booking with a null or uncertain date.

{tools}

Use this format:
Thought: ...
Action: tool_name
Action Input: ...
Observation: ...
... (repeat as needed)
Thought: I now know the final answer
Final Answer: ...

Begin!

Message: {input}
{agent_scratchpad}"""

PROMPT = PromptTemplate(
    input_variables=["customer_history", "today", "source_channel", "input", "agent_scratchpad", "tools"],
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
    # Pre-fetch Gingr history to inject into the prompt
    customer_history = "No prior history found."
    if sender_email:
        gingr_customer = gingr.lookup_customer_by_email(sender_email)
        if gingr_customer:
            pets_str = ", ".join(
                f"{p.name} ({p.preferred_service or 'no preference'})"
                for p in gingr_customer.pets
            )
            customer_history = (
                f"Name: {gingr_customer.name}\n"
                f"Pets: {pets_str}"
            )

    memory = ConversationBufferMemory(memory_key="agent_scratchpad", return_messages=False)
    if memory_state:
        # Restore serialized memory for multi-turn clarification threads
        memory.chat_memory.messages = memory_state.get("messages", [])

    llm = _get_llm()
    agent = create_react_agent(llm=llm, tools=TOOLS, prompt=PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )

    try:
        result = executor.invoke({
            "input": message_body,
            "customer_history": customer_history,
            "today": date.today().isoformat(),
            "source_channel": source_channel,
        })
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
