"""FastAPI app — webhook routes only. Business logic lives in agent.py."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status

# Configure structlog to write to stdout in a readable format
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    logger_factory=structlog.PrintLoggerFactory(),
)
logging.basicConfig(level=logging.DEBUG)

import app.db as db
import app.email_client as email_client
from app.agent import run_intake
from app.models import BookingRequest, EmailWebhookPayload, FormWebhookPayload

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the DB connection pool on startup
    await db.get_pool()
    yield
    await db.close_pool()


app = FastAPI(title="booking-intake-agent", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /webhook/email  — Gmail push notification (Pub/Sub)
# ---------------------------------------------------------------------------

async def _process_email_message(msg: dict):
    """Background task: run the agent on a single inbound email message."""
    sender = msg.get("sender", "")
    if "<" in sender:
        sender_email = sender.split("<")[1].rstrip(">").strip()
    else:
        sender_email = sender.strip()

    msg_id = await db.insert_message(
        customer_id=None,
        channel="email",
        body=msg["body"],
    )
    log.info("email_received", sender=sender_email, msg_id=msg_id)

    result = await run_intake(
        message_body=msg["body"],
        sender_email=sender_email,
        source_channel="email",
        raw_message_id=msg_id,
    )
    log.info("intake_complete", result=result)


@app.post("/webhook/email", status_code=status.HTTP_204_NO_CONTENT)
async def webhook_email(payload: EmailWebhookPayload, background_tasks: BackgroundTasks):
    """
    Gmail push notifications arrive here as Pub/Sub messages.
    We decode the envelope, fetch the actual message from Gmail, and run the agent.
    Returns 204 immediately — agent runs in a background task so Pub/Sub doesn't retry.
    """
    try:
        notification = email_client.decode_pubsub_message(payload.dict())
        new_history_id = str(notification.get("historyId", ""))
        if not new_history_id:
            log.warning("email_webhook_missing_history_id")
            return

        stored_history_id = await db.get_state("gmail_history_id")
        start_history_id = stored_history_id or str(int(new_history_id) - 1)

        log.info("email_webhook_received",
                 new_history_id=new_history_id,
                 start_history_id=start_history_id)

        # Advance pointer before fetching so stale/deleted messages never cause retries
        await db.set_state("gmail_history_id", new_history_id)

        messages = email_client.fetch_new_messages(start_history_id)
    except Exception as exc:
        log.error("email_webhook_decode_error", error=str(exc))
        raise HTTPException(status_code=400, detail="Failed to decode Gmail notification")

    for msg in messages:
        background_tasks.add_task(_process_email_message, msg)


# ---------------------------------------------------------------------------
# POST /webhook/form  — Squarespace custom-JS form submission
# ---------------------------------------------------------------------------

@app.post("/webhook/form", status_code=status.HTTP_204_NO_CONTENT)
async def webhook_form(payload: FormWebhookPayload, background_tasks: BackgroundTasks):
    """
    Squarespace custom JS intercepts the form submit and POSTs here.

    If the form contains all required fields (name, email, pet, service, date),
    we bypass the LLM and write directly to the DB — faster and more reliable.
    If any required field is missing we fall back to the agent to fill the gaps.

    Returns 204 immediately — any DB/email work runs in a background task.
    """
    sender_email = str(payload.email) if payload.email else None
    required_fields_present = all([
        payload.name,
        payload.email,
        payload.pet_name,
        payload.service,
        payload.requested_date,
    ])

    msg_id = await db.insert_message(
        customer_id=None,
        channel="form",
        body=str(payload.dict()),
    )
    log.info("form_received", email=sender_email, msg_id=msg_id,
             complete=required_fields_present)

    if required_fields_present:
        # Fast path: all data is structured — skip the LLM entirely.
        async def _direct_book():
            try:
                request = BookingRequest(
                    customer_name=payload.name,
                    customer_email=sender_email,
                    pet_name=payload.pet_name,
                    service=payload.service,
                    requested_date=payload.requested_date,
                    requested_time=payload.requested_time,
                    notes=payload.notes,
                    source_channel="form",
                    raw_message_id=msg_id,
                )
                customer_id = await db.upsert_customer(
                    name=request.customer_name,
                    email=sender_email,
                    phone=None,
                    channel="form",
                )
                pet_id = await db.upsert_pet(
                    customer_id=customer_id,
                    name=request.pet_name,
                )
                booking_id = await db.create_booking(
                    request, customer_id=customer_id, pet_id=pet_id
                )
                log.info("form_booking_created", booking_id=booking_id)

                owner_email = os.environ["OWNER_EMAIL"]
                email_client.send_booking_confirmation_request(
                    owner_email=owner_email,
                    booking_id=booking_id,
                    customer_name=request.customer_name,
                    pet_name=request.pet_name,
                    service=request.service,
                    requested_date=str(request.requested_date),
                    requested_time=str(request.requested_time) if request.requested_time else None,
                )
            except Exception as exc:
                log.error("form_direct_book_error", error=str(exc))

        background_tasks.add_task(_direct_book)
    else:
        # Slow path: missing fields — use the agent to fill gaps from message context.
        parts = []
        if payload.name:
            parts.append(f"Name: {payload.name}")
        if payload.email:
            parts.append(f"Email: {payload.email}")
        if payload.pet_name:
            parts.append(f"Pet: {payload.pet_name}")
        if payload.service:
            parts.append(f"Service: {payload.service}")
        if payload.requested_date:
            parts.append(f"Date: {payload.requested_date}")
        if payload.requested_time:
            parts.append(f"Time: {payload.requested_time}")
        if payload.notes:
            parts.append(f"Notes: {payload.notes}")
        message_body = "\n".join(parts) if parts else "(empty form submission)"

        async def _run():
            result = await run_intake(
                message_body=message_body,
                sender_email=sender_email,
                source_channel="form",
                raw_message_id=msg_id,
            )
            log.info("intake_complete", result=result)

        background_tasks.add_task(_run)


# ---------------------------------------------------------------------------
# POST /webhook/reply  — owner email reply (Y / N)
# ---------------------------------------------------------------------------

@app.post("/webhook/reply", status_code=status.HTTP_204_NO_CONTENT)
async def webhook_reply(request: Request):
    """
    The store owner replies to the booking notification email.
    We parse the reply for Y/N and update the booking status accordingly.

    For MVP this is polled via Gmail (the same push notification path as
    inbound customer email). The subject line contains the booking ID:
      [Booking #42] Approve? Mochi — grooming on 2025-06-14

    This route is a stub for a future direct reply webhook integration.
    The Gmail push path handles owner replies by subject-line parsing.
    """
    body = await request.json()
    log.info("owner_reply_received", body=body)

    booking_id: Optional[int] = body.get("booking_id")
    reply_text: str = body.get("reply", "").strip().upper()

    if not booking_id:
        raise HTTPException(status_code=400, detail="booking_id required")

    if reply_text.startswith("Y"):
        await db.update_booking_status(booking_id, "confirmed")
        log.info("booking_confirmed", booking_id=booking_id)
    elif reply_text.startswith("N"):
        await db.update_booking_status(booking_id, "rejected")
        log.info("booking_rejected", booking_id=booking_id)
    else:
        log.warning("owner_reply_unrecognized", reply=reply_text)
        raise HTTPException(status_code=422, detail=f"Unrecognized reply: {reply_text!r}. Send Y or N.")
