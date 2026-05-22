"""FastAPI app — webhook routes only. Business logic lives in agent.py."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI, HTTPException, Request, status

import app.db as db
import app.email_client as email_client
from app.agent import run_intake
from app.models import EmailWebhookPayload, FormWebhookPayload

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

@app.post("/webhook/email", status_code=status.HTTP_204_NO_CONTENT)
async def webhook_email(payload: EmailWebhookPayload):
    """
    Gmail push notifications arrive here as Pub/Sub messages.
    We decode the envelope, fetch the actual message from Gmail, and run the agent.
    """
    try:
        notification = email_client.decode_pubsub_message(payload.dict())
        history_id = notification.get("historyId")
        if not history_id:
            log.warning("email_webhook_missing_history_id")
            return

        messages = email_client.fetch_new_messages(history_id)
    except Exception as exc:
        log.error("email_webhook_decode_error", error=str(exc))
        raise HTTPException(status_code=400, detail="Failed to decode Gmail notification")

    for msg in messages:
        sender = msg.get("sender", "")
        # Extract bare email from "Name <email>" format
        if "<" in sender:
            sender_email = sender.split("<")[1].rstrip(">").strip()
        else:
            sender_email = sender.strip()

        # Persist raw message (customer_id resolved later inside agent)
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


# ---------------------------------------------------------------------------
# POST /webhook/form  — Squarespace custom-JS form submission
# ---------------------------------------------------------------------------

@app.post("/webhook/form", status_code=status.HTTP_204_NO_CONTENT)
async def webhook_form(payload: FormWebhookPayload):
    """
    Squarespace custom JS intercepts the form submit and POSTs here.
    All fields are optional — the agent fills gaps from Gingr history.
    """
    # Build a natural-language message from the form fields for the agent
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

    msg_id = await db.insert_message(
        customer_id=None,
        channel="form",
        body=message_body,
    )

    log.info("form_received", email=payload.email, msg_id=msg_id)

    result = await run_intake(
        message_body=message_body,
        sender_email=str(payload.email) if payload.email else None,
        source_channel="form",
        raw_message_id=msg_id,
    )
    log.info("intake_complete", result=result)


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
