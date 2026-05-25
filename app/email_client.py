"""Gmail API helpers — inbound push notification decoding + outbound send."""

from __future__ import annotations

import base64
import json
import os
from email.mime.text import MIMEText
from typing import Optional

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import structlog

log = structlog.get_logger()

# Gmail push notifications are delivered as Pub/Sub messages; the agent needs
# read scope to fetch the full message body and send scope to reply.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]


def _get_service():
    """Build an authenticated Gmail API service from env-provided credentials."""
    raw = os.environ["GMAIL_CREDENTIALS"]
    creds_dict = json.loads(raw)
    creds = Credentials.from_authorized_user_info(creds_dict, SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# Inbound
# ---------------------------------------------------------------------------

def decode_pubsub_message(payload: dict) -> dict:
    """
    Decode a Gmail Pub/Sub push notification envelope into a dict with:
      - email_address: the Gmail address that received the message
      - history_id: the new historyId to fetch from Gmail API
    """
    data = payload["message"]["data"]
    decoded = base64.urlsafe_b64decode(data + "==").decode("utf-8")
    return json.loads(decoded)


def fetch_new_messages(history_id: str, user_id: str = "me") -> list[dict]:
    """
    List messages newer than history_id and return each as a dict with:
      id, subject, sender, body (plain text), thread_id
    """
    service = _get_service()
    history = (
        service.users()
        .history()
        .list(userId=user_id, startHistoryId=history_id, historyTypes=["messageAdded"])
        .execute()
    )

    messages = []
    for record in history.get("history", []):
        for added in record.get("messagesAdded", []):
            msg_id = added["message"]["id"]
            try:
                msg = (
                    service.users()
                    .messages()
                    .get(userId=user_id, id=msg_id, format="full")
                    .execute()
                )
                messages.append(_parse_message(msg))
            except HttpError as exc:
                if exc.resp.status == 404:
                    log.warning("gmail_message_not_found", msg_id=msg_id)
                else:
                    raise

    return messages


def _parse_message(msg: dict) -> dict:
    headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
    body = _extract_body(msg["payload"])
    return {
        "id": msg["id"],
        "thread_id": msg["threadId"],
        "subject": headers.get("Subject", ""),
        "sender": headers.get("From", ""),
        "body": body,
    }


def _extract_body(payload: dict) -> str:
    """Recursively extract plain-text body from a Gmail message payload."""
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        return base64.urlsafe_b64decode(data + "==").decode("utf-8") if data else ""

    for part in payload.get("parts", []):
        result = _extract_body(part)
        if result:
            return result

    return ""


# ---------------------------------------------------------------------------
# Outbound
# ---------------------------------------------------------------------------

def send_email(*, to: str, subject: str, body: str, user_id: str = "me") -> str:
    """Send a plain-text email via Gmail API. Returns the sent message id."""
    service = _get_service()
    mime = MIMEText(body)
    mime["to"] = to
    mime["subject"] = subject
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    sent = service.users().messages().send(userId=user_id, body={"raw": raw}).execute()
    log.info("email_sent", to=to, subject=subject, msg_id=sent["id"])
    return sent["id"]


def send_booking_confirmation_request(
    *,
    owner_email: str,
    booking_id: int,
    customer_name: str,
    pet_name: str,
    service: str,
    requested_date: str,
    requested_time: Optional[str],
) -> str:
    """
    Email the store owner asking them to approve or reject a booking.
    Owner replies 'Y' or 'N' (case-insensitive) — the /webhook/reply route
    parses their reply and calls update_booking_status.
    """
    time_str = f" at {requested_time}" if requested_time else ""
    subject = f"[Booking #{booking_id}] Approve? {pet_name} — {service} on {requested_date}"
    body = (
        f"New booking request:\n\n"
        f"  Customer: {customer_name}\n"
        f"  Pet:      {pet_name}\n"
        f"  Service:  {service}\n"
        f"  Date:     {requested_date}{time_str}\n\n"
        f"Reply Y to confirm or N to reject.\n"
        f"(Booking ID: {booking_id})"
    )
    return send_email(to=owner_email, subject=subject, body=body)


def send_clarification_email(
    *, to: str, customer_name: str, missing_field: str
) -> str:
    """
    Send one clarifying email to the customer when required info is missing.
    The agent calls this when it cannot extract requested_date with confidence.
    """
    subject = "Quick question about your booking request"
    body = (
        f"Hi {customer_name},\n\n"
        f"Thanks for reaching out! We just need one more detail to complete your booking:\n\n"
        f"  → {missing_field}\n\n"
        f"Reply to this email with that info and we'll take it from there.\n\n"
        f"— The Team"
    )
    return send_email(to=to, subject=subject, body=body)
