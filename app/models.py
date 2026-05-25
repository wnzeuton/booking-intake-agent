"""Pydantic schemas — all data shapes live here, never inline in route handlers."""

from __future__ import annotations

from datetime import date, time
from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ---------------------------------------------------------------------------
# Inbound webhook payloads
# ---------------------------------------------------------------------------

class EmailWebhookPayload(BaseModel):
    """Gmail push notification payload (Pub/Sub message wrapper)."""
    message: dict  # raw Pub/Sub envelope; decoded in email_client.py
    subscription: str


class FormWebhookPayload(BaseModel):
    """
    Squarespace custom-JS form submission.
    Field names match whatever the Squarespace form sends; optional everywhere
    because form fields vary and the agent fills gaps from Gingr history.
    """
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    pet_name: Optional[str] = None
    service: Optional[str] = None
    requested_date: Optional[str] = None  # free-text; agent parses to date
    requested_time: Optional[str] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class BookingRequest(BaseModel):
    """
    Structured booking extracted by the agent.
    This is what the agent *must* produce before creating a draft booking.
    """
    customer_name: str
    customer_email: Optional[EmailStr] = None
    pet_name: str
    service: str
    requested_date: date = Field(..., description="Must be extracted with high confidence")
    requested_time: Optional[time] = None
    source_channel: Literal["email", "form"]
    raw_message_id: Optional[int] = None
    notes: Optional[str] = None

    @field_validator("customer_email", mode="before")
    @classmethod
    def empty_email_to_none(cls, v):
        """Coerce empty string → None so EmailStr validation doesn't reject it."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("customer_name", "pet_name", mode="before")
    @classmethod
    def required_str_not_empty(cls, v):
        """Reject empty or whitespace-only strings for required name fields."""
        if isinstance(v, str) and not v.strip():
            raise ValueError("field is required and cannot be empty")
        return v


class BookingRecord(BookingRequest):
    """BookingRequest as stored in Postgres (adds PK + status)."""
    id: int
    customer_id: int
    pet_id: int
    status: Literal["pending", "confirmed", "rejected"] = "pending"

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Owner reply (email reply from owner to approve/reject)
# ---------------------------------------------------------------------------

class OwnerReply(BaseModel):
    """Parsed intent from an owner's email reply."""
    booking_id: int
    decision: Literal["confirmed", "rejected"]
    reply_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Gingr API response shapes (read-only)
# ---------------------------------------------------------------------------

class GingrPet(BaseModel):
    id: int
    name: str
    breed: Optional[str] = None
    preferred_service: Optional[str] = None
    notes: Optional[str] = None


class GingrCustomer(BaseModel):
    id: int
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    pets: list[GingrPet] = []
