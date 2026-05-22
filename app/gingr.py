"""Gingr API client — read-only. Never attempt writes to Gingr."""

from __future__ import annotations

import os
from typing import Optional

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.models import GingrCustomer, GingrPet

log = structlog.get_logger()

# Gingr's REST API base URL — set GINGR_API_BASE in env if it differs.
GINGR_BASE = os.environ.get("GINGR_API_BASE", "https://app.gingrapp.com/api/v2")


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['GINGR_API_KEY']}",
        "Accept": "application/json",
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def lookup_customer_by_email(email: str) -> Optional[GingrCustomer]:
    """
    Search Gingr for a customer by email.
    Returns a GingrCustomer (with pets) or None if not found.
    """
    with httpx.Client(timeout=10) as client:
        resp = client.get(
            f"{GINGR_BASE}/customers",
            headers=_headers(),
            params={"email": email},
        )
        resp.raise_for_status()
        data = resp.json()

    results = data.get("data", [])
    if not results:
        log.info("gingr_customer_not_found", email=email)
        return None

    raw = results[0]
    pets = [
        GingrPet(
            id=p["id"],
            name=p["name"],
            breed=p.get("breed"),
            preferred_service=p.get("preferred_service"),
            notes=p.get("notes"),
        )
        for p in raw.get("pets", [])
    ]
    customer = GingrCustomer(
        id=raw["id"],
        name=raw["name"],
        email=raw.get("email"),
        phone=raw.get("phone"),
        pets=pets,
    )
    log.info("gingr_customer_found", customer_id=customer.id, num_pets=len(pets))
    return customer


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def check_availability(date: str, service: str) -> list[dict]:
    """
    Return existing Gingr reservations on the given date for the given service.
    Used by the agent to warn about conflicts (does not block booking creation).

    date: ISO 8601 string, e.g. '2025-06-14'
    service: e.g. 'grooming', 'boarding', 'daycare'
    """
    with httpx.Client(timeout=10) as client:
        resp = client.get(
            f"{GINGR_BASE}/reservations",
            headers=_headers(),
            params={"date": date, "service_type": service},
        )
        resp.raise_for_status()
        data = resp.json()

    reservations = data.get("data", [])
    log.info("gingr_availability_checked", date=date, service=service, count=len(reservations))
    return reservations
