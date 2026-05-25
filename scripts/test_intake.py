"""
Integration test — posts a booking request directly to /webhook/form
and polls Postgres for the resulting booking record.

Usage:
    python scripts/test_intake.py
    python scripts/test_intake.py --message "Book daycare for Luna on June 20th"
"""

import argparse
import asyncio
import json
import os
import sys
import time

import asyncpg
import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://booking:booking@localhost:5432/booking",
)

# ---------------------------------------------------------------------------
# Default test payload — enough info for the agent to make a booking
# ---------------------------------------------------------------------------

DEFAULT_PAYLOAD = {
    "name": "Will Nzeuton",
    "email": "will.nzeuton@gmail.com",
    "pet_name": "Max",
    "service": "grooming",
    "requested_date": "2025-06-15",
    "notes": "He's a golden retriever, very friendly.",
}


async def get_latest_booking(conn, after_id: int) -> dict | None:
    """Poll for a booking row created after `after_id`."""
    row = await conn.fetchrow(
        """
        SELECT b.id, b.status, b.service, b.requested_date,
               c.name AS customer_name, c.email AS customer_email,
               p.name AS pet_name
        FROM bookings b
        JOIN customers c ON c.id = b.customer_id
        JOIN pets      p ON p.id = b.pet_id
        WHERE b.id > $1
        ORDER BY b.id DESC
        LIMIT 1
        """,
        after_id,
    )
    return dict(row) if row else None


async def max_booking_id(conn) -> int:
    row = await conn.fetchrow("SELECT COALESCE(MAX(id), 0) AS m FROM bookings")
    return row["m"]


async def run_test(payload: dict):
    # Connect to Postgres directly (host port, not Docker internal)
    conn = await asyncpg.connect(DATABASE_URL)
    baseline_id = await max_booking_id(conn)
    print(f"  baseline booking id: {baseline_id}")

    # POST to the form webhook
    print(f"\n→ POST {API_BASE}/webhook/form")
    print(f"  payload: {json.dumps(payload, indent=2)}")

    async with httpx.AsyncClient(timeout=30) as client:
        t0 = time.monotonic()
        resp = await client.post(f"{API_BASE}/webhook/form", json=payload)
        elapsed = time.monotonic() - t0

    print(f"\n← HTTP {resp.status_code}  ({elapsed:.1f}s)")
    if resp.status_code not in (200, 204):
        print(f"  ERROR: {resp.text}")
        await conn.close()
        sys.exit(1)

    # Agent runs as a background task — poll DB until a booking appears (up to 3 min)
    print("\n⏳ Waiting for agent to finish (polling DB)...")
    for i in range(90):
        booking = await get_latest_booking(conn, baseline_id)
        if booking:
            break
        if i % 10 == 0 and i > 0:
            print(f"  still waiting... ({i}s) — check 'docker-compose logs api -f' for agent trace")
        await asyncio.sleep(2)

    await conn.close()

    if not booking:
        print("  ✗ No booking record found after request completed.")
        print("  Check 'docker-compose logs api' for agent output.")
        sys.exit(1)

    print("\n✓ Booking created!")
    print(f"  id:       {booking['id']}")
    print(f"  customer: {booking['customer_name']} ({booking['customer_email']})")
    print(f"  pet:      {booking['pet_name']}")
    print(f"  service:  {booking['service']}")
    print(f"  date:     {booking['requested_date']}")
    print(f"  status:   {booking['status']}")


def main():
    parser = argparse.ArgumentParser(description="Integration test for booking intake agent")
    parser.add_argument("--name", default=DEFAULT_PAYLOAD["name"])
    parser.add_argument("--email", default=DEFAULT_PAYLOAD["email"])
    parser.add_argument("--pet", default=DEFAULT_PAYLOAD["pet_name"])
    parser.add_argument("--service", default=DEFAULT_PAYLOAD["service"])
    parser.add_argument("--date", default=DEFAULT_PAYLOAD["requested_date"])
    parser.add_argument("--notes", default=DEFAULT_PAYLOAD["notes"])
    args = parser.parse_args()

    payload = {
        "name": args.name,
        "email": args.email,
        "pet_name": args.pet,
        "service": args.service,
        "requested_date": args.date,
        "notes": args.notes,
    }

    asyncio.run(run_test(payload))


if __name__ == "__main__":
    main()
