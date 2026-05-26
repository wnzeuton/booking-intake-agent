"""PostgreSQL connection + all queries. No raw SQL outside this file."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import asyncpg
from asyncpg import Connection, Pool

from app.models import BookingRequest, BookingRecord

_pool: Optional[Pool] = None


async def get_pool() -> Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=os.environ["DATABASE_URL"],
            min_size=1,
            max_size=5,
        )
    return _pool


@asynccontextmanager
async def acquire() -> AsyncIterator[Connection]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


# ---------------------------------------------------------------------------
# Connection-explicit variants (used by agent tools running in threads,
# where the global pool's event loop may differ from the calling loop)
# ---------------------------------------------------------------------------

async def upsert_customer_conn(conn, *, name: str, email, phone, channel: str) -> int:
    row = await conn.fetchrow(
        """
        INSERT INTO customers (name, email, phone, channel)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (email) DO UPDATE
            SET name = EXCLUDED.name,
                phone = COALESCE(EXCLUDED.phone, customers.phone),
                channel = EXCLUDED.channel
        RETURNING id
        """,
        name, email, phone, channel,
    )
    return row["id"]


async def upsert_pet_conn(conn, *, customer_id: int, name: str,
                          breed=None, preferred_service=None, notes=None) -> int:
    row = await conn.fetchrow(
        """
        INSERT INTO pets (customer_id, name, breed, preferred_service, notes)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (customer_id, name) DO UPDATE
            SET breed = COALESCE(EXCLUDED.breed, pets.breed),
                preferred_service = COALESCE(EXCLUDED.preferred_service, pets.preferred_service),
                notes = COALESCE(EXCLUDED.notes, pets.notes)
        RETURNING id
        """,
        customer_id, name, breed, preferred_service, notes,
    )
    return row["id"]


async def create_booking_conn(conn, request, *, customer_id: int, pet_id: int) -> int:
    row = await conn.fetchrow(
        """
        INSERT INTO bookings
            (customer_id, pet_id, service, requested_date, requested_time,
             status, source_channel, raw_message_id)
        VALUES ($1, $2, $3, $4, $5, 'pending', $6, $7)
        RETURNING id
        """,
        customer_id, pet_id, request.service, request.requested_date,
        request.requested_time, request.source_channel, request.raw_message_id,
    )
    return row["id"]


async def get_booking_conn(conn, booking_id: int):
    row = await conn.fetchrow(
        """
        SELECT b.*, c.name AS customer_name, c.email AS customer_email, p.name AS pet_name
        FROM bookings b
        JOIN customers c ON c.id = b.customer_id
        JOIN pets      p ON p.id = b.pet_id
        WHERE b.id = $1
        """,
        booking_id,
    )
    return dict(row) if row else None


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

async def insert_message(
    *,
    customer_id: Optional[int],
    channel: str,
    body: str,
    direction: str = "inbound",
    thread_id: Optional[str] = None,
) -> int:
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO messages (customer_id, channel, body, direction, thread_id)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            customer_id,
            channel,
            body,
            direction,
            thread_id,
        )
        return row["id"]


async def is_known_thread(thread_id: str) -> bool:
    """Return True if we have any message (inbound or outbound) in this Gmail thread."""
    async with acquire() as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM messages WHERE thread_id = $1 LIMIT 1",
            thread_id,
        )
        return row is not None


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------

async def find_customer_by_email(email: str) -> Optional[dict]:
    async with acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM customers WHERE email = $1", email
        )
        return dict(row) if row else None


async def upsert_customer(
    *, name: str, email: Optional[str], phone: Optional[str], channel: str
) -> int:
    """Insert or update customer by email; returns customer id."""
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO customers (name, email, phone, channel)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (email) DO UPDATE
                SET name = EXCLUDED.name,
                    phone = COALESCE(EXCLUDED.phone, customers.phone),
                    channel = EXCLUDED.channel
            RETURNING id
            """,
            name,
            email,
            phone,
            channel,
        )
        return row["id"]


# ---------------------------------------------------------------------------
# Pets
# ---------------------------------------------------------------------------

async def find_pet(*, customer_id: int, pet_name: str) -> Optional[dict]:
    async with acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM pets WHERE customer_id = $1 AND LOWER(name) = LOWER($2)",
            customer_id,
            pet_name,
        )
        return dict(row) if row else None


async def upsert_pet(
    *,
    customer_id: int,
    name: str,
    breed: Optional[str] = None,
    preferred_service: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO pets (customer_id, name, breed, preferred_service, notes)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (customer_id, name) DO UPDATE
                SET breed = COALESCE(EXCLUDED.breed, pets.breed),
                    preferred_service = COALESCE(EXCLUDED.preferred_service, pets.preferred_service),
                    notes = COALESCE(EXCLUDED.notes, pets.notes)
            RETURNING id
            """,
            customer_id,
            name,
            breed,
            preferred_service,
            notes,
        )
        return row["id"]


# ---------------------------------------------------------------------------
# Bookings
# ---------------------------------------------------------------------------

async def create_booking(request: BookingRequest, *, customer_id: int, pet_id: int) -> int:
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO bookings
                (customer_id, pet_id, service, requested_date, requested_time,
                 status, source_channel, raw_message_id)
            VALUES ($1, $2, $3, $4, $5, 'pending', $6, $7)
            RETURNING id
            """,
            customer_id,
            pet_id,
            request.service,
            request.requested_date,
            request.requested_time,
            request.source_channel,
            request.raw_message_id,
        )
        return row["id"]


async def get_booking(booking_id: int) -> Optional[dict]:
    """Return booking row joined with customer name and pet name."""
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                b.*,
                c.name  AS customer_name,
                c.email AS customer_email,
                p.name  AS pet_name
            FROM bookings b
            JOIN customers c ON c.id = b.customer_id
            JOIN pets      p ON p.id = b.pet_id
            WHERE b.id = $1
            """,
            booking_id,
        )
        return dict(row) if row else None


async def update_booking_status(booking_id: int, status: str) -> None:
    assert status in ("pending", "confirmed", "rejected")
    async with acquire() as conn:
        await conn.execute(
            "UPDATE bookings SET status = $1 WHERE id = $2", status, booking_id
        )


# ---------------------------------------------------------------------------
# Conversations (clarification threads)
# ---------------------------------------------------------------------------

async def open_conversation(*, customer_id: int, booking_id: Optional[int] = None) -> int:
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO conversations (customer_id, booking_id, status)
            VALUES ($1, $2, 'open')
            RETURNING id
            """,
            customer_id,
            booking_id,
        )
        return row["id"]


async def get_open_conversation(customer_id: int) -> Optional[dict]:
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM conversations
            WHERE customer_id = $1 AND status = 'open'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            customer_id,
        )
        return dict(row) if row else None


async def resolve_conversation(conversation_id: int) -> None:
    async with acquire() as conn:
        await conn.execute(
            "UPDATE conversations SET status = 'resolved' WHERE id = $1",
            conversation_id,
        )


# ---------------------------------------------------------------------------
# App state (generic key-value)
# ---------------------------------------------------------------------------

async def get_state(key: str) -> Optional[str]:
    async with acquire() as conn:
        row = await conn.fetchrow(
            "SELECT value FROM app_state WHERE key = $1", key
        )
        return row["value"] if row else None


async def set_state(key: str, value: str) -> None:
    async with acquire() as conn:
        await conn.execute(
            """
            INSERT INTO app_state (key, value, updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value,
                    updated_at = NOW()
            """,
            key,
            value,
        )
