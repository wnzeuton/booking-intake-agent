"""
Call Gmail's watch API to start pushing inbox notifications to Pub/Sub.
Uses the GMAIL_CREDENTIALS already in .env.

Usage:
    python scripts/setup_gmail_watch.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

load_dotenv(Path(__file__).parent.parent / ".env")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]

TOPIC = "projects/booking-intake-agent/topics/gmail-push"


async def seed_history_id(history_id: str) -> None:
    """Store the initial historyId in the DB so the first notification has a start point."""
    conn = await asyncpg.connect(dsn=os.environ["DATABASE_URL"])
    try:
        await conn.execute(
            """
            INSERT INTO app_state (key, value, updated_at)
            VALUES ('gmail_history_id', $1, NOW())
            ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = NOW()
            """,
            history_id,
        )
        print(f"  Seeded gmail_history_id={history_id} into DB")
    finally:
        await conn.close()


def main():
    raw = os.environ.get("GMAIL_CREDENTIALS")
    if not raw:
        print("Error: GMAIL_CREDENTIALS not set in .env")
        sys.exit(1)

    creds = Credentials.from_authorized_user_info(json.loads(raw), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

    service = build("gmail", "v1", credentials=creds, cache_discovery=False)

    result = service.users().watch(
        userId="me",
        body={"topicName": TOPIC, "labelIds": ["INBOX"]},
    ).execute()

    history_id = str(result["historyId"])

    print("Gmail watch set up successfully:")
    print(f"  historyId:  {history_id}")
    print(f"  expiration: {result['expiration']} (ms since epoch, ~7 days)")
    print()
    print("Gmail will now push inbox notifications to:")
    print(f"  {TOPIC}")
    print()

    asyncio.run(seed_history_id(history_id))


if __name__ == "__main__":
    main()
