"""
Run the agent directly on a message — no webhooks, no DB, no email.
Just paste text in, see what the agent thinks and does.

Usage:
    python scripts/test_agent.py
    python scripts/test_agent.py "Book grooming for my dog Max on June 20th"
    python scripts/test_agent.py --email user@example.com "I need boarding for Luna next Saturday"
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Make sure the project root is on the path regardless of where this is run from
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env so LLAMA_ENDPOINT etc. are available
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Point Ollama at localhost (not host.docker.internal — we're running on the host now)
# host.docker.internal only resolves inside Docker — force localhost when running on host
os.environ["LLAMA_ENDPOINT"] = "http://localhost:11434"
os.environ["DRY_RUN"] = "1"  # never send real emails during agent testing
os.environ.setdefault("OWNER_EMAIL", "test@example.com")
os.environ.setdefault("GINGR_API_KEY", "fake")
# Override DB URL: Docker internal hostname 'postgres' → localhost for host runs
os.environ["DATABASE_URL"] = "postgresql://booking:booking@localhost:5432/booking"

from app.agent import run_intake  # noqa: E402 — after env setup


async def main(message: str, email: str | None):
    print(f"\n{'='*60}")
    print(f"MESSAGE: {message}")
    print(f"EMAIL:   {email or '(none)'}")
    print(f"{'='*60}\n")

    result = await run_intake(
        message_body=message,
        sender_email=email,
        source_channel="form",
    )

    print(f"\n{'='*60}")
    print(f"STATUS:     {result['status']}")
    print(f"BOOKING ID: {result['booking_id']}")
    print(f"OUTPUT:\n{result['output']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("message", nargs="?",
                        default="Hi, I'd like to book a grooming for my dog Max on June 15th.")
    parser.add_argument("--email", default=None)
    args = parser.parse_args()

    asyncio.run(main(args.message, args.email))
