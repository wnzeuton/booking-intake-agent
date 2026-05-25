"""
One-time script to complete the Gmail OAuth flow and print the
GMAIL_CREDENTIALS JSON string for your .env file.

Usage:
    pip install google-auth-oauthlib google-api-python-client
    python scripts/get_gmail_token.py path/to/client_secret.json

What it does:
    1. Opens a browser tab for Google's consent screen
    2. You log in as the Gmail account that will send/receive booking emails
    3. Exchanges the auth code for access + refresh tokens
    4. Prints the JSON to paste into GMAIL_CREDENTIALS in your .env

Run this once locally. The refresh token doesn't expire unless you
revoke access in your Google account settings.
"""

import json
import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/get_gmail_token.py path/to/client_secret.json")
        sys.exit(1)

    client_secret_path = Path(sys.argv[1])
    if not client_secret_path.exists():
        print(f"Error: file not found: {client_secret_path}")
        sys.exit(1)

    flow = InstalledAppFlow.from_client_secrets_file(
        str(client_secret_path),
        scopes=SCOPES,
    )

    # Opens http://localhost on a random port for the redirect
    creds = flow.run_local_server(port=0, prompt="consent")

    output = {
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "refresh_token": creds.refresh_token,
        "token_uri": "https://oauth2.googleapis.com/token",
    }

    print("\n" + "=" * 60)
    print("Add this to your .env as GMAIL_CREDENTIALS (one line):")
    print("=" * 60)
    print(f"GMAIL_CREDENTIALS='{json.dumps(output)}'")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
