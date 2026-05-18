# booking-intake-agent

An agentic booking intake system for a multi-channel pet store. Customers book via email, SMS, phone, or a Squarespace form — this agent ingests all of it, extracts structured booking requests, and notifies the owners via SMS for one-tap approval.

---

## The Problem

Customers book through whatever channel is convenient for them: email, text, phone call, or the website form. Everything lands in different places and has to be manually reconciled into Gingr (the store's booking system). This agent sits in the middle and automates that process.

---

## Architecture

```
Email (Gmail)
SMS (Twilio)
Voicemail (Twilio + Whisper)          →   FastAPI webhook
Squarespace form (custom JS → POST)   →   receiver

         ↓

Fetch customer history from Gingr
         ↓
LangChain agent extracts BookingRequest (Pydantic)
         ↓
Write to Postgres as status = pending
         ↓
SMS parents: "New booking: Mochi, grooming, Saturday. Reply Y to confirm"
SMS customer: "Got your request — we'll confirm shortly"
         ↓
Parent replies Y → status = confirmed
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Agent | LangChain + Claude API (claude-sonnet-4) |
| Database | PostgreSQL |
| Data validation | Pydantic |
| SMS / calls | Twilio |
| Voicemail transcription | Whisper |
| Email ingestion | Gmail API (push notifications) |
| Form ingestion | Custom JS on Squarespace → POST |
| Deployment | Railway |
| Local dev | Docker Compose + Ngrok |

---

## Database Schema

### `customers`
Stores customer contact info and preferred communication channel.

```sql
CREATE TABLE customers (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    email       VARCHAR(150),
    phone       VARCHAR(20),
    channel     VARCHAR(20),  -- 'email' | 'sms' | 'phone' | 'form'
    created_at  TIMESTAMP DEFAULT NOW()
);
```

### `pets`
Each customer can have multiple pets. Stores service preferences to help the agent fill in missing fields.

```sql
CREATE TABLE pets (
    id                 SERIAL PRIMARY KEY,
    customer_id        INTEGER REFERENCES customers(id),
    name               VARCHAR(100) NOT NULL,
    breed              VARCHAR(100),
    preferred_service  VARCHAR(100),  -- e.g. 'grooming', 'boarding', 'daycare'
    notes              TEXT,
    created_at         TIMESTAMP DEFAULT NOW()
);
```

### `bookings`
Core table. Tracks the full lifecycle of a booking request from intake to confirmation.

```sql
CREATE TABLE bookings (
    id               SERIAL PRIMARY KEY,
    customer_id      INTEGER REFERENCES customers(id),
    pet_id           INTEGER REFERENCES pets(id),
    service          VARCHAR(100) NOT NULL,
    requested_date   DATE NOT NULL,
    requested_time   TIME,
    status           VARCHAR(20) DEFAULT 'pending',  -- 'pending' | 'confirmed' | 'rejected'
    source_channel   VARCHAR(20),                    -- 'email' | 'sms' | 'phone' | 'form'
    raw_message_id   INTEGER REFERENCES messages(id),
    created_at       TIMESTAMP DEFAULT NOW(),
    updated_at       TIMESTAMP DEFAULT NOW()
);
```

### `messages`
Raw incoming messages. Every booking is traceable back to the original message for debugging.

```sql
CREATE TABLE messages (
    id          SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    channel     VARCHAR(20) NOT NULL,
    body        TEXT NOT NULL,
    direction   VARCHAR(10) DEFAULT 'inbound',  -- 'inbound' | 'outbound'
    created_at  TIMESTAMP DEFAULT NOW()
);
```

### `conversations`
Tracks active clarification threads — used when the agent needs to ask a follow-up question and wait for a reply.

```sql
CREATE TABLE conversations (
    id             SERIAL PRIMARY KEY,
    customer_id    INTEGER REFERENCES customers(id),
    booking_id     INTEGER REFERENCES bookings(id),
    status         VARCHAR(20) DEFAULT 'open',  -- 'open' | 'resolved'
    memory_state   JSONB,                        -- LangChain ConversationBufferMemory serialized
    created_at     TIMESTAMP DEFAULT NOW(),
    updated_at     TIMESTAMP DEFAULT NOW()
);
```

---

## Agent Logic

The LangChain agent runs on every inbound message. It has access to the following tools:

- `lookup_customer(phone_or_email)` — fetches customer + pet history from Gingr and local DB
- `check_availability(date, service)` — reads existing Gingr reservations for conflicts
- `create_draft_booking(BookingRequest)` — writes a pending booking to Postgres
- `send_sms(to, message)` — sends via Twilio (used for acks and clarifications)
- `notify_owners(booking_id)` — fires SMS to parents with approve/reject prompt

**Prompt strategy:** Customer history from Gingr is injected into the extraction prompt before the agent runs, so familiar customers with a single pet and a usual service rarely need clarification. `ConversationBufferMemory` handles multi-turn threads for new customers or ambiguous requests.

**Confidence threshold:** If the agent cannot extract a `requested_date` with reasonable confidence, it sends one clarifying SMS to the customer rather than creating a draft with a missing field.

---

## Webhook Endpoints

```
POST /webhook/sms        ← Twilio: inbound SMS
POST /webhook/call       ← Twilio: voicemail transcription callback
POST /webhook/email      ← Gmail push notification
POST /webhook/form       ← Squarespace custom JS form submission
POST /webhook/reply      ← Twilio: owner replies Y/N to confirm booking
```

---

## Squarespace Form Ingestion

No Zapier. Custom JS injected into the Squarespace page intercepts form submissions and POSTs to the FastAPI endpoint:

```javascript
window.addEventListener("submit", async (e) => {
  const formData = new FormData(e.target);
  await fetch("https://your-api.railway.app/webhook/form", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(Object.fromEntries(formData)),
  });
});
```

---

## Local Development

```bash
# Start FastAPI + Postgres locally
docker-compose up

# Expose local server to Twilio/Gmail webhooks
ngrok http 8000
# Paste the ngrok URL into Twilio console + Gmail push config
```

`docker-compose.yml` runs two services: the FastAPI app and a Postgres instance. Local dev uses Ollama (`llama3`) as the LLM to avoid API costs — swapped for Claude in production via env var.

---

## Deployment (Railway)

1. Push Docker image — Railway detects the `Dockerfile` and builds automatically
2. Add Postgres as a Railway add-on (one click)
3. Set env vars: `ANTHROPIC_API_KEY`, `TWILIO_AUTH_TOKEN`, `GMAIL_CREDENTIALS`, `DATABASE_URL`
4. Railway provides a public URL — plug into Twilio console and Gmail push config

---

## Estimated Monthly Costs

| Service | Cost |
|---|---|
| Railway (app + Postgres) | ~$5 |
| Twilio (number + SMS) | ~$2-3 |
| Claude API | ~$5-10 |
| Gmail API | Free |
| Gingr API | Included in existing subscription |
| Squarespace | Included in existing subscription |
| **Total** | **~$12-18/month** |

---

## Repo Structure (planned)

```
paw-router/
├── app/
│   ├── main.py              # FastAPI app + webhook routes
│   ├── agent.py             # LangChain agent + tools
│   ├── models.py            # Pydantic schemas (BookingRequest etc.)
│   ├── db.py                # Postgres connection + queries
│   ├── gingr.py             # Gingr API client
│   └── twilio_client.py     # Twilio send/receive helpers
├── sql/
│   └── schema.sql           # DB schema
├── Dockerfile
├── docker-compose.yml
└── README.md
```