# booking-intake-agent

An agentic booking intake system for a pet store. Customers book via email or a Squarespace form — this agent ingests both channels, extracts structured booking requests, and emails the owners for one-tap approval.

---

## The Problem

Customers book through whatever channel is convenient for them: email or the website form. Everything has to be manually reconciled into Gingr (the store's booking system). This agent sits in the middle and automates that process.

---

## Architecture

```
Email (Gmail)
Squarespace form (custom JS → POST)   →   FastAPI webhook receiver

         ↓

Fetch customer history from Gingr
         ↓
LangChain + Llama 3 agent extracts BookingRequest (Pydantic)
         ↓
Write to Postgres as status = pending
         ↓
Email owners: "New booking: Mochi, grooming, Saturday. Reply Y to confirm"
         ↓
Owner replies Y → status = confirmed
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Agent | LangChain + Llama 3 (4-bit quantized via llama.cpp) |
| Database | PostgreSQL (AWS RDS) |
| Data validation | Pydantic v2 |
| Email ingestion | Gmail API (push notifications) |
| Email outbound | Gmail API (owner approval + clarifications) |
| Form ingestion | Custom JS on Squarespace → POST |
| Deployment | AWS EC2 + ECR |
| CI/CD | GitHub Actions → ECR → EC2 (SSM) |
| Local dev | Docker Compose + Ngrok + Ollama |

---

## Database Schema

### `customers`
Stores customer contact info and preferred communication channel.

```sql
CREATE TABLE customers (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    email       VARCHAR(150) UNIQUE,
    phone       VARCHAR(20),
    channel     VARCHAR(20),  -- 'email' | 'form'
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
    created_at         TIMESTAMP DEFAULT NOW(),
    UNIQUE (customer_id, name)
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
    source_channel   VARCHAR(20),                    -- 'email' | 'form'
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
    channel     VARCHAR(20) NOT NULL,  -- 'email' | 'form'
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
- `notify_owners(booking_id)` — emails owners with an approve/reject prompt
- `send_clarification_email(to, customer_name, missing_field)` — one clarifying email when date is ambiguous

**Prompt strategy:** Customer history from Gingr is injected into the extraction prompt before the agent runs, so familiar customers with a single pet and a usual service rarely need clarification. `ConversationBufferMemory` handles multi-turn threads for new customers or ambiguous requests.

**Confidence threshold:** If the agent cannot extract a `requested_date` with reasonable confidence, it sends one clarifying email to the customer rather than creating a draft with a missing field.

---

## Webhook Endpoints

```
POST /webhook/email      ← Gmail push notification (Pub/Sub)
POST /webhook/form       ← Squarespace custom JS form submission
POST /webhook/reply      ← Owner replies Y/N to approve/reject booking
```

---

## Squarespace Form Ingestion

No Zapier. Custom JS injected into the Squarespace page intercepts form submissions and POSTs to the FastAPI endpoint:

```javascript
window.addEventListener("submit", async (e) => {
  const formData = new FormData(e.target);
  await fetch("https://<your-ec2-domain>/webhook/form", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(Object.fromEntries(formData)),
  });
});
```

---

## Local Development

```bash
# Copy and fill in env vars
cp .env.example .env

# Start FastAPI + Postgres
docker-compose up

# Expose local server to Gmail push webhook
ngrok http 8000
# Paste the ngrok URL into Gmail push config (Google Cloud Pub/Sub)
```

For the local LLM, run Ollama and set `LLAMA_ENDPOINT=http://localhost:11434` in `.env`:

```bash
ollama run llama3
```

---

## Deployment (AWS)

GitHub Actions builds and deploys on every push to `main`:

1. **OIDC auth** — no long-lived AWS keys; GitHub assumes an IAM role via OIDC
2. **ECR push** — Docker image tagged with the commit SHA
3. **EC2 deploy** — `aws ssm send-command` pulls the new image and restarts the container (no open SSH port required)

Required GitHub secrets: `AWS_DEPLOY_ROLE_ARN`, `EC2_INSTANCE_ID`

Required env vars on EC2 (in `/opt/booking-intake-agent/.env`):

```
DATABASE_URL
GMAIL_CREDENTIALS
OWNER_EMAIL
GINGR_API_KEY
LLAMA_ENDPOINT
```

---

## AWS Infrastructure

| Resource | Details |
|---|---|
| EC2 t2.micro | Runs the FastAPI container |
| RDS db.t3.micro | Managed Postgres; private subnet, accessible only from EC2 |
| ECR | Docker image registry |
| CloudWatch | Application logs, agent reasoning traces, webhook hits |
| IAM | EC2 instance role scoped to ECR pull + CloudWatch write + RDS access |
| SSM | Used for deployments (no open port 22) |

---

## Estimated Monthly Costs

| Service | Cost |
|---|---|
| EC2 t2.micro | ~$8 |
| RDS db.t3.micro | ~$13 |
| ECR | ~$1 |
| CloudWatch | ~$1 |
| Gmail API | Free |
| Gingr API | Included in existing subscription |
| Squarespace | Included in existing subscription |
| **Total** | **~$23/month** |

---

## Repo Structure

```
booking-intake-agent/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app + webhook routes
│   ├── agent.py             # LangChain agent + tools
│   ├── models.py            # Pydantic schemas (BookingRequest etc.)
│   ├── db.py                # Postgres connection + queries (asyncpg)
│   ├── gingr.py             # Gingr API client (read-only)
│   └── email_client.py      # Gmail API send/receive helpers
├── sql/
│   └── schema.sql           # DB schema (source of truth)
├── .github/
│   └── workflows/
│       └── deploy.yml       # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```
