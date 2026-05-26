# booking-intake-agent

An agentic booking intake system for a pet store. Customers book via email or a Squarespace form — this agent ingests both channels, extracts structured booking requests, emails the owner for one-tap approval, and writes confirmed bookings directly into the store's CRM (Gingr) via Playwright automation.

---

## The Problem

Customers book through whatever channel is convenient: email or the website form. Everything has to be manually reconciled into Gingr (the store's booking CRM). This agent sits in the middle and automates the entire flow — from raw message to confirmed CRM entry.

---

## Architecture

```
Email (Gmail)
Squarespace form (custom JS → POST)   →   FastAPI webhook receiver
                                                    ↓
                                      Claude agent extracts BookingRequest
                                                    ↓
                                      Write to Postgres (status = pending)
                                                    ↓
                                      Email owner: "New booking: Max, grooming,
                                      June 20. Reply Y to confirm"
                                                    ↓
                                      Owner replies Y → status = confirmed
                                                    ↓
                                      Playwright writes booking into Gingr
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Agent | LangChain + Claude API (Haiku / Sonnet) |
| CRM automation | Playwright |
| Database | PostgreSQL (AWS RDS) |
| Data validation | Pydantic v2 |
| Email ingestion | Gmail API (Pub/Sub push notifications) |
| Email outbound | Gmail API (owner approval + clarifications) |
| Form ingestion | Custom JS on Squarespace → POST |
| Deployment | AWS EC2 + ECR |
| CI/CD | GitHub Actions → ECR → EC2 (SSM) |
| Local dev | Docker Compose + Ngrok |

---

## Database Schema

### `customers`
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
```sql
CREATE TABLE pets (
    id                 SERIAL PRIMARY KEY,
    customer_id        INTEGER REFERENCES customers(id),
    name               VARCHAR(100) NOT NULL,
    breed              VARCHAR(100),
    preferred_service  VARCHAR(100),
    notes              TEXT,
    created_at         TIMESTAMP DEFAULT NOW(),
    UNIQUE (customer_id, name)
);
```

### `bookings`
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
```sql
CREATE TABLE messages (
    id          SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    channel     VARCHAR(20) NOT NULL,
    body        TEXT NOT NULL,
    direction   VARCHAR(10) DEFAULT 'inbound',
    created_at  TIMESTAMP DEFAULT NOW()
);
```

### `app_state`
```sql
CREATE TABLE app_state (
    key        VARCHAR(100) PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW()
);
```

---

## Agent

The LangChain agent runs on every inbound message. It uses Claude's native tool-calling API (not text-based ReAct) for reliable structured output.

**Tools:**
- `create_draft_booking` — validates and writes a pending booking to Postgres
- `notify_owners` — emails the owner with approve/reject prompt
- `send_clarification_email` — one email to the customer when required fields are missing

**Behavior:**
- If all fields are present → create booking, email owner
- If `customer_name`, `pet_name`, or `requested_date` are missing → send clarification email and stop
- Form webhook with all required fields skips the LLM entirely and writes directly to the DB

---

## Webhook Endpoints

```
POST /webhook/email      ← Gmail push notification (Pub/Sub)
POST /webhook/form       ← Squarespace custom JS form submission
POST /webhook/reply      ← Owner replies Y/N to approve/reject booking
```

---

## Squarespace Form Ingestion

No Zapier. Custom JS injected into the Squarespace page intercepts form submissions:

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
cp .env.example .env
docker-compose up       # FastAPI + Postgres
ngrok http 8000         # expose webhooks to Gmail push
```

Test the agent directly without webhooks:

```bash
source .venv/bin/activate
python scripts/test_agent.py "Book grooming for Max on June 20th, I'm Will, will@example.com"

# With DRY_RUN=1 set in .env, no real emails are sent
```

---

## Deployment (AWS)

GitHub Actions builds and deploys on every push to `main`:

1. **OIDC auth** — no long-lived AWS keys; GitHub assumes an IAM role via OIDC
2. **ECR push** — Docker image tagged with the commit SHA
3. **EC2 deploy** — `aws ssm send-command` pulls the new image and restarts the container (no open SSH port)

Required GitHub secrets: `AWS_DEPLOY_ROLE_ARN`, `EC2_INSTANCE_ID`

Required env vars on EC2:

```
ANTHROPIC_API_KEY
DATABASE_URL
GMAIL_CREDENTIALS
OWNER_EMAIL
GINGR_USERNAME
GINGR_PASSWORD
```

---

## AWS Infrastructure

| Resource | Details |
|---|---|
| EC2 t2.micro | Runs the FastAPI + Playwright container |
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
| Claude API | ~$1 (Haiku, ~10k bookings/mo) |
| Gmail API | Free |
| Gingr | Included in existing subscription |
| Squarespace | Included in existing subscription |
| **Total** | **~$24/month** |

---

## Repo Structure

```
booking-intake-agent/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app + webhook routes
│   ├── agent.py             # LangChain agent + tools
│   ├── models.py            # Pydantic schemas
│   ├── db.py                # Postgres connection + queries (asyncpg)
│   ├── gingr.py             # Gingr API client (read-only lookups)
│   ├── gingr_writer.py      # Playwright — writes confirmed bookings into Gingr
│   └── email_client.py      # Gmail API send/receive helpers
├── sql/
│   └── schema.sql           # DB schema (source of truth)
├── scripts/
│   ├── test_agent.py        # Run agent directly (no webhooks needed)
│   ├── get_gmail_token.py   # One-time Gmail OAuth flow
│   └── setup_gmail_watch.py # Register Gmail Pub/Sub push notifications
├── .github/
│   └── workflows/
│       └── deploy.yml       # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```
