# booking-intake-agent

Agentic booking intake system for a multi-channel pet store. Ingests booking requests from email (Gmail) and a Squarespace form, extracts structured booking data via a LangChain + Groq agent, notifies owners via email for approval, and writes confirmed bookings into Gingr via Playwright browser automation.

## Stack

- **Runtime:** Python 3.11
- **Backend:** FastAPI + Docker
- **Agent:** LangChain + Groq API (free tier)
- **CRM automation:** Playwright — writes confirmed bookings into Gingr web UI
- **Database:** PostgreSQL (AWS RDS, db.t3.micro)
- **Validation:** Pydantic v2
- **Email:** Gmail API (push notifications inbound + outbound send)
- **Deployment:** AWS EC2 (t2.micro) via Docker + ECR
- **Container registry:** AWS ECR
- **Logging:** AWS CloudWatch
- **CI/CD:** GitHub Actions → ECR → EC2 (SSM send-command, no SSH)
- **Local dev:** Docker Compose + Ngrok

## Local Development

```bash
cp .env.example .env        # fill in values
docker-compose up           # starts FastAPI + Postgres
ngrok http 8000             # exposes webhooks to Gmail push
```

For local agent testing without webhooks:
```bash
source .venv/bin/activate
python scripts/test_agent.py "Book grooming for Max on June 20, name Will, will@example.com"
```

Eval suite (`scripts/evals.py`) checks tool selection + extracted fields against expected results, with all DB/email/Gingr side effects patched out:
```bash
python scripts/evals.py --quick   # ~35-case representative subset — use this while iterating; fits one day's Groq free-tier token budget on any model
python scripts/evals.py           # full 104-case suite — run before merging or when comparing/changing GROQ_MODEL; may need to be paced across the free-tier daily token limit depending on the model
```

Set `DRY_RUN=1` in `.env` to log emails instead of sending during development.

Environment variables go in `.env` — never commit this file.

Required: `GROQ_API_KEY`, `GMAIL_CREDENTIALS`, `DATABASE_URL`, `OWNER_EMAIL`, `GINGR_USERNAME`, `GINGR_PASSWORD`

## AWS Infrastructure

- **EC2 t2.micro** — runs FastAPI container
- **RDS db.t3.micro** — managed Postgres, private subnet, only reachable from EC2 via security group
- **ECR** — stores Docker image; GitHub Actions pushes on every merge to main
- **CloudWatch** — all application logs, agent reasoning traces, webhook hits
- **IAM** — EC2 instance role scoped to ECR pull, CloudWatch write, RDS access; GitHub Actions uses OIDC (no long-lived keys)
- **SSM** — GitHub Actions deploys to EC2 via `aws ssm send-command` (no open SSH port)

## Project Structure

```
app/
  __init__.py
  main.py            # FastAPI app + webhook routes
  agent.py           # LangChain agent + tool definitions
  models.py          # Pydantic schemas
  db.py              # Postgres connection + queries (asyncpg)
  gingr.py           # Gingr API client (read-only lookups)
  gingr_writer.py    # Playwright automation — writes confirmed bookings into Gingr
  email_client.py    # Gmail API send/receive helpers
sql/
  schema.sql         # DB schema (source of truth)
scripts/
  test_agent.py      # Run agent directly against a message (no webhooks needed)
  evals.py           # Eval suite (--quick for a fast subset, full run for merges/model changes)
  get_gmail_token.py # One-time Gmail OAuth flow
  setup_gmail_watch.py # Register Gmail Pub/Sub push notifications
.github/
  workflows/
    deploy.yml       # GitHub Actions CI/CD pipeline
Dockerfile
docker-compose.yml
.env.example
```

## Conventions

- All webhook routes live in `main.py` and follow the pattern `POST /webhook/{channel}` where channel is `email`, `form`, or `reply`
- LangChain tools are defined in `agent.py` and named in snake_case verbs: `create_draft_booking`, `notify_owners`, `send_clarification_email`
- Pydantic models live in `models.py` — never define schemas inline in route handlers
- All DB queries go through `db.py` — no raw SQL outside that file
- Booking status values: `pending`, `confirmed`, `rejected` — no other values
- Source channel values: `email`, `form` — no other values for MVP

## Key Constraints

- Gingr API is read-only — all writes go through Playwright (`gingr_writer.py`)
- If `requested_date` cannot be extracted with confidence, send one clarifying email to the customer — do not create a draft with a null or uncertain date
- Owner approval flow is email-only: owners reply Y to confirm, N to reject
- On Y reply: update booking status to confirmed, then trigger Playwright to write into Gingr
- LLM is served via Groq's free tier (`GROQ_MODEL`, no dev/prod split needed unless eval results say otherwise); free-tier rate limits (~30 RPM) are acceptable to hit at this booking volume, not something to design around
- No Twilio / SMS in MVP scope
- Set `DRY_RUN=1` to prevent real emails during local testing
