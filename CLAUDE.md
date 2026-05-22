# booking-intake-agent

Agentic booking intake system for a multi-channel pet store. Ingests booking requests from email (Gmail) and a Squarespace form, extracts structured booking data via a LangChain + Llama 3 agent, and notifies owners via email for approval.

## Stack

- **Runtime:** Python 3.11
- **Backend:** FastAPI + Docker
- **Agent:** LangChain + Llama 3 (4-bit quantized via llama.cpp, served on EC2)
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

For local LLM: set `LLAMA_ENDPOINT=http://localhost:11434` and run Ollama with `ollama run llama3`.

Environment variables go in `.env` — never commit this file.

Required: `GMAIL_CREDENTIALS`, `DATABASE_URL`, `OWNER_EMAIL`, `GINGR_API_KEY`, `LLAMA_ENDPOINT`

## AWS Infrastructure

- **EC2 t2.micro** — runs FastAPI container; llama.cpp inference server runs on a separate GPU instance or is proxied
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
  gingr.py           # Gingr API client (read-only)
  email_client.py    # Gmail API send/receive helpers
sql/
  schema.sql         # DB schema (source of truth)
.github/
  workflows/
    deploy.yml       # GitHub Actions CI/CD pipeline
Dockerfile
docker-compose.yml
.env.example
```

## Conventions

- All webhook routes live in `main.py` and follow the pattern `POST /webhook/{channel}` where channel is `email`, `form`, or `reply`
- LangChain tools are defined in `agent.py` and named in snake_case verbs: `lookup_customer`, `check_availability`, `create_draft_booking`, `notify_owners`, `send_clarification_email`
- Pydantic models live in `models.py` — never define schemas inline in route handlers
- All DB queries go through `db.py` — no raw SQL outside that file
- Booking status values: `pending`, `confirmed`, `rejected` — no other values
- Source channel values: `email`, `form` — no other values for MVP

## Key Constraints

- Gingr API is read-only — never attempt writes to Gingr
- The agent must inject customer history from Gingr before running extraction
- If `requested_date` cannot be extracted with confidence, send one clarifying email to the customer — do not create a draft with a null or uncertain date
- Owner approval flow is email-only: owners reply Y to confirm, N to reject
- Llama 3 runs via llama.cpp — never call an external LLM API in production
- No Twilio / SMS in MVP scope
