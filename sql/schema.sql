-- booking-intake-agent: source-of-truth DB schema
-- Apply with: psql $DATABASE_URL -f sql/schema.sql

CREATE TABLE IF NOT EXISTS customers (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    email       VARCHAR(150),
    phone       VARCHAR(20),
    channel     VARCHAR(20),          -- 'email' | 'form'
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pets (
    id                 SERIAL PRIMARY KEY,
    customer_id        INTEGER REFERENCES customers(id) ON DELETE CASCADE,
    name               VARCHAR(100) NOT NULL,
    breed              VARCHAR(100),
    preferred_service  VARCHAR(100),  -- e.g. 'grooming', 'boarding', 'daycare'
    notes              TEXT,
    created_at         TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    id          SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    channel     VARCHAR(20) NOT NULL,   -- 'email' | 'form'
    body        TEXT NOT NULL,
    direction   VARCHAR(10) DEFAULT 'inbound',  -- 'inbound' | 'outbound'
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bookings (
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

CREATE TABLE IF NOT EXISTS conversations (
    id             SERIAL PRIMARY KEY,
    customer_id    INTEGER REFERENCES customers(id),
    booking_id     INTEGER REFERENCES bookings(id),
    status         VARCHAR(20) DEFAULT 'open',  -- 'open' | 'resolved'
    memory_state   JSONB,                        -- LangChain ConversationBufferMemory serialized
    created_at     TIMESTAMP DEFAULT NOW(),
    updated_at     TIMESTAMP DEFAULT NOW()
);

-- Auto-update updated_at on bookings
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER bookings_updated_at
    BEFORE UPDATE ON bookings
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE OR REPLACE TRIGGER conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
