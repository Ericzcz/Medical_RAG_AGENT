CREATE TABLE IF NOT EXISTS medical_records (
    record_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    source_session_id TEXT,
    record_type TEXT NOT NULL,
    visit_date DATE,
    notes TEXT NOT NULL,
    source_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS medical_record_items (
    item_id UUID PRIMARY KEY,
    record_id UUID NOT NULL REFERENCES medical_records(record_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    field_type TEXT NOT NULL,
    field_value TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_medical_records_user_created
ON medical_records (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_medical_record_items_user_field_created
ON medical_record_items (user_id, field_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_medical_record_items_record_id
ON medical_record_items (record_id);