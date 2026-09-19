-- db/migrations/001_initial_schema.sql
-- Phase 4: Initial PostgreSQL schema for SatQuery AI
-- Idempotent: all CREATE TABLE use IF NOT EXISTS

-- ─── Sessions ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
  id               TEXT PRIMARY KEY,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at       TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 minutes'),
  call_count       INTEGER NOT NULL DEFAULT 0,
  history          JSONB NOT NULL DEFAULT '[]',
  metadata         JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions (expires_at);

-- ─── Assets (image metadata only — no bytes stored) ────────────────────────────
CREATE TABLE IF NOT EXISTS assets (
  id               TEXT PRIMARY KEY,
  session_id       TEXT REFERENCES sessions(id) ON DELETE CASCADE,
  storage_key      TEXT NOT NULL,
  sha256           TEXT NOT NULL,
  mime_type        TEXT NOT NULL DEFAULT 'image/jpeg',
  size_bytes       INTEGER,
  width_px         INTEGER,
  height_px        INTEGER,
  modality         TEXT NOT NULL DEFAULT 'optical',
  sensor_hint      TEXT,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at       TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 minutes'),
  metadata         JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_assets_session_id ON assets (session_id);
CREATE INDEX IF NOT EXISTS idx_assets_sha256 ON assets (sha256);
CREATE INDEX IF NOT EXISTS idx_assets_expires_at ON assets (expires_at);

-- ─── Analysis Runs ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS analysis_runs (
  id               TEXT PRIMARY KEY,
  session_id       TEXT REFERENCES sessions(id) ON DELETE CASCADE,
  task_type        TEXT NOT NULL,
  status           TEXT NOT NULL DEFAULT 'pending',
  agent_version    TEXT NOT NULL DEFAULT 'SatQuery-Agent-v3.0',
  tool_id          TEXT,
  started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at     TIMESTAMPTZ,
  duration_ms      INTEGER,
  is_demo          BOOLEAN NOT NULL DEFAULT FALSE,
  is_cached        BOOLEAN NOT NULL DEFAULT FALSE,
  error_tag        TEXT,
  metadata         JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_analysis_runs_session_id ON analysis_runs (session_id);
CREATE INDEX IF NOT EXISTS idx_analysis_runs_started_at ON analysis_runs (started_at);

-- ─── Analysis Inputs ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS analysis_inputs (
  id               TEXT PRIMARY KEY,
  run_id           TEXT NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
  input_type       TEXT NOT NULL DEFAULT 'query',
  query_text       TEXT,
  asset_id         TEXT REFERENCES assets(id) ON DELETE SET NULL,
  modality         TEXT,
  label            TEXT,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analysis_inputs_run_id ON analysis_inputs (run_id);

-- ─── Analysis Results ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS analysis_results (
  id               TEXT PRIMARY KEY,
  run_id           TEXT NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
  result_json      JSONB NOT NULL,
  confidence       TEXT,
  confidence_pct   REAL,
  task_type        TEXT,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analysis_results_run_id ON analysis_results (run_id);

-- ─── Agent Trace Steps ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_trace_steps (
  id               SERIAL PRIMARY KEY,
  run_id           TEXT NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
  step_index       INTEGER NOT NULL,
  tool             TEXT NOT NULL,
  description      TEXT,
  input_summary    TEXT,
  output_summary   TEXT,
  duration_ms      REAL,
  status           TEXT NOT NULL DEFAULT 'success',
  parameters       JSONB,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trace_steps_run_id ON agent_trace_steps (run_id);

-- ─── Analysis Cache ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS analysis_cache (
  cache_key        TEXT PRIMARY KEY,
  session_id       TEXT,
  task_type        TEXT NOT NULL,
  result_json      JSONB NOT NULL,
  model_id         TEXT,
  agent_version    TEXT,
  cache_schema_ver INTEGER NOT NULL DEFAULT 1,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at       TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 minutes'),
  hit_count        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON analysis_cache (expires_at);
CREATE INDEX IF NOT EXISTS idx_cache_task_type ON analysis_cache (task_type);

-- ─── Datasets ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS datasets (
  id                  TEXT PRIMARY KEY,
  name                TEXT NOT NULL UNIQUE,
  description         TEXT,
  source_url          TEXT,
  license             TEXT,
  modality            TEXT NOT NULL DEFAULT 'optical',
  version             TEXT,
  storage_uri         TEXT,
  availability_status TEXT NOT NULL DEFAULT 'not_downloaded',
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─── Dataset Versions ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dataset_versions (
  id               TEXT PRIMARY KEY,
  dataset_id       TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  version_tag      TEXT NOT NULL,
  release_date     DATE,
  notes            TEXT,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─── Dataset Splits ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dataset_splits (
  id               TEXT PRIMARY KEY,
  dataset_id       TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  split_name       TEXT NOT NULL,
  sample_count     INTEGER,
  storage_uri      TEXT,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─── Dataset Task Support ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dataset_task_support (
  id               SERIAL PRIMARY KEY,
  dataset_id       TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  task_type        TEXT NOT NULL,
  benchmark_name   TEXT,
  UNIQUE (dataset_id, task_type)
);

-- ─── Models ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS models (
  id                  TEXT PRIMARY KEY,
  name                TEXT NOT NULL,
  version             TEXT,
  enabled             BOOLEAN NOT NULL DEFAULT FALSE,
  checkpoint_uri      TEXT,
  remote_sensing_adapted BOOLEAN NOT NULL DEFAULT FALSE,
  required_dataset    TEXT REFERENCES datasets(id) ON DELETE SET NULL,
  availability        TEXT NOT NULL DEFAULT 'unavailable',
  unavailable_reason  TEXT,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─── Model Capabilities ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_capabilities (
  id               SERIAL PRIMARY KEY,
  model_id         TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  task_type        TEXT NOT NULL,
  benchmark        TEXT,
  metric_name      TEXT,
  metric_value     REAL,
  notes            TEXT,
  UNIQUE (model_id, task_type)
);
