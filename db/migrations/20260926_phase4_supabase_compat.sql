-- ORBITAL-AI Phase 4 Supabase compatibility
-- Keeps the repository layer aligned with the live analysis_* data plane.
ALTER TABLE public.analysis_sessions
  ADD COLUMN IF NOT EXISTS call_count integer NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS history jsonb NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE public.analysis_runs
  ADD COLUMN IF NOT EXISTS tool_id text,
  ADD COLUMN IF NOT EXISTS is_demo boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS duration_ms integer,
  ADD COLUMN IF NOT EXISTS error_tag text,
  ADD COLUMN IF NOT EXISTS is_cached boolean NOT NULL DEFAULT false;

ALTER TABLE public.analysis_assets
  ADD COLUMN IF NOT EXISTS storage_key text,
  ADD COLUMN IF NOT EXISTS sha256 text,
  ADD COLUMN IF NOT EXISTS size_bytes bigint,
  ADD COLUMN IF NOT EXISTS modality text,
  ADD COLUMN IF NOT EXISTS sensor_hint text;

CREATE TABLE IF NOT EXISTS public.analysis_cache (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  cache_key text NOT NULL UNIQUE,
  session_id uuid REFERENCES public.analysis_sessions(id) ON DELETE CASCADE,
  task_type text NOT NULL,
  model_id text,
  agent_version text,
  result_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  hit_count integer NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz NOT NULL DEFAULT (now() + interval '30 minutes')
);

CREATE INDEX IF NOT EXISTS idx_analysis_sessions_expires ON public.analysis_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_analysis_runs_session ON public.analysis_runs(session_id);
CREATE INDEX IF NOT EXISTS idx_analysis_assets_session ON public.analysis_assets(session_id);
CREATE INDEX IF NOT EXISTS idx_analysis_assets_sha ON public.analysis_assets(sha256);
CREATE INDEX IF NOT EXISTS idx_analysis_cache_lookup ON public.analysis_cache(cache_key, expires_at);
