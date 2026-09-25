-- ORBITAL-AI Supabase data/control plane
-- Apply through Supabase SQL editor/migration tooling.
create extension if not exists pgcrypto;

create table if not exists public.analysis_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade,
  status text not null default 'created' check (status in ('created','queued','running','completed','failed')),
  query text,
  modality text,
  task_type text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  expires_at timestamptz not null default (now() + interval '30 minutes')
);

create table if not exists public.analysis_runs (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.analysis_sessions(id) on delete cascade,
  status text not null default 'queued' check (status in ('queued','running','completed','failed','unavailable')),
  task_type text,
  model_name text,
  model_adaptation text,
  input_metadata jsonb not null default '{}'::jsonb,
  result jsonb not null default '{}'::jsonb,
  evidence jsonb not null default '[]'::jsonb,
  execution_trace jsonb not null default '[]'::jsonb,
  confidence numeric,
  confidence_status text not null default 'not_calibrated',
  error_message text,
  created_at timestamptz not null default now(),
  completed_at timestamptz
);

create table if not exists public.analysis_assets (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.analysis_sessions(id) on delete cascade,
  storage_path text not null,
  asset_role text not null check (asset_role in ('input','temporal_before','temporal_after','optical','sar','evidence')),
  mime_type text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  expires_at timestamptz not null default (now() + interval '30 minutes')
);

create index if not exists analysis_sessions_user_created_idx
  on public.analysis_sessions(user_id, created_at desc);
create index if not exists analysis_runs_session_created_idx
  on public.analysis_runs(session_id, created_at desc);
create index if not exists analysis_assets_session_idx
  on public.analysis_assets(session_id);

alter table public.analysis_sessions enable row level security;
alter table public.analysis_runs enable row level security;
alter table public.analysis_assets enable row level security;

drop policy if exists "users manage own sessions" on public.analysis_sessions;
create policy "users manage own sessions" on public.analysis_sessions
  for all to authenticated
  using ((select auth.uid()) = user_id)
  with check ((select auth.uid()) = user_id);

drop policy if exists "users read own runs" on public.analysis_runs;
create policy "users read own runs" on public.analysis_runs
  for select to authenticated
  using (exists (
    select 1 from public.analysis_sessions s
    where s.id = analysis_runs.session_id
      and s.user_id = (select auth.uid())
  ));

drop policy if exists "users read own assets" on public.analysis_assets;
create policy "users read own assets" on public.analysis_assets
  for select to authenticated
  using (exists (
    select 1 from public.analysis_sessions s
    where s.id = analysis_assets.session_id
      and s.user_id = (select auth.uid())
  ));

-- Service-side writes should use a server/worker credential, never a service-role key in the browser.
