# ORBITAL-AI Supabase data plane

Supabase is used for durable session/job/result state and metadata. Heavy PyTorch/Transformers inference remains in a replaceable GPU worker.

## Tables

- `analysis_sessions`: query/session lifecycle and 30-minute expiry.
- `analysis_runs`: model provenance, result, evidence and execution trace.
- `analysis_assets`: references to input/evidence objects stored in Supabase Storage.

## Security

All exposed tables have RLS enabled. Browser code must use only the Supabase publishable key. Never expose a service-role key.

## Setup

1. Create a Supabase project.
2. Apply `schema.sql`.
3. Create a private Storage bucket for analysis assets.
4. Set `VITE_SUPABASE_URL` and `VITE_SUPABASE_PUBLISHABLE_KEY`.
5. Keep worker/server credentials server-side only.

The application continues to work without Supabase; the data plane becomes active only when these environment variables are configured.
