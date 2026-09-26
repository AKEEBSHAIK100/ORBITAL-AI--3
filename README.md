# ORBITAL-AI — Remote-Sensing Query Assistant

ORBITAL-AI is a query-driven remote-sensing prototype for asking natural-language questions about satellite and aerial imagery.

The project is designed around **evidence-grounded analysis**: when an executable specialist is unavailable, ORBITAL-AI reports that limitation instead of generating a synthetic result.

## Current execution paths

### Single-image VQA / scene description
The server can use the configured Python remote-sensing backend when available. It also supports an external public Hugging Face ZeroGPU remote-sensing specialist as a free fallback.

External worker provenance is reported explicitly and is **not** treated as an ORBITAL-AI benchmark result.

### Bi-temporal change understanding
Two supplied observations can be sent to the Python change pipeline or the external ZeroGPU specialist. The system does not invent change percentages, alignment confidence, dates, or geographic regions.

### Optical + SAR
The application accepts an optical image and a SAR image and can route them to the available fusion specialist. It does **not** assume that two uploaded files are co-registered. Sensor metadata and registration evidence are required before making sensor-specific physical claims.

### Other specialists
The repository contains pilot/integration code for BigEarthNet-adapted VQA/captioning, building detection, grounding, change detection, and optical-SAR processing. These are marked unavailable unless their executable runtime and weights are verified.

## Honesty and provenance rules

- No synthetic fallback results are used for production/judging paths.
- Confidence is null unless a calibrated specialist explicitly supplies it.
- RGB imagery is not treated as multispectral imagery.
- RGB imagery is not used to fabricate NDVI.
- Raw SAR brightness is not presented as calibrated dB backscatter.
- Co-registration is not claimed merely because two files were uploaded.
- Unsupported building/object counts are not estimated.
- External ZeroGPU results are labelled as external and unbenchmarked by ORBITAL-AI.

## Data layer

Phase 4 uses PostgreSQL/Supabase-compatible tables for sessions, analysis runs, uploaded asset metadata, execution traces, analysis cache, and session history.

The application uses a **30-minute cache/session image TTL**. Expired records are pruned by the data-plane repository layer.

## Architecture

```
src/
  App.tsx
  components/
  lib/agentController.ts
  pages/PrivacyPage.tsx
  pages/TermsPage.tsx
  utils/

api/
  _lib.ts
  _hfWorker.ts
  analyze.ts
  compare.ts
  fuse.ts
  buildings.ts
  health.ts

backend/
  main.py
  agents/
  routers/
  services/
  tools/

db/
  migrations/
  repositories/

server.ts
scripts/dev-full.mjs
```

The deterministic controller performs task classification, input validation, specialist routing, and execution-trace construction.

## Local development

```bash
cp .env.example .env
npm install
npm run dev:full
```

Frontend: `http://localhost:8443`  
API: `http://localhost:8787`

Useful checks:

```bash
npm run typecheck
npm run build
npm run test:phase4
```

## Environment

- `PYTHON_BACKEND_URL` — optional executable FastAPI remote-sensing backend.
- `ENABLE_HF_RS_WORKER` — enables the external free ZeroGPU specialist.
- `HF_RS_WORKER_URL` — optional external specialist URL override.
- `DATABASE_URL` — optional PostgreSQL connection string.
- `SESSION_TTL_MINUTES` / `CACHE_TTL_MINUTES` — default to 30 minutes.
- `SESSION_CALL_LIMIT` — server-side session guard.

No production or judging workflow should enable synthetic/mock analysis.

## Legal pages

Real routes are provided at `/privacy` and `/terms`. They describe the actual session-image/cache behavior and third-party inference dependencies.

## Project status

This is an **SIH prototype/integration project**. Pilot adapters and documented model integrations are not represented as production-ready or benchmark-validated unless their executable runtime has been verified.

External inference availability can change independently of this repository.
