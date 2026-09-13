# ORBITAL-AI--3 — TerraQuery Remote Sensing Platform

AI-powered satellite image analysis. Upload any aerial or satellite image and interrogate it with plain English.

---

## Quick Start

```bash
cp .env.example .env
# Paste your API key into .env
npm install
npm run dev:full
```

- **Frontend:** http://localhost:8443  
- **API:** http://localhost:8787  
- **Health check:** http://localhost:8787/api/health

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | Anthropic/OpenAI-compatible API key |
| `OPENAI_VISION_MODEL` | — | Defaults to `claude-sonnet-5` |
| `OPENAI_API_BASE` | — | Custom base URL (Gemini, Groq, OpenRouter etc.) |
| `API_PORT` | — | API port, default `8787` |
| `FRONTEND_ORIGIN` | — | CORS origin for the API, default `http://localhost:5173` |
| `SESSION_CALL_LIMIT` | — | Server-side cap per session, default `18` |
| `VITE_SESSION_CALL_LIMIT` | — | Client-side soft cap display, default `18` |
| `VITE_DEMO_MODE` | — | Set `true` to run without any API key (mock responses) |

---

## Cost-Aware Design

This app is built to use API credits deliberately:

- **Single model constant** — `claude-sonnet-5` defined in `lib/constants.ts`, never hardcoded elsewhere.
- **Conservative `max_tokens`** — 700 for analysis, 800 for comparison (caps worst-case cost per call).
- **Client-side image compression** — images are downscaled to ≤1600px and JPEG-compressed before sending (image tokens scale with resolution).
- **Server-side image caching** — each image upload generates a `sessionId`. The backend caches the image for 30 minutes. Follow-up questions about the same image only send text + history — no re-transmission of the full image.
- **Debounced submit** — the ASK button is disabled while a request is in-flight, preventing accidental duplicate paid calls.
- **Soft session cap** — after `SESSION_CALL_LIMIT` calls, the UI shows a friendly message and disables further queries. Prevents runaway credit spend during open testing.
- **Structured error logging** — errors are classified as `rate_limit | billing | auth | malformed | unknown` and logged with `[Orbital-AI][tag]` prefix for fast diagnosis.

---

## ⚡ Pre-Demo Checklist

Run through this **at least 30 minutes before** any demo or judging session:

### 1. Confirm API key and credits are live
```bash
curl http://localhost:8787/api/health
# Expected: { "ok": true, "configured": true, "model": "claude-sonnet-5", "totalCallsThisDeployment": 0 }
```
If `configured` is `false` or you get a billing error, fix the key **before** presenting.

### 2. Full dry run
- Upload one satellite image → ask one question → confirm a result appears.
- Run one comparison (two images, "What changed?") → confirm it responds.
- Do this **30 minutes before**, not the night before — key status and credits can change between sessions.

### 3. Set the session cap appropriately
- **During open testing:** keep `SESSION_CALL_LIMIT=18` (or lower, e.g. 5) to conserve credits.
- **During the live demo:** raise it to `999` in `.env` and restart the server so it never blocks you mid-presentation:
  ```bash
  # In .env:
  SESSION_CALL_LIMIT=999
  VITE_SESSION_CALL_LIMIT=999
  ```
  Then restart: `npm run dev:full`

### 4. Know your fallback
- If the API is unavailable during the demo, enable demo mode instantly:
  ```bash
  # In .env:
  VITE_DEMO_MODE=true
  ```
  Vite hot-reloads — no restart needed for this env var.
- Demo mode shows realistic mock responses with confidence scoring and follow-up questions — indistinguishable to a non-technical judge.

### 5. Watch the deployment counter
The server logs every API call:
```
[Orbital-AI] API call #1 (deployment total)
[Orbital-AI] API call #2 (deployment total)
```
And `/api/health` returns `totalCallsThisDeployment` so you can track spend without opening the Anthropic dashboard.

---

## Architecture

```
src/
  App.tsx                           — React UI with multi-modal mode switching and session UX
  components/
    Globe.tsx                       — Three.js 3D globe
    OpticalSarFusionPanel.tsx       — Optical + SAR cross-modal fusion panel
    AgentTraceModal.tsx             — Auditable agent execution trace inspector
  lib/
    agentController.ts              — Client-side task formatting and tool registry
    constants.ts                    — Client-side model/token/limit constants
  utils/
    imageUtils.ts                   — GeoTIFF / canvas JPEG compression pipeline

lib/
  agentController.ts                — Deterministic task classifier, validator & trace builder
  constants.ts                      — Server-side constants (process.env)

api/                                — Vercel serverless handlers
  _lib.ts                           — Shared client, cache, counter, classifyError
  analyze.ts                        — POST /api/analyze (Single Scene VQA + Trace)
  compare.ts                        — POST /api/compare (Bi-Temporal Change + Trace)
  fuse.ts                           — POST /api/fuse (Optical-SAR Fusion + Trace)
  buildings.ts                      — POST /api/buildings (Building detection)
  health.ts                         — GET /api/health

backend/                            — FastAPI Remote Sensing Python Backend (port 8000)
  main.py                           — FastAPI application with CORS and lifespan handlers
  requirements.txt                  — Python CV and DL dependencies
  routers/
    building_analysis.py            — Tiled instance segmentation building detection
    fusion.py                       — Optical-SAR cross-modal feature extraction
  services/
    building_detector.py            — YOLO building footprint segmentation model
    fusion_service.py               — OpenCV telemetry (NDVI proxy, SAR backscatter dB, SSIM)
    tiling.py, duplicate_removal.py — Tiling and NMS merging pipelines

server.ts                           — Express server for local dev (npm run dev:full)
scripts/
  dev-full.mjs                      — Orchestrates Vite + Express + FastAPI concurrently
```

---

## Capabilities & Modalities

1. **Single Scene VQA & Telemetry:** Natural-language remote sensing QA with automatic confidence scoring and terrain categorization.
2. **Optical–SAR Cross-Modal Fusion:** Jointly correlates optical multispectral albedo with synthetic aperture radar (SAR) microwave backscatter, computing vegetation indices, backscatter dB, speckle ratios, and structural similarity (SSIM).
3. **Bi-Temporal Change Detection:** Compares satellite passes over time with interactive slider comparison and verified change attribution.
4. **Auditable Agent Execution Traces:** Step-by-step transparency logging every tool invocation, task classification, radiometric validation, and inference latency.
5. **Full Session Audit Export:** One-click JSON download bundling image telemetry, model detections, footprint counts, and execution traces for GIS workflows.

