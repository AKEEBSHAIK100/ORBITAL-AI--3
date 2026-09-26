import 'dotenv/config'
import cors from 'cors'
import express from 'express'
import OpenAI from 'openai'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { MODEL, MAX_TOKENS_ANALYZE, MAX_TOKENS_COMPARE, SESSION_CALL_LIMIT } from './lib/constants'
import {
  classifyTask, validateInputs, buildExecutionTrace,
  TOOL_REGISTRY, ExecutionTraceStep, FusionFeatures,
  checkToolAvailability, buildUnavailableResponse,
} from './lib/agentController'
// ─── Phase 4: PostgreSQL DB layer ────────────────────────────────────────────────
import { runMigrations } from './db/migrate.js'
import { dbPing, poolStats } from './db/pool.js'
import {
  upsertSession, appendSessionHistory, getSessionHistory,
  deleteSession, pruneExpiredSessions, createAnalysisRun,
  completeAnalysisRun, persistAnalysisResult, persistTraceSteps,
} from './db/repositories/sessions.js'
import { registerAsset, pruneExpiredAssets, computeImageSha256 } from './db/repositories/assets.js'
import { buildCacheKey, isCacheable, getCached, putCached, pruneExpiredCache } from './db/repositories/cache.js'
import { listDatasets, listModels, datasetStats, modelStats } from './db/repositories/catalog.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const app = express()
const port = Number(process.env.API_PORT ?? 8787)
const _rawFrontendOrigin = process.env.FRONTEND_ORIGIN ?? 'http://localhost:8443,https://localhost:8443'
const _allowedOrigins = _rawFrontendOrigin.split(',').map(o => o.trim()).filter(Boolean)

export const PYTHON_BACKEND_URL = (process.env.PYTHON_BACKEND_URL || 'http://127.0.0.1:8000')
  .replace(/\/api\/analyze.*$/, '')
  .replace(/\/+$/, '')

// ─── OpenAI/Anthropic-compatible client ───────────────────────────────────────
const client = new OpenAI({
  apiKey: process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_API_BASE || undefined,
})

// ─── Deployment-wide call counter ─────────────────────────────────────────────
let totalCallsThisDeployment = 0
function incrementCallCounter(): number {
  totalCallsThisDeployment += 1
  console.log(`[Orbital-AI] API call #${totalCallsThisDeployment} (deployment total)`)
  return totalCallsThisDeployment
}

// ─── Per-session image cache ──────────────────────────────────────────────────
// In-memory store remains as primary fast-path.
// When DB is available, image metadata (sha256, expiry) is also written to `assets`.
const IMAGE_TTL_MS = 30 * 60 * 1000
type CacheEntry = { dataUrl: string; expiresAt: number }
const imageCache = new Map<string, CacheEntry>()

function setCachedImage(sessionId: string, dataUrl: string): void {
  imageCache.set(sessionId, { dataUrl, expiresAt: Date.now() + IMAGE_TTL_MS })
  // Fire-and-forget metadata registration to DB (non-fatal)
  registerAsset(dataUrl, sessionId).catch(() => { /* DB unavailable — in-memory is sufficient */ })
}

function getCachedImage(sessionId: string): string | null {
  const entry = imageCache.get(sessionId)
  if (!entry) return null
  if (Date.now() > entry.expiresAt) { imageCache.delete(sessionId); return null }
  return entry.dataUrl
}

// ─── Per-IP rate limiter ───────────────────────────────────────────────────────
const requests = new Map<string, number[]>()
function checkRateLimit(ip: string) {
  // Never throttle localhost, unknown IP, or active demo sessions
  if (
    ip === '127.0.0.1' ||
    ip === '::1' ||
    ip.includes('127.0.0.1') ||
    ip.includes('localhost') ||
    ip === 'unknown' ||
    SESSION_CALL_LIMIT >= 99
  ) {
    return true
  }
  const now = Date.now()
  const recent = (requests.get(ip) ?? []).filter(t => now - t < 60_000)
  if (recent.length >= 60) return false
  recent.push(now)
  requests.set(ip, recent)
  return true
}

// ─── Error classification ────────────────────────────────────────────────     
type ClassifiedError = { httpStatus: number; userMessage: string; logTag: string }
function classifyError(error: unknown): ClassifiedError {
  const msg = error instanceof Error ? error.message : String(error)
  const status = (error as Record<string, unknown>)?.status as number | undefined

  if (status === 429 || msg.includes('rate limit') || msg.includes('429')) {
    console.error('[Orbital-AI][rate_limit]', msg)
    return { httpStatus: 429, userMessage: 'Analysis is temporarily unavailable — please try again in a moment.', logTag: 'rate_limit' }
  }
  if (status === 402 || msg.includes('credit') || msg.includes('billing') || msg.includes('quota') || msg.includes('overloaded')) {
    console.error('[Orbital-AI][billing]', msg)
    return { httpStatus: 402, userMessage: 'Analysis is temporarily unavailable — the API credit limit has been reached. Please try again later.', logTag: 'billing' }
  }
  if (status === 401 || msg.includes('auth') || msg.includes('API key') || msg.includes('credentials') || msg.includes('Missing credentials')) {
    console.error('[Orbital-AI][auth]', msg)
    return { httpStatus: 500, userMessage: 'The analysis service is misconfigured. Please contact support.', logTag: 'auth' }
  }
  if (status === 400 || msg.includes('malformed') || msg.includes('invalid')) {
    console.error('[Orbital-AI][malformed]', msg)
    return { httpStatus: 400, userMessage: 'The request could not be processed — please check your image and try again.', logTag: 'malformed' }
  }
  console.error('[Orbital-AI][unknown]', msg)
  return { httpStatus: 500, userMessage: 'Image analysis failed. Please try again.', logTag: 'unknown' }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function parseDataUrl(value: unknown) {
  if (typeof value !== 'string') throw new Error('Image data is required.')
  const match = value.match(/^data:(image\/(?:jpeg|png|webp|gif));base64,([A-Za-z0-9+/=]+)$/)
  if (!match) throw new Error('Image must be a compressed JPEG, PNG, WEBP, or GIF data URL.')
  return { mediaType: match[1], data: match[2], full: match[0] }
}

function cleanJson(text: string) {
  return JSON.parse(text.replace(/^```json\s*/i, '').replace(/^```\s*/i, '').replace(/\s*```$/i, '').trim())
}

function historyText(history: unknown) {
  if (!Array.isArray(history)) return 'None'
  return history.slice(-6).map(item => {
    const row = item as { question?: string; answer?: string }
    return `User: ${row.question ?? ''}\nAssistant: ${row.answer ?? ''}`
  }).join('\n\n') || 'None'
}

function imageContent(image: string) {
  return { type: 'image_url' as const, image_url: { url: image, detail: 'high' as const } }
}

// ─── System prompt ────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are SatQuery AI, a specialized agentic vision-language assistant for remote sensing imagery and Earth observation.
You operate with domain adaptation calibrated to the BigEarthNet 43-class Corine Land Cover taxonomy, RSVQA conventions, VRSBench scene captioning/grounding, and CDVQA multitemporal change detection.
Analyze the supplied satellite/aerial imagery with high scientific rigor.

Domain Adaptation & Reasoning Guidelines:
1. BigEarthNet Vocabulary: Map land-cover and surface objects to standardized BigEarthNet categories (Urban fabric, Industrial units, Arable land, Permanent crops, Pastures, Complex cultivation, Coniferous/Broad-leaved forest, Inland/Marine waters, Wetlands, Bare rock, Sparsely vegetated areas).
2. Spatial Grounding: When asked to locate, highlight, or pinpoint an entity (e.g. "Highlight the water body referred to in the query", "Find the building complex"), populate region with normalized bounding box percentages: { x_percent, y_percent, w_percent, h_percent } (0-100 relative to top-left).
3. Scene Captioning (VRSBench): When asked to describe or caption the scene, generate a structured, multi-attribute remote sensing description covering topography, dominant land cover, object distribution, and visible sensor characteristics.
4. Building Footprint Count: Visually inspect structural rooftop footprints visible in the image. If resolution allows direct enumeration, provide the exact count. If a high-density grid, provide a calibrated structural estimate based on rooftop footprint density per hectare. Always provide an explicit integer in "building_count".
5. Multitemporal Change (CDVQA): When comparing passes or analyzing changes, clearly state whether features increased, decreased, or remained unchanged, and localize where the change occurred.
6. Return valid JSON only with this shape:
{
  "answer": "2-4 complete plain-English sentences clearly answering the user question, evidence-grounded",
  "building_count": number|null,
  "confidence": "high|medium|low",
  "confidence_percent": number,
  "confidence_reason": "short reason",
  "detected_features": ["3-5 BigEarthNet vocabulary features"],
  "estimated_coverage_percent": number|null,
  "water_coverage_percent": number|null,
  "vegetation_percent": number|null,
  "data_limitation_note": "string|null",
  "region": {"x_percent": number,"y_percent": number,"w_percent": number,"h_percent": number}|null,
  "label": "2-4 word summary",
  "suggested_followups": ["3-5 image-specific questions"]
}`

// ─── Middleware ───────────────────────────────────────────────────────────────
app.use(cors({
  origin: (origin, callback) => {
    if (!origin) return callback(null, true)
    if (
      _allowedOrigins.includes('*') ||
      _allowedOrigins.includes(origin) ||
      origin.includes('localhost') ||
      origin.includes('127.0.0.1')
    ) {
      return callback(null, true)
    }
    return callback(new Error(`Origin ${origin} not allowed by CORS`))
  },
  credentials: true,
}))
app.use(express.json({ limit: '12mb' }))
if (process.env.NODE_ENV === 'production') {
  app.use(express.static('dist'))
}

// ─── Routes ───────────────────────────────────────────────────────────────────
app.get('/api/health', async (_req, res) => {
  const [dbResult, dsResult, mdResult] = await Promise.allSettled([
    dbPing(),
    datasetStats(),
    modelStats(),
  ])
  res.json({
    ok: true,
    provider: 'openai-compat',
    model: MODEL,
    configured: Boolean(process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY),
    totalCallsThisDeployment,
    sessionCallLimit: SESSION_CALL_LIMIT,
    database: {
      connected: dbResult.status === 'fulfilled' ? dbResult.value : false,
      ...poolStats(),
    },
    datasets: dsResult.status === 'fulfilled' ? dsResult.value : { total: 10, available: 0, not_downloaded: 10 },
    models: mdResult.status === 'fulfilled' ? mdResult.value : { total: 7, available: 0, unavailable: 7 },
  })
})

// ─── Phase 4: Session / history / catalog endpoints ───────────────────────────

/** GET /api/session/:id/history — returns persisted analysis history */
app.get('/api/session/:id/history', async (req, res) => {
  const sessionId = req.params['id']
  if (!sessionId) return res.status(400).json({ error: 'Session ID required.' })
  const history = await getSessionHistory(sessionId)
  return res.json({ session_id: sessionId, history, count: history.length })
})

/** DELETE /api/session/:id — deletes session + all cascaded DB data */
app.delete('/api/session/:id', async (req, res) => {
  const sessionId = req.params['id']
  if (!sessionId) return res.status(400).json({ error: 'Session ID required.' })
  const deleted = await deleteSession(sessionId)
  imageCache.delete(sessionId)
  return res.json({ ok: true, session_id: sessionId, deleted })
})

/** GET /api/catalog/datasets — dataset catalog with availability status */
app.get('/api/catalog/datasets', async (_req, res) => {
  const datasets = await listDatasets()
  return res.json({ datasets, total: datasets.length })
})

/** GET /api/catalog/models — model registry with availability */
app.get('/api/catalog/models', async (_req, res) => {
  const models = await listModels()
  return res.json({ models, total: models.length })
})

/** GET /api/agent/tools — tool registry with real-time availability flags */
app.get('/api/agent/tools', (_req, res) => {
  const tools = Object.values(TOOL_REGISTRY).map((t) => ({
    id: t.id,
    name: t.name,
    supported_tasks: t.supported_tasks,
    modalities: t.modalities,
    availability: t.availability,
    unavailable_reason: t.unavailable_reason ?? null,
    required_dataset: t.required_dataset ?? null,
    model_id: t.model_id,
  }))
  return res.json({ tools, total: tools.length })
})

app.get('/api/model-status', async (_req, res) => {
  try {
    const pyRes = await fetch(`${PYTHON_BACKEND_URL}/api/model-status`, { signal: AbortSignal.timeout(5000) })
    if (pyRes.ok) return res.json(await pyRes.json())
  } catch {}
  return res.json({
    specialists: Object.values(TOOL_REGISTRY).map(t => ({
      id: t.id, name: t.name, task: t.supported_tasks[0] || 'general',
      modality: t.modalities, is_available: t.availability === 'available',
      unavailable_reason: t.unavailable_reason ?? null,
    })),
    datasets: [], evaluations: [],
  })
})

// ─── BigEarthNet classification ──────────────────────────────────────────────
 // Classification is served by the verified Python specialist only. This Node
 // server deliberately does not estimate labels from encoded image bytes.
 app.post(['/classify', '/api/classify'], async (req, res) => {
   const { image, top_k = 5, threshold = 0.25 } = req.body ?? {}
   if (!image) return res.status(400).json({ error: 'An image is required.' })
   try {
     const pyRes = await fetch(`${PYTHON_BACKEND_URL}/classify`, {
       method: 'POST', headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({ image, top_k, threshold }),
       signal: AbortSignal.timeout(15_000),
     })
     if (pyRes.ok) return res.json(await pyRes.json())
   } catch {}
   return res.status(503).json({
     available: false, labels: [], active_labels: [], top_label: null,
     confidence: null, model_id: null,
     error: 'BigEarthNet classifier is unavailable in this deployment.',
     note: 'No heuristic classification is generated.',
   })
 })

app.post('/api/analyze', async (req, res) => {
  const startTime = Date.now()
  const { image, question, query } = req.body as Record<string, any>
  const prompt = String(query || question || '').trim()
  if (!prompt) return res.status(400).json({ error: 'A question is required.' })
  if (!image) return res.status(400).json({ error: 'An image is required for visual analysis.' })

  try {
    const pyRes = await fetch(`${PYTHON_BACKEND_URL}/api/analyze`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...req.body, query: prompt }),
      signal: AbortSignal.timeout(45_000),
    })
    if (pyRes.ok) {
      const data = await pyRes.json()
      if (data && data.answer && data.status !== 'SPECIALIST_UNAVAILABLE') {
        return res.json(data)
      }
    }
  } catch {}

  try {
    const worker = await runWorkerVqaOrCaption(image, prompt)
    return res.json({
      answer: worker.answer,
      confidence: null, confidence_percent: null, confidenceScore: null,
      confidence_source: 'none', confidence_status: 'not_calibrated',
      label: worker.task === 'caption' ? 'External Remote-Sensing Caption' : 'External Remote-Sensing VQA',
      mode: 'external_hf_zero_gpu', is_synthetic: false,
      provenance: worker.provenance,
      data_limitation_note: 'External public ZeroGPU specialist; not an ORBITAL-AI benchmark result.',
      execution_trace: buildExecutionTrace('vqa', [], Date.now()-startTime, validateInputs('vqa', 1, ['optical'], ['jpeg']), worker.task === 'caption' ? 'external_rs_caption' : 'external_rs_vqa'),
    })
  } catch (err) {
    console.warn('[Orbital-AI] Analysis specialists unavailable:', err)
    return res.status(503).json({
      error: 'Remote-sensing specialist is temporarily unavailable. No synthetic analysis is returned.',
      execution_trace: buildExecutionTrace('vqa', [], Date.now()-startTime, validateInputs('vqa', 1, ['optical'], ['jpeg']), 'rs_vqa'),
    })
  }
})

// ── Python FastAPI generic proxy helper ────────────────────────────────────────
async function proxyToPython(
  req: express.Request,
  res: express.Response,
  pythonPath: string,
  fallbackFn?: () => void
): Promise<void> {
  const targetUrl = `${PYTHON_BACKEND_URL}${pythonPath}`
  console.log(`[Orbital-AI] Proxying specialist request: ${req.method} ${targetUrl}`)
  try {
    const response = await fetch(targetUrl, {
      method: req.method,
      headers: { 'Content-Type': 'application/json' },
      body: req.method !== 'GET' ? JSON.stringify(req.body) : undefined,
      signal: AbortSignal.timeout(60_000),
    })
    console.log(`[Orbital-AI] Proxy response from ${targetUrl}: HTTP ${response.status}`)
    if (!response.ok) {
      const errText = await response.text().catch(() => '')
      console.warn(`[Orbital-AI] Proxy target ${targetUrl} returned HTTP ${response.status}: ${errText.slice(0, 300)}`)
    }
    const data = await response.json()
    res.status(response.status).json(data)
  } catch (err: any) {
    console.warn(`[Orbital-AI] Python backend not reachable at ${targetUrl}: ${err?.message || err}`)
    if (fallbackFn) fallbackFn()
    else res.status(502).json({ error: `Python backend unavailable at ${targetUrl}: ${err?.message || 'Connection failed'}` })
  }
}

app.all(['/analyze/buildings', '/api/analyze/buildings', '/api/buildings'], async (req, res) => {
  await proxyToPython(req, res, '/api/analyze/buildings', () => {
    try {
      const p = path.join(__dirname, 'backend', 'data', 'default_detections.json')
      if (fs.existsSync(p)) {
        res.json(JSON.parse(fs.readFileSync(p, 'utf8')))
        return
      }
    } catch { /* pass */ }
    res.status(502).json({ error: 'Building detection service unavailable.' })
  })
})

// New unified analysis sub-endpoints — proxy to FastAPI first
app.post(['/api/analyze/vqa'], async (req, res) => {
  await proxyToPython(req, res, '/api/analyze/vqa')
})
app.post(['/api/analyze/grounding'], async (req, res) => {
  await proxyToPython(req, res, '/api/analyze/grounding')
})
app.post(['/api/analyze/change', '/api/compare/change'], async (req, res) => {
  // Transform /api/compare shape to /api/analyze schema if needed
  const body = req.body as Record<string, unknown>
  if (body.beforeImage || body.afterImage) {
    req.body = {
      query: body.question || 'What changed between these two satellite passes?',
      image: body.beforeImage,
      secondary_image: body.afterImage,
      task_type: 'change_detection',
      modality: 'optical',
      secondary_modality: 'optical',
    }
  }
  await proxyToPython(req, res, '/api/analyze/change')
})
app.post(['/api/analyze/optical-sar', '/api/fuse/direct'], async (req, res) => {
  const body = req.body as Record<string, unknown>
  if (body.opticalImage || body.sarImage) {
    req.body = {
      query: body.question || 'Identify built-up and water-covered regions using joint optical and SAR data',
      image: body.opticalImage,
      secondary_image: body.sarImage,
      task_type: 'sar_optical_fusion',
      modality: 'optical',
      secondary_modality: 'sar',
    }
  }
  await proxyToPython(req, res, '/api/analyze/optical-sar')
})
app.get(['/api/tools'], async (req, res) => {
  await proxyToPython(req, res, '/api/tools')
})
app.get(['/api/models'], async (req, res) => {
  await proxyToPython(req, res, '/api/models')
})


const OPTICAL_SAR_SYSTEM_PROMPT = `${SYSTEM_PROMPT}
You are operating in Optical–SAR Multi-Modal Fusion mode.
Analyze the two co-registered remote-sensing views:
1. OPTICAL VIEW (surface albedo, vegetation chlorophyll, spectral reflectance).
2. SAR VIEW (microwave backscatter intensity, surface roughness, dielectric properties, structural double-bounce).
Evaluate the scene combining both modalities and cross-reference features.`

app.post('/api/fuse', async (req, res) => {
  const startTime = Date.now()
  const traceSteps: ExecutionTraceStep[] = []
  try {
    const { opticalImage, sarImage, question = 'Explain what the optical and SAR observations show together.' } =
      req.body as Record<string, string | undefined>
    if (!opticalImage || !sarImage) return res.status(400).json({ error: 'Both optical and SAR images are required.' })

    const taskType = classifyTask(question, 2, ['optical', 'sar'])
    const validation = validateInputs(taskType, 2, ['optical', 'sar'], ['jpeg', 'png'])
    traceSteps.push({
      step: 1, tool: 'rs_task_classifier',
      description: 'Rule-based query routing and modality validation',
      input_summary: question.slice(0, 120),
      output_summary: `Task classified as: ${taskType}`,
      duration_ms: 1, status: 'success', parameters: { task_type: taskType },
    })

    // Prefer the configured Python specialist.
    try {
      const pyRes = await fetch(`${PYTHON_BACKEND_URL}/analyze/fusion`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ optical_image: opticalImage, sar_image: sarImage, query: question }),
        signal: AbortSignal.timeout(15_000),
      })
      if (pyRes.ok) {
        const data = await pyRes.json()
        if (data && (data.answer || data.fusion_features)) {
          traceSteps.push({ step: 2, tool: 'rs_optical_sar_specialist', description: 'Configured Python remote-sensing fusion specialist', input_summary: 'Optical + SAR', output_summary: 'Specialist returned a result', duration_ms: 1, status: 'success' })
          return res.json({ ...data, execution_trace: buildExecutionTrace(taskType, traceSteps, Date.now()-startTime, validation, 'rs_optical_sar_specialist') })
        }
      }
    } catch {}

    // Free external specialist fallback. It must not fabricate calibration/registration.
    try {
      const worker = await runWorkerFusion(opticalImage, sarImage)
      traceSteps.push({ step: 2, tool: 'external_rs_fusion', description: 'External public Hugging Face ZeroGPU remote-sensing specialist', input_summary: 'Optical + SAR', output_summary: 'External specialist returned a semantic result', duration_ms: 1, status: 'success', parameters: { confidence_status: 'not_calibrated' } })
      return res.json({
        answer: worker.answer,
        confidence: null,
        confidence_percent: null,
        confidence_source: 'none',
        confidence_status: 'not_calibrated',
        label: 'External Remote-Sensing Optical + SAR Analysis',
        mode: 'external_hf_zero_gpu',
        provenance: worker.method,
        data_limitation_note: worker.note,
        execution_trace: buildExecutionTrace(taskType, traceSteps, Date.now()-startTime, validation, 'external_rs_fusion'),
      })
    } catch (err) {
      console.warn('[Orbital-AI] Fusion specialists unavailable:', err)
      return res.status(503).json({
        error: 'Optical + SAR specialist is temporarily unavailable. No synthetic telemetry or confidence is returned.',
        execution_trace: buildExecutionTrace(taskType, traceSteps, Date.now()-startTime, validation, 'rs_optical_sar_specialist'),
      })
    }
  } catch (err) {
    return res.status(500).json({ error: 'Fusion request could not be processed.' })
  }
})

app.post('/api/compare', async (req, res) => {
  const startTime = Date.now()
  const { beforeImage, afterImage, question = 'What changed between these two observations?' } = req.body as Record<string, any>
  if (!beforeImage || !afterImage) return res.status(400).json({ error: 'Two images are required for change analysis.' })
  try {
    const pyRes = await fetch(`${PYTHON_BACKEND_URL}/api/compare`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...req.body, question }),
      signal: AbortSignal.timeout(45_000),
    })
    if (pyRes.ok) {
      const data = await pyRes.json()
      if (data && data.answer && data.status !== 'SPECIALIST_UNAVAILABLE') return res.json(data)
    }
  } catch {}
  try {
    const worker = await runWorkerChange(beforeImage, afterImage)
    return res.json({
      answer: worker.answer,
      confidence: null, confidence_percent: null, confidence_source: 'none',
      confidence_status: 'not_calibrated',
      label: 'External Remote-Sensing Change Analysis',
      mode: 'external_hf_zero_gpu', provenance: worker.method,
      data_limitation_note: worker.note,
      execution_trace: buildExecutionTrace('change_detection', [], Date.now()-startTime, validateInputs('change_detection', 2, ['optical'], ['jpeg']), 'external_rs_change'),
    })
  } catch (err) {
    console.warn('[Orbital-AI] Change specialists unavailable:', err)
    return res.status(503).json({ error: 'Change specialist is temporarily unavailable. No synthetic change result is returned.' })
  }
})

if (process.env.NODE_ENV === 'production' || fs.existsSync(path.join(__dirname, 'dist', 'index.html'))) {
  app.get('*splat', (req, res, next) => {
    if (req.path.startsWith('/api') || req.path.startsWith('/analyze') || req.path.startsWith('/classify')) {
      return next()
    }
    const indexPath = path.join(__dirname, 'dist', 'index.html')
    if (fs.existsSync(indexPath)) {
      return res.sendFile(indexPath)
    }
    next()
  })
}

// ─── Startup ──────────────────────────────────────────────────────────────────
async function start() {
  // Run DB migrations (non-fatal — server starts even if DB is unavailable)
  await runMigrations()

  app.listen(port, () => {
    console.log(`Orbital-AI API listening on http://localhost:${port} · model: ${MODEL} · session limit: ${SESSION_CALL_LIMIT}`)
  })

  // Schedule DB housekeeping every hour (prune expired sessions/assets/cache)
  setInterval(async () => {
    await Promise.allSettled([
      pruneExpiredSessions(),
      pruneExpiredAssets(),
      pruneExpiredCache(),
    ])
  }, 60 * 60 * 1000).unref()
}

start().catch((err) => {
  console.error('[Orbital-AI] Fatal startup error:', err)
  process.exit(1)
})
