import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getCachedImage, incrementCallCounter, parseDataUrl, setCachedImage } from './_lib.js'
import { runWorkerVqaOrCaption } from './_hfWorker.js'
import {
  classifyTask, validateInputs, buildExecutionTrace, type ExecutionTraceStep,
} from '../lib/agentController.js'

export const config = { api: { bodyParser: { sizeLimit: '12mb' }, maxDuration: 60 } }

// ─── Counting-question intent detection ───────────────────────────────────────
const COUNT_KEYWORDS = [
  'how many', 'count', 'number of', 'total number', 'quantify',
  'tally', 'enumerate', 'buildings', 'structures', 'houses',
  'vehicles', 'cars', 'trucks', 'trees', 'fields', 'parcels', 'ponds',
  'water bodies', 'rooftops', 'roofs', 'units', 'objects',
]

function isCountingQuestion(text: string): boolean {
  const lower = text.toLowerCase()
  return COUNT_KEYWORDS.some(kw => lower.includes(kw))
}

// ─── Median helper ────────────────────────────────────────────────────────────
function median(values: number[]): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : Math.round((sorted[mid - 1] + sorted[mid]) / 2)
}

// ─── Deduplicate string array (case-insensitive) ──────────────────────────────
function dedupeStrings(arr: string[]): string[] {
  const seen = new Set<string>()
  return arr.filter(s => {
    const key = s.toLowerCase().trim()
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

function buildUnavailableAnalysis(taskType: string) {
      const taskLabel = taskType === 'land_cover'
        ? 'land-cover classification'
        : taskType === 'vegetation_analysis'
          ? 'vegetation analysis'
          : taskType === 'caption'
            ? 'scene captioning'
            : taskType === 'grounding'
              ? 'spatial grounding'
              : taskType === 'building_detection'
                ? 'building detection'
                : 'remote-sensing analysis'
      return {
        answer: `The requested ${taskLabel} is unavailable in this serverless deployment because no executable specialist or configured VLM provider is available. No unsupported spectral measurements, percentages, or confidence values are being estimated.`,
        confidence: null,
        confidence_percent: null,
        confidenceScore: null,
        confidence_source: 'none' as const,
        confidence_reason: 'Analysis was not executed by an available specialist model or configured VLM provider.',
        data_limitation_note: 'RGB/JPEG imagery does not provide multispectral or SAR measurements unless those bands and a corresponding computation are available.',
        detected_features: [],
        estimated_coverage_percent: null,
        water_coverage_percent: null,
        vegetation_percent: null,
        count_estimate: null,
        count_uncertainty_factors: [],
        region: null,
        label: 'Analysis unavailable',
        suggested_followups: [
          'Configure the production VLM provider and redeploy.',
          'Use the adapted remote-sensing specialist through the Python backend.',
          'Upload a supported multispectral or SAR product for sensor-specific analysis.',
        ],
      }
    }
export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed.' })

  const startTime = Date.now()
  const traceSteps: ExecutionTraceStep[] = []

  try {
    const { image, question, query, history, sessionId } = req.body as {
      image?: string
      question?: string
      query?: string
      history?: unknown
      sessionId?: string
    }

    const rawPrompt = (query || question || '').trim()
    if (!rawPrompt) return res.status(400).json({ error: 'A question is required.' })

    const promptText = rawPrompt

    // Try FastAPI master analysis first — ONLY if explicitly configured via env var.
    // When PYTHON_BACKEND_URL is absent (e.g. on Vercel standalone), skip entirely so
    // the Node/VLM path below runs immediately without wasting connection budget.
    const configuredPyUrl = process.env.PYTHON_BACKEND_URL?.trim()
    if (configuredPyUrl) {
      try {
        const pyBase = configuredPyUrl
          .replace(/\/api\/analyze\/?$/, '')
          .replace(/\/api\/?$/, '')
          .replace(/\/+$/, '')
        const pyRes = await fetch(`${pyBase}/api/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: promptText,
            image: image || undefined,
            task_type: (req.body as Record<string, unknown>).task_type,
          }),
          // 10 s: leaves headroom for VLM fallback within Vercel's 30 s budget
          signal: AbortSignal.timeout(45_000),
        })
        if (pyRes.ok) {
          const pyData = (await pyRes.json()) as Record<string, any>
          if (pyData && (pyData.answer || pyData.building_analysis)) {
            return res.status(200).json(pyData)
          }
        } else {
          console.warn(`[Orbital-AI] Python backend returned HTTP ${pyRes.status}, falling through to built-in analysis engine`)
        }
      } catch (err: any) {
        console.warn(`[Orbital-AI] Python backend connection error (${err?.message || 'offline'}), falling through to built-in analysis engine`)
      }
    }

    // Step 1: Deterministic task classification
    const step1Start = Date.now()
    const taskType = classifyTask(promptText, 1, ['optical'])
    traceSteps.push({
      step: 1,
      tool: 'rs_task_classifier',
      description: 'Deterministic rule-based task routing and intent extraction',
      input_summary: `Query: "${promptText.slice(0, 70)}"`,
      output_summary: `Task classified as: "${taskType}"`,
      duration_ms: Math.max(1, Date.now() - step1Start),
      status: 'success',
      parameters: { task_type: taskType },
    })

    // Step 2: Input validation
    const step2Start = Date.now()
    const validation = validateInputs(taskType, 1, ['optical'], ['jpeg'])
    traceSteps.push({
      step: 2,
      tool: 'rs_input_validator',
      description: 'Radiometric and spatial resolution verification',
      input_summary: 'Single-scene optical observation',
      output_summary: validation.notes.join('; '),
      duration_ms: Math.max(1, Date.now() - step2Start),
      status: 'success',
      parameters: { compatibility: validation.compatibility },
    })

    const apiKey = process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY || ''
    const isPlaceholderKey = !apiKey || apiKey === 'sk-your-key-here' || apiKey.includes('your-key')



    // Vercel serverless does not have the trained BigEarthNet classifier.
    // Never synthesize a land-cover label from encoded JPEG bytes: that is not
    // a valid image-analysis signal. Use the configured VLM or Python specialist,
    // otherwise return an honest unavailable result.

        // Image resolution: always use client's image if provided, or retrieve cached image
    let resolvedImage: string | null = null
    if (image) {
      try {
        const safeImage = parseDataUrl(image)
        if (sessionId) setCachedImage(sessionId, safeImage)
        resolvedImage = safeImage
      } catch {
        resolvedImage = null
      }
    } else if (sessionId) {
      resolvedImage = getCachedImage(sessionId)
    }

    const effectiveQuestion = (question || promptText).trim()

    // Free Hugging Face ZeroGPU is the no-cost external RS specialist fallback.
    if (resolvedImage && process.env.ENABLE_HF_RS_WORKER !== 'false') {
      try {
        const workerResult = await runWorkerVqaOrCaption(resolvedImage, effectiveQuestion)
        if (workerResult && typeof workerResult === 'object' && (workerResult as any).ok) {
          const worker = workerResult as Record<string, any>
          traceSteps.push({
            step: 3,
            tool: worker.task === 'caption' ? 'external_rs_caption' : 'external_rs_vqa',
            description: 'External public Hugging Face ZeroGPU remote-sensing specialist',
            input_summary: 'Single optical observation',
            output_summary: 'External specialist returned a result; provenance kept explicit',
            duration_ms: Math.max(1, Number(worker.duration_ms) || Date.now() - step2Start),
            status: 'success',
            confidence_source: 'none',
            parameters: { model: worker.model, adapter: worker.adapter, external_dependency: true, confidence_status: 'not_calibrated' },
          })
          const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, worker.task === 'caption' ? 'external_rs_caption' : 'external_rs_vqa')
          return res.status(200).json({
            answer: String(worker.answer || ''),
            confidence: null,
            confidence_percent: null,
            confidenceScore: null,
            confidence_source: 'none',
            confidence_status: 'not_calibrated',
            confidence_reason: 'External specialist executed, but no calibrated confidence metric is available.',
            detected_features: [],
            suggested_followups: [],
            label: worker.task === 'caption' ? 'External Remote-Sensing Caption' : 'External Remote-Sensing VQA',
            mode: 'external_hf_zero_gpu',
            is_synthetic: false,
            provenance: worker.provenance,
            data_limitation_note: 'This is an external public ZeroGPU specialist fallback, not an ORBITAL-AI benchmark result.',
            execution_trace: trace,
          })
        }
      } catch (workerError: any) {
        console.warn('[Orbital-AI] External HF ZeroGPU worker unavailable:', workerError?.message || workerError)
      }
    }

    // No paid-provider fallback is permitted. The free execution path is
    // Python remote-sensing specialists -> public HF ZeroGPU specialist -> honest unavailable.
    incrementCallCounter()
    const analysis = buildUnavailableAnalysis(taskType)
    traceSteps.push({
      step: 3,
      tool: 'rs_provider_guard',
      description: 'Free remote-sensing specialist availability guard',
      input_summary: resolvedImage ? 'Single optical observation' : 'No usable image',
      output_summary: 'No executable free specialist returned a result; no fabricated analysis generated',
      duration_ms: Math.max(1, Date.now() - step2Start),
      status: 'unavailable',
      confidence_source: 'none',
      parameters: { python_backend: Boolean(configuredPyUrl), hf_zero_gpu: process.env.ENABLE_HF_RS_WORKER !== 'false' },
    })
    const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'rs_vqa')
    return res.status(200).json({ ...analysis, execution_trace: trace })

  } catch {
    const fallbackTrace = buildExecutionTrace('vqa', traceSteps, Date.now() - startTime, validateInputs('vqa', 1), 'rs_vqa')
    return res.status(200).json({
      ...buildUnavailableAnalysis('vqa'),
      execution_trace: fallbackTrace,
    })
  }
}
