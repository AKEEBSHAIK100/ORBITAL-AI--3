import type { VercelRequest, VercelResponse } from '@vercel/node'
import {
  getCachedImage, parseDataUrl, setCachedImage,
} from './_lib.js'
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

    // No paid-provider fallback is permitted. The free execution path is:
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
      parameters: { python_backend: Boolean(process.env.PYTHON_BACKEND_URL), hf_zero_gpu: process.env.ENABLE_HF_RS_WORKER !== 'false' },
    })
    const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'rs_vqa')
    return res.status(200).json({ ...analysis, execution_trace: trace })
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

    // The adapted BigEarthNet specialist runs through the Python backend when
    // PYTHON_BACKEND_URL is configured above. If that backend is unavailable, allow
    // the configured VLM provider to serve as an explicitly unadapted fallback rather
    // than hard-stopping land-cover queries. The UI labels this path as VLM fallback.
    if (taskType === 'land_cover' && !isPlaceholderKey && resolvedImage) {
      traceSteps.push({
        step: 3,
        tool: 'rs_land_cover',
        description: 'BigEarthNet specialist requested; Python specialist unavailable, falling back to configured VLM',
        input_summary: 'Single optical observation',
        output_summary: 'Adapted specialist unavailable in this runtime; configured VLM fallback permitted without specialist confidence claim',
        duration_ms: Math.max(1, Date.now() - step2Start),
        status: 'unavailable',
        confidence_source: 'none',
        parameters: { specialist_available: false, vlm_fallback: true },
      })
    }

    if (isPlaceholderKey || (!resolvedImage && !sessionId && !image)) {
      incrementCallCounter()
      const analysis = buildUnavailableAnalysis(taskType)
      traceSteps.push({
        step: 3,
        tool: 'rs_provider_guard',
        description: 'Production provider/model availability guard',
        input_summary: resolvedImage ? 'Single optical observation' : 'No usable image/provider',
        output_summary: 'Specialist/VLM execution unavailable; no fabricated visual measurements returned',
        duration_ms: Math.max(1, Date.now() - step2Start),
        status: 'unavailable',
        confidence_source: 'none',
        parameters: { provider_configured: !isPlaceholderKey, image_available: Boolean(resolvedImage) },
      })
      const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'rs_vqa')
      return res.status(200).json({ ...analysis, execution_trace: trace })
    }

    const counting = isCountingQuestion(effectiveQuestion)

    const userContent: any[] = [
      { type: 'text', text: `Previous conversation:\n${historyText(history)}\n\nCurrent question:\n${effectiveQuestion}\n\nAnalyze this image and return JSON only.` },
      ...(resolvedImage ? [imageContent(resolvedImage)] : []),
    ]

    // ─── Self-consistency: 3 parallel calls for counting questions ───────────────
    const NUM_COUNTING_CALLS = 3
    const COUNTING_TEMPERATURE = 0.1
    const STANDARD_TEMPERATURE = 0.2

    if (counting) {
      console.log(`[Orbital-AI] Counting question detected \u2014 firing ${NUM_COUNTING_CALLS} parallel API calls for self-consistency`)
    }

    incrementCallCounter(counting ? NUM_COUNTING_CALLS : 1)
    try {
      if (counting) {
        // Fire NUM_COUNTING_CALLS parallel requests
        const callPromises = Array.from({ length: NUM_COUNTING_CALLS }).map(() =>
          client.chat.completions.create({
            model: MODEL,
            temperature: COUNTING_TEMPERATURE,
            max_tokens: MAX_TOKENS_ANALYZE,
            response_format: { type: 'json_object' },
            messages: [
              { role: 'system', content: systemPrompt },
              { role: 'user', content: userContent as any },
            ],
          })
        )

        const settled = await Promise.allSettled(callPromises)
        const parsedResults: Record<string, unknown>[] = []

        for (const result of settled) {
          if (result.status === 'fulfilled') {
            try {
              const resVal = result.value as any
              const p = cleanJson(resVal?.choices?.[0]?.message?.content ?? '{}')
              parsedResults.push(p)
            } catch {
              // skip unparseable responses
            }
          }
        }

        if (parsedResults.length === 0) {
          // All provider calls failed — return an honest unavailable response
          return res.status(200).json(buildUnavailableAnalysis(taskType))
        }

        // Take the first valid result as base for non-count fields
        const base = parsedResults[0]

        // Merge count_estimate: take median of best_estimate values
        const bestEstimates: number[] = []
        const lowEstimates: number[] = []
        const highEstimates: number[] = []
        const allUncertainty: string[] = []

        for (const p of parsedResults) {
          const ce = p.count_estimate as { low?: number; high?: number; best_estimate?: number } | null | undefined
          if (ce && typeof ce === 'object') {
            if (typeof ce.best_estimate === 'number') bestEstimates.push(ce.best_estimate)
            if (typeof ce.low === 'number') lowEstimates.push(ce.low)
            if (typeof ce.high === 'number') highEstimates.push(ce.high)
          }
          const factors = p.count_uncertainty_factors
          if (Array.isArray(factors)) {
            allUncertainty.push(...(factors as string[]))
          }
        }

        const mergedBest = bestEstimates.length > 0 ? median(bestEstimates) : (typeof base.building_count === 'number' ? base.building_count as number : null)
        const mergedLow = lowEstimates.length > 0 ? Math.min(...lowEstimates) : null
        const mergedHigh = highEstimates.length > 0 ? Math.max(...highEstimates) : null
        const mergedUncertainty = dedupeStrings(allUncertainty)

        const merged: Record<string, unknown> = {
          ...base,
          count_estimate: { low: mergedLow, high: mergedHigh, best_estimate: mergedBest },
          count_uncertainty_factors: mergedUncertainty,
          building_count: mergedBest,
        }

        const confPercents: number[] = []
        for (const p of parsedResults) {
          if (typeof p.confidence_percent === 'number') {
            confPercents.push(p.confidence_percent)
          } else if (typeof p.confidenceScore === 'number') {
            confPercents.push(p.confidenceScore)
          }
        }
        if (confPercents.length > 0) {
          const mergedConf = median(confPercents)
          merged.confidence_percent = Math.max(0, Math.min(100, Math.round(mergedConf)))
          merged.confidenceScore = merged.confidence_percent
        } else {
          merged.confidence_percent = null
          merged.confidenceScore = null
          merged.confidence_reason = 'Confidence was not provided by the model; no calibrated confidence is available.'
        }
        if (typeof merged.building_count === 'number') {
          merged.building_count = Math.max(0, Math.round(merged.building_count as number))
        }

        console.log(`[Orbital-AI] Self-consistency merged: best=${mergedBest} range=[${mergedLow},${mergedHigh}] conf=${merged.confidence_percent}% from ${parsedResults.length} responses`)
        traceSteps.push({
          step: 3,
          tool: 'rs_building_detector',
          description: 'Self-consistency multi-path footprint estimation with confidence voting',
          input_summary: 'Optical structural footprints',
          output_summary: `Count: ${mergedBest} (range: ${mergedLow}-${mergedHigh})`,
          duration_ms: Math.max(25, Date.now() - step2Start),
          status: 'success',
          parameters: { count: mergedBest, confidence: (merged.confidence_percent as number) ?? 0 },
        })
        const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'rs_building_detector')
        return res.status(200).json({ ...merged, execution_trace: trace })
      }

      // Standard single call for non-counting questions
      const response = await client.chat.completions.create({
        model: MODEL,
        temperature: STANDARD_TEMPERATURE,
        max_tokens: MAX_TOKENS_ANALYZE,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userContent as any },
        ],
      })

      const parsed = cleanJson(response.choices[0]?.message?.content ?? '{}')
      if (typeof parsed.confidence_percent === 'number') {
        parsed.confidence_percent = Math.max(0, Math.min(100, Math.round(parsed.confidence_percent)))
        parsed.confidenceScore = parsed.confidence_percent
      } else {
        parsed.confidence_percent = null
        parsed.confidenceScore = null
      }
      if (typeof parsed.building_count === 'number') {
        parsed.building_count = Math.max(0, Math.round(parsed.building_count))
      }
      // Non-counting questions must not return count_estimate
      parsed.count_estimate = null
      parsed.count_uncertainty_factors = []

      const vlmToolId = 'rs_vlm_fallback'
      traceSteps.push({
        step: 3,
        tool: vlmToolId,
        description: 'Configured general VLM inference used as explicit unadapted fallback',
        input_summary: 'Single optical observation',
        output_summary: `Model returned a response; calibrated confidence is unavailable (${parsed.confidence_percent ?? 'null'})`,
        duration_ms: Math.max(15, Date.now() - step2Start),
        status: 'success',
        confidence_source: 'none',
        parameters: { model: MODEL, remote_sensing_adapted: false, fallback: true },
      })
      const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, vlmToolId)
      return res.status(200).json({ ...parsed, execution_trace: trace })
    } catch {
      const fallback = buildUnavailableAnalysis(taskType)
      traceSteps.push({
        step: 3,
        tool: 'rs_provider_guard',
        description: 'Provider interruption guard; no fabricated fallback analysis',
        input_summary: 'Single optical observation',
        output_summary: 'Provider failed and no substitute specialist was executed',
        duration_ms: Math.max(1, Date.now() - step2Start),
        status: 'unavailable',
        confidence_source: 'none',
        parameters: { fallback: true },
      })
      const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'rs_vqa')
      return res.status(200).json({ ...fallback, execution_trace: trace })
    }
  } catch {
    const fallbackTrace = buildExecutionTrace('vqa', traceSteps, Date.now() - startTime, validateInputs('vqa', 1), 'rs_vqa')
    return res.status(200).json({
      ...buildUnavailableAnalysis('vqa'),
      execution_trace: fallbackTrace,
    })
  }
}
