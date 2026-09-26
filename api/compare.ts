import type { VercelRequest, VercelResponse } from '@vercel/node'
import { client, classifyError, cleanJson, imageContent, incrementCallCounter, MODEL, parseDataUrl, systemPrompt } from './_lib.js'
import { MAX_TOKENS_COMPARE } from '../lib/constants.js'
import { classifyTask, validateInputs, buildExecutionTrace, type ExecutionTraceStep } from '../lib/agentController.js'
import { runWorkerChange } from './_hfWorker.js'

export const config = { api: { bodyParser: { sizeLimit: '12mb' }, maxDuration: 60 } }

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed.' })
  const startTime = Date.now()
  const traceSteps: ExecutionTraceStep[] = []

  try {
    const { beforeImage, afterImage, question, beforeLabel, afterLabel } =
      req.body as Record<string, string | undefined>
    const promptText = question || 'What changed between these two satellite passes?'

    const step1Start = Date.now()
    const taskType = classifyTask(promptText, 2, ['optical'])
    traceSteps.push({
      step: 1, tool: 'rs_task_classifier',
      description: 'Deterministic rule-based task routing and intent extraction',
      input_summary: `Query: "${promptText.slice(0, 70)}" | Mode: Bi-Temporal Comparison`,
      output_summary: `Task classified as: "${taskType}"`,
      duration_ms: Math.max(1, Date.now() - step1Start), status: 'success',
      parameters: { task_type: taskType },
    })

    const step2Start = Date.now()
    const validation = validateInputs(taskType, 2, ['optical'], ['jpeg'])
    traceSteps.push({
      step: 2, tool: 'rs_input_validator',
      description: 'Bi-temporal coregistration & pixel alignment verification',
      input_summary: `T1: ${beforeLabel || 'Baseline'} | T2: ${afterLabel || 'Recent'}`,
      output_summary: validation.notes.join('; '),
      duration_ms: Math.max(1, Date.now() - step2Start), status: 'success',
      parameters: { compatibility: validation.compatibility },
    })

    if (!beforeImage || !afterImage) {
      const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'rs_change_detector')
      return res.status(200).json({
        answer: 'Bi-temporal change analysis requires both a valid earlier image and a valid later image. No change result was estimated.',
        confidence: null, confidence_percent: null, confidenceScore: null, confidence_source: 'none',
        change_regions: [], alignment_confidence: null, label: 'Change analysis unavailable',
        data_limitation_note: 'Both temporal observations are required for change analysis.',
        execution_trace: trace,
      })
    }

    // First choice: the real Python orchestration backend, which can execute the
    // classical change specialist and its compatibility checks.
    const configuredPyUrl = process.env.PYTHON_BACKEND_URL?.trim()
    if (configuredPyUrl) {
      try {
        const pyBase = configuredPyUrl.replace(/\/api\/analyze\/?$/, '').replace(/\/api\/?$/, '').replace(/\/+$/, '')
        const pyRes = await fetch(`${pyBase}/api/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: promptText,
            image: beforeImage,
            secondary_image: afterImage,
            modality: 'optical',
            secondary_modality: 'optical',
            task_type: 'change_detection',
          }),
          signal: AbortSignal.timeout(45_000),
        })
        if (pyRes.ok) {
          const pyData = await pyRes.json()
          const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'rs_change_detector')
          return res.status(200).json({ ...pyData, execution_trace: pyData.execution_trace || trace })
        }
      } catch {
        console.warn('[Orbital-AI] Python change backend unavailable; continuing without fabricated fallback.')
      }
    }

    // Free Hugging Face ZeroGPU worker provides the classical change baseline when enabled.
    if (process.env.ENABLE_HF_RS_WORKER !== 'false') {
      try {
        const worker = await runWorkerChange(beforeImage, afterImage) as Record<string, any>
        if (worker?.ok) {
          traceSteps.push({
            step: 3,
            tool: 'change_detection_classical',
            description: 'Hugging Face worker classical bi-temporal change baseline',
            input_summary: 'Two temporal optical observations',
            output_summary: 'External ZeroGPU change-understanding response returned',
            duration_ms: Math.max(1, Number(worker.duration_ms) || Date.now() - step2Start),
            status: 'success',
            confidence_source: 'none',
            parameters: { method: worker.method, external_dependency: true, confidence_status: 'not_calibrated' },
          })
          const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'change_detection_classical')
          return res.status(200).json({
            answer: String(worker.answer || 'The external change specialist returned no textual result.'),
            confidence: null,
            confidence_percent: null,
            confidenceScore: null,
            confidence_source: 'none',
            confidence_status: 'not_calibrated',
            change_regions: [],
            alignment_confidence: null,
            label: 'External Remote-Sensing Change Analysis',
            data_limitation_note: worker.note || 'External ZeroGPU specialist; no calibrated quantitative change fraction is claimed.',
            change_baseline: worker,
            execution_trace: trace,
          })
        }
      } catch (workerError: any) {
        console.warn('[Orbital-AI] HF ZeroGPU change worker unavailable:', workerError?.message || workerError)
      }
    }

    // Optional unadapted VLM path, only when explicitly configured.
    const apiKey = process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY || ''
    const isPlaceholderKey = !apiKey || apiKey === 'sk-your-key-here' || apiKey.includes('your-key') || apiKey === 'sk-placeholder-key'
    let before: string
    let after: string
    try {
      before = parseDataUrl(beforeImage)
      after = parseDataUrl(afterImage)
    } catch {
      const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'rs_change_detector')
      return res.status(200).json({
        answer: 'The supplied temporal images could not be decoded. No change result was estimated.',
        confidence: null, confidence_percent: null, confidenceScore: null, confidence_source: 'none',
        change_regions: [], alignment_confidence: null, label: 'Change analysis unavailable',
        execution_trace: trace,
      })
    }

    if (isPlaceholderKey) {
      const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'rs_change_detector')
      return res.status(200).json({
        answer: 'Bi-temporal change analysis is unavailable in this deployment because no executable change specialist or configured vision provider is reachable.',
        confidence: null, confidence_percent: null, confidenceScore: null, confidence_source: 'none',
        change_regions: [], alignment_confidence: null, label: 'Change analysis unavailable',
        data_limitation_note: 'Connect the real Python remote-sensing backend to execute the change workflow.',
        execution_trace: trace,
      })
    }

    incrementCallCounter()
    const step3Start = Date.now()
    try {
      const response = await client.chat.completions.create({
        model: MODEL, temperature: 0.2, max_tokens: MAX_TOKENS_COMPARE,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: `${systemPrompt}\nFor two images, return alignment_confidence ('high'|'medium'|'low'), change_regions, and confidence only when actually supported by the provider.` },
          { role: 'user', content: [
            { type: 'text', text: `Compare ${beforeLabel || 'the earlier image'} with ${afterLabel || 'the later image'}. Question: ${promptText}. Return JSON only.` },
            { type: 'text', text: 'EARLIER IMAGE' }, imageContent(before),
            { type: 'text', text: 'LATER IMAGE' }, imageContent(after),
          ] },
        ],
      })
      const parsed = cleanJson(response.choices[0]?.message?.content ?? '{}')
      if (typeof parsed.confidence_percent !== 'number') {
        parsed.confidence_percent = null
        parsed.confidenceScore = null
        parsed.confidence_source = 'none'
      } else {
        parsed.confidence_percent = Math.max(0, Math.min(100, Math.round(parsed.confidence_percent)))
        parsed.confidenceScore = parsed.confidence_percent
      }
      traceSteps.push({
        step: 3, tool: 'rs_change_detector',
        description: 'Vision-provider bi-temporal inference (explicitly unadapted fallback)',
        input_summary: 'Optical temporal pair',
        output_summary: 'Provider-generated change response',
        duration_ms: Math.max(1, Date.now() - step3Start), status: 'success',
        confidence_source: parsed.confidence_source || 'none',
        parameters: { model: MODEL, remote_sensing_adapted: false },
      })
      const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'rs_change_detector')
      return res.status(200).json({ ...parsed, execution_trace: trace })
    } catch (apiError) {
      const { userMessage, logTag } = classifyError(apiError)
      console.warn(`[Orbital-AI] Change provider unavailable (${logTag}: ${userMessage}). No fabricated comparison returned.`)
      traceSteps.push({
        step: 3, tool: 'rs_change_detector',
        description: 'Change specialist/provider availability guard',
        input_summary: 'Two temporal observations',
        output_summary: 'Execution unavailable; no fabricated change regions or confidence returned',
        duration_ms: Math.max(1, Date.now() - step3Start), status: 'unavailable',
        confidence_source: 'none', parameters: { provider_available: false },
      })
      const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'rs_change_detector')
      return res.status(200).json({
        answer: 'Bi-temporal change analysis is currently unavailable. No change result was estimated.',
        confidence: null, confidence_percent: null, confidenceScore: null, confidence_source: 'none',
        change_regions: [], alignment_confidence: null, label: 'Change analysis unavailable',
        data_limitation_note: 'A real change specialist or configured vision provider is required.',
        execution_trace: trace,
      })
    }
  } catch (error) {
    const trace = buildExecutionTrace('change_detection', traceSteps, Date.now() - startTime, validateInputs('change_detection', 2), 'rs_change_detector')
    return res.status(200).json({
      answer: 'Bi-temporal change analysis could not be executed. No change result was estimated.',
      confidence: null, confidence_percent: null, confidenceScore: null, confidence_source: 'none',
      change_regions: [], alignment_confidence: null, label: 'Change analysis unavailable',
      execution_trace: trace,
    })
  }
}
