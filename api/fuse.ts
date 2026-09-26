import type { VercelRequest, VercelResponse } from '@vercel/node'
import { classifyError } from './_lib.js'
import {
  classifyTask, validateInputs, buildExecutionTrace,
  type ExecutionTraceStep, type FusionFeatures,
} from '../lib/agentController.js'
import { runWorkerFusion } from './_hfWorker.js'

export const config = { api: { bodyParser: { sizeLimit: '15mb' } } }

const OPTICAL_SAR_SYSTEM_PROMPT = `${systemPrompt}
You are operating in Optical–SAR joint-analysis mode.
Two user-supplied images are provided. Do not assume they are perfectly co-registered unless metadata or a verified preprocessing step establishes that.

COMMUNICATION RULES:
- Explain the result in plain language for a non-expert.
- Separate observations supported by the optical image from observations supported by SAR.
- Do not invent sensor bands, physical units, percentages, dates, or confidence scores.
- Do not call RGB imagery multispectral unless the supplied data actually contains those bands.
- Do not claim NDVI from RGB-only imagery.
- Do not interpret raw image brightness as calibrated SAR backscatter in dB unless the input is a radiometrically calibrated SAR product.
- If alignment cannot be verified, say so and treat cross-modal conclusions as provisional.

Return valid JSON using the standard ORBITAL-AI schema.`

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed.' })

  const startTime = Date.now()
  const traceSteps: ExecutionTraceStep[] = []

  try {
    const {
      opticalImage,
      sarImage,
      question = 'Conduct joint optical and SAR cross-modal feature analysis.',
      opticalLabel = 'Cartosat-2S / Optical RGB',
      sarLabel = 'RISAT-1A / Sentinel-1 SAR',
    } = req.body as Record<string, string | undefined>

    // 1. Step 1: Agent Intent Classification
    const step1Start = Date.now()
    const taskType = classifyTask(question, 2, ['optical', 'sar'])
    traceSteps.push({
      step: 1,
      tool: 'rs_task_classifier',
      description: 'Deterministic rule-based task routing and intent extraction',
      input_summary: `Query: "${question.slice(0, 70)}" | Sensors: [Optical, SAR]`,
      output_summary: `Task classified as: "${taskType}" (Routing to multi-sensor fusion pipeline)`,
      duration_ms: Math.max(1, Date.now() - step1Start),
      status: 'success',
      parameters: { task_type: taskType, modalities_count: 2 },
    })

    // 2. Step 2: Input Verification
    const step2Start = Date.now()
    const validation = validateInputs(taskType, 2, ['optical', 'sar'], ['jpeg', 'png'])
    traceSteps.push({
      step: 2,
      tool: 'rs_input_validator',
      description: 'Multi-sensor coregistration and radiometric verification',
      input_summary: `Optical: ${opticalLabel} | SAR: ${sarLabel}`,
      output_summary: validation.notes.join('; '),
      duration_ms: Math.max(1, Date.now() - step2Start),
      status: 'success',
      parameters: { compatibility: validation.compatibility },
    })

    // Free Hugging Face ZeroGPU worker executes the explicitly-labelled optical-SAR baseline.
    if (process.env.ENABLE_HF_RS_WORKER !== 'false' && opticalImage && sarImage) {
      try {
        const worker = await runWorkerFusion(opticalImage, sarImage) as Record<string, any>
        if (worker?.ok) {
          traceSteps.push({
            step: 3,
            tool: 'optical_sar_classical',
            description: 'Hugging Face worker classical optical-SAR joint telemetry',
            input_summary: 'Optical + SAR image pair',
            output_summary: 'External ZeroGPU optical-SAR reasoning response returned',
            duration_ms: Math.max(1, Number(worker.duration_ms) || Date.now() - step2Start),
            status: 'success',
            confidence_source: 'none',
            parameters: { method: worker.method, external_dependency: true, confidence_status: 'not_calibrated' },
          })
          const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'optical_sar_classical')
          return res.status(200).json({
            answer: String(worker.answer || 'The external optical-SAR specialist returned no textual result.'),
            confidence: null,
            confidence_percent: null,
            confidenceScore: null,
            confidence_source: 'none',
            confidence_status: 'not_calibrated',
            detected_features: [],
            estimated_coverage_percent: null,
            water_coverage_percent: null,
            vegetation_percent: null,
            data_limitation_note: worker.note || 'External ZeroGPU specialist; ORBITAL-AI does not claim independent benchmark validation of this result.',
            region: null,
            label: 'External Optical-SAR Analysis',
            suggested_followups: ['Provide CRS and geotransform metadata for compatibility verification.', 'Use a radiometrically calibrated SAR product for physical backscatter interpretation.'],
            fusion_features: worker,
            execution_trace: trace,
          })
        }
      } catch (workerError: any) {
        console.warn('[Orbital-AI] HF ZeroGPU optical-SAR worker unavailable:', workerError?.message || workerError)
      }
    }

    // 3. Step 3: Classical-CV Feature Extraction
    const step3Start = Date.now()
    let fusionFeatures: FusionFeatures | null = null

    // Try live Python FastAPI backend if reachable
    try {
      if (opticalImage && sarImage) {
        const pyBackend = (process.env.PYTHON_BACKEND_URL || '').replace(/\/+$/, '')
        if (pyBackend) {
          const pyRes = await fetch(`${pyBackend}/analyze/fusion`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ optical_image: opticalImage, sar_image: sarImage }),
            signal: AbortSignal.timeout(1800),
          })
          if (pyRes.ok) {
            const pyJson = (await pyRes.json()) as Record<string, any>
            if (pyJson && pyJson.fusion_features) {
              fusionFeatures = pyJson.fusion_features
            }
          }
        }
      }
    } catch {
      // Backend offline or timeout; no synthetic fusion fallback is permitted
    }

    if (!fusionFeatures) {
      const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'rs_fusion_cv')
      return res.status(200).json({
        answer: 'Optical–SAR analysis is unavailable because the real Python fusion specialist is not connected. No synthetic SAR, physical measurements, or confidence values were generated.',
        confidence: null,
        confidence_percent: null,
        confidenceScore: null,
        confidence_source: 'none',
        detected_features: [],
        estimated_coverage_percent: null,
        water_coverage_percent: null,
        vegetation_percent: null,
        data_limitation_note: 'A real optical–SAR pair must be processed by the remote-sensing fusion specialist.',
        region: null,
        label: 'Optical–SAR analysis unavailable',
        suggested_followups: ['Connect PYTHON_BACKEND_URL to the FastAPI backend and retry with both optical and SAR images.'],
        execution_trace: trace,
      })
    }

    traceSteps.push({
      step: 3,
      tool: 'rs_fusion_cv',
      description: 'Classical CV visible-band vegetation proxy and raw SAR intensity/speckle telemetry',
      input_summary: 'Co-registered dual sensor matrix',
      output_summary: 'Classical optical/SAR telemetry returned; physical calibration is not inferred',
      duration_ms: Math.max(8, Date.now() - step3Start),
      status: 'success',
      parameters: {
        ssim: fusionFeatures.cross_modal.structural_similarity,
        speckle_index: fusionFeatures.sar.speckle_index,
      },
    })

    // No paid VLM fallback. The classical fusion specialist is the final free path.
    traceSteps.push({
      step: 4,
      tool: 'fusion_result_guard',
      description: 'Returned classical optical-SAR telemetry without unverified semantic synthesis',
      input_summary: 'Joint optical + SAR specialist output',
      output_summary: 'Classical telemetry returned; no paid-provider semantic fallback used',
      duration_ms: 0,
      status: 'success',
      confidence_source: 'none',
      parameters: { semantic_vlm_fallback: false },
    })
    const finalTrace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'rs_fusion_cv')
    return res.status(200).json({
      answer: 'The optical-SAR specialist returned classical cross-modal telemetry. These measurements are image-derived and are not presented as calibrated physical SAR quantities or semantic model conclusions.',
      confidence: null,
      confidence_percent: null,
      confidenceScore: null,
      confidence_source: 'none',
      confidence_status: 'not_calibrated',
      detected_features: [],
      estimated_coverage_percent: null,
      water_coverage_percent: null,
      vegetation_percent: null,
      data_limitation_note: 'Semantic cross-modal VLM reasoning is not enabled. Optical and SAR telemetry should be interpreted with sensor calibration and registration metadata when available.',
      region: null,
      label: 'Classical Optical-SAR Telemetry',
      suggested_followups: ['Provide verified CRS/geotransform metadata for registration checks.', 'Use calibrated SAR products for physical backscatter interpretation.'],
      fusion_features: fusionFeatures,
      execution_trace: finalTrace,
    })
  } catch (err) {
    const { userMessage } = classifyError(err)
    const totalDurationMs = Date.now() - startTime
    const validation = validateInputs('sar_optical_fusion', 2, ['optical', 'sar'])
    const executionTrace = buildExecutionTrace('sar_optical_fusion', traceSteps, totalDurationMs, validation, 'rs_fusion_cv')

    return res.status(200).json({
      answer: 'Optical–SAR analysis could not be executed. No synthetic telemetry, semantic result, or confidence was returned.',
      confidence: null,
      confidence_percent: null,
      confidenceScore: null,
      confidence_source: 'none',
      detected_features: [],
      estimated_coverage_percent: null,
      water_coverage_percent: null,
      vegetation_percent: null,
      data_limitation_note: 'Connect the real Python optical–SAR specialist or a configured provider.',
      region: null,
      label: 'Optical–SAR analysis unavailable',
      suggested_followups: ['Retry after connecting the FastAPI backend.'],
      execution_trace: executionTrace,
    })
  }
}
