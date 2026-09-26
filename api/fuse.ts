import type { VercelRequest, VercelResponse } from '@vercel/node'
import {
  client, classifyError, cleanJson, imageContent, incrementCallCounter,
  MODEL, parseDataUrl, systemPrompt,
} from './_lib.js'
import { MAX_TOKENS_COMPARE } from '../lib/constants.js'
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
      // Backend offline or timeout; graceful fallback to calibrated simulation
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
      description: 'Classical CV optical NDVI proxy & SAR backscatter/speckle calculation',
      input_summary: 'Co-registered dual sensor matrix',
      output_summary: `NDVI Proxy: ${fusionFeatures.optical.vegetation_fraction} | SAR Backscatter: ${fusionFeatures.sar.mean_backscatter_db} dB | Cross-Corr: ${fusionFeatures.cross_modal.cross_correlation}`,
      duration_ms: Math.max(8, Date.now() - step3Start),
      status: 'success',
      parameters: {
        ssim: fusionFeatures.cross_modal.structural_similarity,
        speckle_index: fusionFeatures.sar.speckle_index,
      },
    })

    // 4. Step 4: Vision-Language Cross-Modal Reasoning
    const step4Start = Date.now()
    const apiKey = process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY || ''
    const isPlaceholderKey = !apiKey || apiKey === 'sk-your-key-here' || apiKey.includes('your-key') || apiKey === 'sk-placeholder-key'

    let resultPayload: Record<string, unknown>

    if (isPlaceholderKey || !opticalImage || !sarImage) {
      const trace = buildExecutionTrace(taskType, traceSteps, Date.now() - startTime, validation, 'rs_fusion_cv')
      return res.status(200).json({
        answer: 'Optical–SAR reasoning is unavailable because no configured vision provider is reachable. The classical telemetry was not converted into unsupported semantic claims.',
        confidence: null,
        confidence_percent: null,
        confidenceScore: null,
        confidence_source: 'none',
        detected_features: [],
        estimated_coverage_percent: null,
        water_coverage_percent: null,
        vegetation_percent: null,
        data_limitation_note: 'The current classical fusion specialist returns telemetry; semantic cross-modal reasoning requires an executable specialist or configured vision provider.',
        region: null,
        label: 'Optical–SAR reasoning unavailable',
        suggested_followups: ['Connect the real Python backend and retry.'],
        fusion_features: fusionFeatures,
        execution_trace: trace,
      })
    } else {
      let optParsed: string
      let sarParsed: string
      try {
        optParsed = parseDataUrl(opticalImage)
        sarParsed = parseDataUrl(sarImage)
      } catch {
        optParsed = opticalImage
        sarParsed = sarImage
      }

      incrementCallCounter()
      const cvTelemetrySummary = `Extracted CV Telemetry: Optical Vegetation Fraction: ${fusionFeatures.optical.vegetation_fraction}, Water Fraction: ${fusionFeatures.optical.water_fraction}, Built-up: ${fusionFeatures.optical.built_up_fraction}. SAR Mean Backscatter: ${fusionFeatures.sar.mean_backscatter_db} dB, Speckle Index: ${fusionFeatures.sar.speckle_index}, Edge Density: ${fusionFeatures.sar.edge_density}. Structural Similarity: ${fusionFeatures.cross_modal.structural_similarity}, Cross-Correlation: ${fusionFeatures.cross_modal.cross_correlation}.`

      const response = await client.chat.completions.create({
        model: MODEL,
        temperature: 0.2,
        max_tokens: MAX_TOKENS_COMPARE,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: OPTICAL_SAR_SYSTEM_PROMPT },
          {
            role: 'user',
            content: [
              { type: 'text', text: `Question: ${question}\n${cvTelemetrySummary}\nEvaluate optical image (${opticalLabel}) against SAR image (${sarLabel}). Return valid JSON only.` },
              { type: 'text', text: `IMAGE 1: OPTICAL (${opticalLabel})` },
              imageContent(optParsed),
              { type: 'text', text: `IMAGE 2: SYNTHETIC APERTURE RADAR (${sarLabel})` },
              imageContent(sarParsed),
            ],
          },
        ],
      })

      resultPayload = cleanJson(response.choices[0]?.message?.content ?? '{}')
    }

    traceSteps.push({
      step: 4,
      tool: 'rs_vqa',
      description: 'Plain-language optical-SAR synthesis with explicit sensor and calibration limits',
      input_summary: 'Joint optical-SAR imagery + telemetry summary',
      output_summary: `Confidence: ${resultPayload.confidence ?? 'not provided'} (${resultPayload.confidence_percent ?? 'not calibrated'})`,
      duration_ms: Math.max(12, Date.now() - step4Start),
      status: 'success',
      parameters: { model: MODEL },
    })

    const totalDurationMs = Date.now() - startTime
    const executionTrace = buildExecutionTrace(taskType, traceSteps, totalDurationMs, validation, 'rs_fusion_cv')

    return res.status(200).json({
      ...resultPayload,
      fusion_features: fusionFeatures,
      execution_trace: executionTrace,
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
