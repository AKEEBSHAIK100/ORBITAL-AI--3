import type { VercelRequest, VercelResponse } from '@vercel/node'

export default function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed.' })

  const pythonBackendConfigured = Boolean(process.env.PYTHON_BACKEND_URL?.trim())
  const supabaseConfigured = Boolean(process.env.SUPABASE_URL?.trim() || process.env.VITE_SUPABASE_URL?.trim())
  const hfWorkerConfigured = process.env.ENABLE_HF_RS_WORKER === 'true'
  const providerConfigured = Boolean(
    (process.env.OPENAI_API_KEY && !process.env.OPENAI_API_KEY.includes('your-key')) ||
    (process.env.ANTHROPIC_API_KEY && !process.env.ANTHROPIC_API_KEY.includes('your-key'))
  )

  return res.status(200).json({
    service: 'SatQuery AI production capability registry',
    release_integrity: {
      fabricated_metrics_disabled: true,
      synthetic_change_results_disabled: true,
      synthetic_optical_sar_results_disabled: true,
      confidence_requires_calibrated_source: true,
    },
    orchestration: {
      agentic_router: true,
      input_validation: true,
      execution_trace: true,
      specialist_registry: true,
    },
    required_sih_workflows: {
      single_image_vqa: {
        configured_runtime: pythonBackendConfigured,
        remote_sensing_adaptation: 'BigEarthNet-derived BLIP-VQA LoRA pilot adapter',
      },
      single_image_captioning: {
        configured_runtime: pythonBackendConfigured,
        remote_sensing_adaptation: 'BigEarthNet-derived BLIP caption LoRA pilot adapter',
      },
      bi_temporal_change: {
        configured_runtime: pythonBackendConfigured,
        specialist: 'classical change baseline with compatibility validation',
      },
      optical_sar_joint_analysis: {
        configured_runtime: pythonBackendConfigured,
        specialist: 'classical optical-SAR telemetry plus orchestrated reasoning',
      },
    },
    runtime: {
      python_backend_configured: pythonBackendConfigured,
      supabase_data_plane_configured: supabaseConfigured,
      free_remote_worker: hfWorkerConfigured ? 'Hugging Face ZeroGPU worker enabled' : 'not enabled',
      vision_provider_configured: providerConfigured,
      free_hf_zero_gpu_worker_configured: hfWorkerConfigured,
      note: pythonBackendConfigured
        ? 'Python remote-sensing backend is configured for production routing.'
        : 'Python remote-sensing backend is not configured in this Vercel deployment; specialist execution is unavailable here.',
    },
  })
}
