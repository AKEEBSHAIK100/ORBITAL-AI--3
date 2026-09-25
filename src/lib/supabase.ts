type SupabaseConfig = { url: string }

const DEFAULT_URL = 'https://ywieebckhnozovocbjhd.supabase.co'

function getConfig(): SupabaseConfig | null {
  const url = String(import.meta.env.VITE_SUPABASE_URL || DEFAULT_URL).replace(/\/$/, '')
  return url ? { url } : null
}

export function isSupabaseConfigured(): boolean {
  return getConfig() !== null
}

async function dataPlaneRequest(body: Record<string, unknown>) {
  const config = getConfig()
  if (!config) throw new Error('Supabase data plane is not configured for this deployment.')

  const response = await fetch('/api/data-plane', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(String(payload?.error || `Supabase data-plane request failed: ${response.status}`))
  }
  return payload
}

export async function createAnalysisSession(input: {
  query: string
  modality: string
  taskType: string
  userId?: string | null
}) {
  return dataPlaneRequest({
    action: 'create_session',
    session: {
      query: input.query,
      modality: input.modality,
      task_type: input.taskType,
    },
  })
}

export async function recordAnalysisRun(input: {
  sessionId: string
  taskType: string
  modelName?: string | null
  modelAdaptation?: string | null
  result: unknown
  executionTrace?: unknown
  confidence?: number | null
  confidenceStatus?: string
  status?: string
}) {
  return dataPlaneRequest({
    action: 'record_run',
    run: {
      session_id: input.sessionId,
      task_type: input.taskType,
      model_name: input.modelName ?? null,
      model_adaptation: input.modelAdaptation ?? null,
      result: input.result ?? {},
      execution_trace: input.executionTrace ?? [],
      confidence: input.confidence ?? null,
      confidence_status: input.confidenceStatus ?? 'not_calibrated',
      status: input.status ?? 'completed',
    },
  })
}
