type SupabaseConfig = { url: string; publishableKey: string }

const DEFAULT_URL = 'https://ywieebckhnozovocbjhd.supabase.co'
const DEFAULT_PUBLISHABLE_KEY = 'sb_publishable_i1-oKummSKGEz16-TzzNqg_KtmDDKc0'

function getConfig(): SupabaseConfig | null {
  const url = String(import.meta.env.VITE_SUPABASE_URL || DEFAULT_URL).replace(/\/$/, '')
  const publishableKey = String(import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY || DEFAULT_PUBLISHABLE_KEY)
  return url && publishableKey ? { url, publishableKey } : null
}

export function isSupabaseConfigured(): boolean {
  return getConfig() !== null
}

async function request(path: string, init: RequestInit = {}) {
  const config = getConfig()
  if (!config) throw new Error('Supabase is not configured.')
  const headers = new Headers(init.headers)
  headers.set('apikey', config.publishableKey)
  headers.set('Content-Type', 'application/json')
  return fetch(`${config.url}/rest/v1/${path}`, { ...init, headers })
}

export async function createAnalysisSession(input: {
  query: string
  modality: string
  taskType: string
  userId?: string | null
}) {
  const response = await request('analysis_sessions', {
    method: 'POST',
    headers: { Prefer: 'return=representation' },
    body: JSON.stringify({
      query: input.query,
      modality: input.modality,
      task_type: input.taskType,
      user_id: input.userId ?? null,
      status: 'created',
    }),
  })
  if (!response.ok) throw new Error(`Supabase session creation failed: ${response.status}`)
  const rows = await response.json()
  return rows[0]
}

export async function updateAnalysisSession(sessionId: string, patch: {
  status?: string
  updated_at?: string
}) {
  const response = await request(`analysis_sessions?id=eq.${encodeURIComponent(sessionId)}`, {
    method: 'PATCH',
    headers: { Prefer: 'return=minimal' },
    body: JSON.stringify(patch),
  })
  if (!response.ok) throw new Error(`Supabase session update failed: ${response.status}`)
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
  const response = await request('analysis_runs', {
    method: 'POST',
    headers: { Prefer: 'return=minimal' },
    body: JSON.stringify({
      session_id: input.sessionId,
      task_type: input.taskType,
      model_name: input.modelName ?? null,
      model_adaptation: input.modelAdaptation ?? null,
      result: input.result ?? {},
      evidence: [],
      execution_trace: input.executionTrace ?? [],
      confidence: input.confidence ?? null,
      confidence_status: input.confidenceStatus ?? 'not_calibrated',
      status: input.status ?? 'completed',
      completed_at: new Date().toISOString(),
    }),
  })
  if (!response.ok) throw new Error(`Supabase run persistence failed: ${response.status}`)
}

export async function getAnalysisRuns(sessionId: string) {
  const response = await request(`analysis_runs?session_id=eq.${encodeURIComponent(sessionId)}&order=created_at.desc`)
  if (!response.ok) throw new Error(`Supabase run lookup failed: ${response.status}`)
  return response.json()
}
