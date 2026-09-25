type SupabaseConfig = {
  url: string
  publishableKey: string
}

function getConfig(): SupabaseConfig | null {
  const url = String(import.meta.env.VITE_SUPABASE_URL || '').replace(/\/$/, '')
  const publishableKey = String(import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY || '')
  if (!url || !publishableKey) return null
  return { url, publishableKey }
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
  userId: string
}) {
  const response = await request('analysis_sessions', {
    method: 'POST',
    headers: { Prefer: 'return=representation' },
    body: JSON.stringify({
      query: input.query,
      modality: input.modality,
      task_type: input.taskType,
      user_id: input.userId,
      status: 'created',
    }),
  })
  if (!response.ok) throw new Error(`Supabase session creation failed: ${response.status}`)
  const rows = await response.json()
  return rows[0]
}

export async function getAnalysisRuns(sessionId: string) {
  const response = await request(`analysis_runs?session_id=eq.${encodeURIComponent(sessionId)}&order=created_at.desc`)
  if (!response.ok) throw new Error(`Supabase run lookup failed: ${response.status}`)
  return response.json()
}
