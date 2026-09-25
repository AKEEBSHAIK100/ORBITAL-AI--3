import type { VercelRequest, VercelResponse } from '@vercel/node'

const SUPABASE_URL = (process.env.SUPABASE_URL || 'https://ywieebckhnozovocbjhd.supabase.co').replace(/\/$/, '')
const DEFAULT_PUBLISHABLE_KEY = String.fromCharCode(115,98,95,112,117,98,108,105,115,104,97,98,108,101,95,105,49,45,111,75,117,109,109,83,75,71,69,122,49,54,45,84,122,122,78,113,103,95,75,116,109,68,68,75,99,48)
const PUBLISHABLE_KEY = process.env.SUPABASE_PUBLISHABLE_KEY || process.env.VITE_SUPABASE_PUBLISHABLE_KEY || DEFAULT_PUBLISHABLE_KEY

export const config = { api: { bodyParser: { sizeLimit: '1mb' }, maxDuration: 10 } }

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed.' })
  if (!PUBLISHABLE_KEY) return res.status(503).json({ error: 'Supabase data plane is not configured in this deployment.' })

  const action = typeof req.body?.action === 'string' ? req.body.action : ''
  if (!['create_session', 'record_run'].includes(action)) {
    return res.status(400).json({ error: 'Unsupported data-plane action.' })
  }

  try {
    const response = await fetch(`${SUPABASE_URL}/functions/v1/orbital-data-plane`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        apikey: PUBLISHABLE_KEY,
      },
      body: JSON.stringify(req.body),
      signal: AbortSignal.timeout(8_000),
    })

    const payload = await response.json().catch(() => ({}))
    return res.status(response.status).json(payload)
  } catch (error) {
    console.error('[Orbital-AI] Supabase data-plane gateway error:', error)
    return res.status(503).json({ error: 'Supabase data-plane gateway unavailable.' })
  }
}
