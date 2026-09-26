import type { VercelRequest, VercelResponse } from '@vercel/node'

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '15mb',
    },
  },
}

/**
 * Building detection is only returned when the executable Python specialist
 * actually performs the inference. This serverless handler deliberately has
 * no default/sample detections and no client-side heuristic fallback.
 */
export default async function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader('Access-Control-Allow-Credentials', 'true')
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,POST')
  res.setHeader(
    'Access-Control-Allow-Headers',
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version',
  )

  if (req.method === 'OPTIONS') return res.status(200).end()
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed. Use POST.' })

  const base = (process.env.PYTHON_BACKEND_URL || '').replace(/\/+$/, '')
  if (!base) {
    return res.status(503).json({
      error: 'Building detection specialist is unavailable.',
      code: 'MODEL_UNAVAILABLE',
      building_count: null,
      detections: [],
      confidence: null,
      confidence_level: 'unavailable',
      data_limitation_note: 'No executable building-detection specialist is configured for this deployment. No sample or heuristic detections are returned.',
    })
  }

  try {
    const response = await fetch(`${base}/analyze/buildings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: typeof req.body === 'object' ? JSON.stringify(req.body) : req.body,
      signal: AbortSignal.timeout(10_000),
    })

    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      return res.status(503).json({
        error: 'Building detection specialist is unavailable.',
        code: 'MODEL_UNAVAILABLE',
        building_count: null,
        detections: [],
        confidence: null,
        confidence_level: 'unavailable',
        data_limitation_note: `Building specialist returned HTTP ${response.status}. No fallback estimate was generated.`,
      })
    }

    return res.status(200).json(data)
  } catch {
    return res.status(503).json({
      error: 'Building detection specialist is unavailable.',
      code: 'MODEL_UNAVAILABLE',
      building_count: null,
      detections: [],
      confidence: null,
      confidence_level: 'unavailable',
      data_limitation_note: 'The configured building specialist could not be reached. No sample or heuristic detections are returned.',
    })
  }
}
