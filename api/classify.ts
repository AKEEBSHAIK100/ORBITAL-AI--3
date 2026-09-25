/**
 * SatQuery AI — Vercel Serverless: /api/classify
 * Proxies to the Python backend BEN classifier, or runs a lightweight
 * heuristic estimation if the backend is unavailable.
 *
 * POST /api/classify
 * Body: { image: string (base64/dataURL), top_k?: number, threshold?: number }
 */

import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getBackendUrl } from './_lib.js'

// ── BigEarthNet 19 classes ────────────────────────────────────────────────────
const BEN_CLASSES = [
  { name: 'Urban fabric',                                                          short: 'Urban Fabric' },
  { name: 'Industrial or commercial units',                                        short: 'Industrial/Commercial' },
  { name: 'Arable land',                                                           short: 'Arable Land' },
  { name: 'Permanent crops',                                                       short: 'Permanent Crops' },
  { name: 'Pastures',                                                              short: 'Pastures' },
  { name: 'Complex cultivation patterns',                                          short: 'Complex Cultivation' },
  { name: 'Land principally occupied by agriculture, with significant areas of natural vegetation', short: 'Agri + Natural Veg' },
  { name: 'Agro-forestry areas',                                                   short: 'Agro-Forestry' },
  { name: 'Broad-leaved forest',                                                   short: 'Broad-Leaved Forest' },
  { name: 'Coniferous forest',                                                     short: 'Coniferous Forest' },
  { name: 'Mixed forest',                                                          short: 'Mixed Forest' },
  { name: 'Natural grassland and sparsely vegetated areas',                        short: 'Natural Grassland' },
  { name: 'Moors, heathland and sclerophyllous vegetation',                        short: 'Moors & Heathland' },
  { name: 'Transitional woodland/shrub',                                           short: 'Transitional Woodland' },
  { name: 'Beaches, dunes, sands',                                                 short: 'Beaches & Dunes' },
  { name: 'Inland wetlands',                                                       short: 'Inland Wetlands' },
  { name: 'Coastal wetlands',                                                      short: 'Coastal Wetlands' },
  { name: 'Inland waters',                                                         short: 'Inland Waters' },
  { name: 'Marine waters',                                                         short: 'Marine Waters' },
]

// Vercel serverless does not ship the trained BigEarthNet classifier weights.
 // Never derive a land-cover prediction from encoded JPEG/base64 bytes.
 
// ── Handler ───────────────────────────────────────────────────────────────────
export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS preflight
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
  if (req.method === 'OPTIONS') return res.status(200).end()
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' })

  const { image, top_k = 5, threshold = 0.25 } = req.body ?? {}
  if (!image || typeof image !== 'string') {
    return res.status(400).json({ error: 'Missing required field: image (base64 string)' })
  }

  const topK = Math.min(Math.max(Number(top_k) || 5, 1), 19)
  const thresh = Math.min(Math.max(Number(threshold) || 0.25, 0), 1)

  // ── Try Python backend first ──────────────────────────────────────────────
  const backendUrl = getBackendUrl()
  if (backendUrl) {
    try {
      const backendRes = await fetch(`${backendUrl}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image, top_k: topK, threshold: thresh }),
        signal: AbortSignal.timeout(12000),
      })
      if (backendRes.ok) {
        const data = await backendRes.json()
        return res.status(200).json(data)
      }
    } catch {
      // Fall through to heuristic
    }
  }

  return res.status(503).json({
    labels: [],
    active_labels: [],
    top_label: null,
    confidence: null,
    model_id: null,
    available: false,
    device: null,
    note: 'The trained BigEarthNet classifier is not available in the Vercel serverless runtime. No heuristic land-cover prediction is generated.',
  })
}
