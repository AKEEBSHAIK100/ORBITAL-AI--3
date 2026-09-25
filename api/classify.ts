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

// ── Heuristic pixel-based land-cover fallback ─────────────────────────────────
function heuristicBENScores(imageBase64: string): Record<string, number> {
  // Decode a small sample of the base64 string to estimate spectral content
  try {
    const sample = imageBase64.replace(/^data:image\/[^;]+;base64,/, '').slice(0, 3000)
    let rSum = 0, gSum = 0, bSum = 0, count = 0
    for (let i = 0; i < sample.length - 3; i += 4) {
      const byte = sample.charCodeAt(i) & 0xFF
      if (count % 3 === 0) rSum += byte
      else if (count % 3 === 1) gSum += byte
      else bSum += byte
      count++
    }
    const r = rSum / (count / 3 + 1), g = gSum / (count / 3 + 1), b = bSum / (count / 3 + 1)
    const scores: number[] = new Array(19).fill(0)
    if (b > r * 1.1 && b > 50) {
      // Water dominant
      scores[17] = 0.82; scores[15] = 0.30; scores[16] = 0.22
    } else if (g > r * 1.08 && g > 40) {
      // Vegetation dominant
      scores[8] = 0.74; scores[2] = 0.52; scores[4] = 0.40; scores[10] = 0.28
    } else if (r > 120 && g > 90 && b < 90) {
      // Arid
      scores[11] = 0.68; scores[13] = 0.48; scores[14] = 0.35
    } else {
      // Urban
      scores[0] = 0.78; scores[1] = 0.42; scores[2] = 0.18
    }
    return Object.fromEntries(scores.map((s, i) => [String(i), s]))
  } catch {
    const scores: Record<string, number> = {}
    scores['0'] = 0.60 // Urban fabric fallback
    return scores
  }
}

function buildHeuristicResponse(imageBase64: string, topK: number, threshold: number) {
  const rawScores = heuristicBENScores(imageBase64)
  const labels = BEN_CLASSES.map((cls, i) => ({
    name: cls.name,
    short: cls.short,
    score: rawScores[String(i)] ?? 0,
    active: (rawScores[String(i)] ?? 0) >= threshold,
  })).sort((a, b) => b.score - a.score)

  const topLabels = labels.slice(0, topK)
  const activeLabels = labels.filter(l => l.active).slice(0, topK)
  const top = topLabels[0]

  return {
    labels: topLabels,
    active_labels: activeLabels,
    top_label: top?.short ?? 'Unknown',
    confidence: Math.round((top?.score ?? 0) * 100 * 10) / 10,
    model_id: 'heuristic-vercel-fallback',
    available: false,
    device: 'cpu',
    note: 'Heuristic estimation — Python backend with configilm not available on Vercel.',
    citation: '',
  }
}

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

  // ── Heuristic fallback ────────────────────────────────────────────────────
  const result = buildHeuristicResponse(image, topK, thresh)
  return res.status(200).json(result)
}
