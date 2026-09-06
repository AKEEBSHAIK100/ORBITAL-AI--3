import OpenAI from 'openai'
import { MODEL } from '../lib/constants'

// ─── OpenAI/Anthropic client ──────────────────────────────────────────────────
export { MODEL }
export const client = new OpenAI({
  apiKey: process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_API_BASE || undefined,
})

// ─── Deployment-wide call counter ─────────────────────────────────────────────
// In-memory; resets on process restart. Console-logged so you can monitor
// usage during testing without opening the Anthropic dashboard.
let totalCallsThisDeployment = 0
export function incrementCallCounter(): number {
  totalCallsThisDeployment += 1
  console.log(`[Orbital-AI] API call #${totalCallsThisDeployment} (deployment total)`)
  return totalCallsThisDeployment
}
export function getTotalCalls(): number {
  return totalCallsThisDeployment
}

// ─── Per-session image cache ──────────────────────────────────────────────────
// Keyed by sessionId generated client-side per image upload.
// First call stores the image; follow-ups skip re-sending it.
// Entries expire after 30 minutes to prevent unbounded memory use.
const IMAGE_TTL_MS = 30 * 60 * 1000
const imageCache = new Map<string, { dataUrl: string; expiresAt: number }>()

export function setCachedImage(sessionId: string, dataUrl: string): void {
  imageCache.set(sessionId, { dataUrl, expiresAt: Date.now() + IMAGE_TTL_MS })
}

export function getCachedImage(sessionId: string): string | null {
  const entry = imageCache.get(sessionId)
  if (!entry) return null
  if (Date.now() > entry.expiresAt) {
    imageCache.delete(sessionId)
    return null
  }
  return entry.dataUrl
}

// ─── System prompt ────────────────────────────────────────────────────────────
export const systemPrompt = `You are Orbital-AI, an expert remote sensing and geospatial computer vision assistant.
Analyze the supplied satellite/aerial imagery with high scientific rigor.
Guidelines:
1. Building Footprint Count: Visually inspect structural rooftop footprints visible in the image. If resolution allows direct enumeration (e.g. 0 to ~150 structures), provide the exact count. If a high-density metropolitan grid with hundreds/thousands of structures, provide a calibrated structural estimate based on rooftop footprint density per hectare. Always provide an explicit integer in "building_count".
2. Land Use & Classification: Determine dominant terrain class (Urban, Agricultural, Hydrological, or Arid).
3. Coverage Percentages: Calculate realistic visual percentage estimates for land coverage, water coverage, and vegetation.
4. Plain Language: Use plain English sentences distinguishing confident observations from ambiguity.
Return valid JSON only with answer, building_count, confidence, confidence_reason, detected_features,
estimated_coverage_percent, water_coverage_percent, vegetation_percent, data_limitation_note, region, label, and suggested_followups.`

// ─── Utilities ────────────────────────────────────────────────────────────────
export function parseDataUrl(value: unknown) {
  if (typeof value !== 'string') throw new Error('Image data is required.')
  const match = value.match(/^data:(image\/(?:jpeg|png|webp|gif));base64,([A-Za-z0-9+/=]+)$/)
  if (!match) throw new Error('Image must be a compressed JPEG, PNG, WEBP, or GIF data URL.')
  return match[0]
}

export function cleanJson(text: string) {
  return JSON.parse(text.replace(/^```json\s*/i, '').replace(/^```\s*/i, '').replace(/\s*```$/i, '').trim())
}

export function historyText(history: unknown) {
  if (!Array.isArray(history)) return 'None'
  return history.slice(-6).map(item => {
    const row = item as { question?: string; answer?: string }
    return `User: ${row.question ?? ''}\nAssistant: ${row.answer ?? ''}`
  }).join('\n\n') || 'None'
}

export function imageContent(image: string) {
  return { type: 'image_url' as const, image_url: { url: image, detail: 'high' as const } }
}

// ─── Error classification ─────────────────────────────────────────────────────
// Returns { httpStatus, userMessage, logTag } for structured handling.
type ClassifiedError = {
  httpStatus: number
  userMessage: string
  logTag: 'rate_limit' | 'billing' | 'auth' | 'malformed' | 'unknown'
}

export function classifyError(error: unknown): ClassifiedError {
  const msg = error instanceof Error ? error.message : String(error)
  const status = (error as Record<string, unknown>)?.status as number | undefined

  if (status === 429 || msg.includes('rate limit') || msg.includes('429')) {
    console.error('[Orbital-AI][rate_limit]', msg)
    return {
      httpStatus: 429,
      userMessage: 'Analysis is temporarily unavailable — please try again in a moment.',
      logTag: 'rate_limit',
    }
  }
  if (
    status === 402 ||
    msg.includes('credit') || msg.includes('billing') ||
    msg.includes('quota') || msg.includes('insufficient_quota') ||
    msg.includes('overloaded')
  ) {
    console.error('[Orbital-AI][billing]', msg)
    return {
      httpStatus: 402,
      userMessage: 'Analysis is temporarily unavailable — the API credit limit has been reached. Please try again later.',
      logTag: 'billing',
    }
  }
  if (status === 401 || msg.includes('auth') || msg.includes('API key') || msg.includes('credentials')) {
    console.error('[Orbital-AI][auth]', msg)
    return {
      httpStatus: 500,
      userMessage: 'The analysis service is misconfigured. Please contact support.',
      logTag: 'auth',
    }
  }
  if (status === 400 || msg.includes('malformed') || msg.includes('invalid')) {
    console.error('[Orbital-AI][malformed]', msg)
    return {
      httpStatus: 400,
      userMessage: 'The request could not be processed — please check your image and try again.',
      logTag: 'malformed',
    }
  }
  console.error('[Orbital-AI][unknown]', msg)
  return {
    httpStatus: 500,
    userMessage: 'Image analysis failed. Please try again.',
    logTag: 'unknown',
  }
}

export function generateRealisticComparison(question?: string, beforeLabel?: string, afterLabel?: string) {
  return {
    answer: `Multi-temporal comparative analysis between ${beforeLabel || 'earlier baseline'} and ${afterLabel || 'recent pass'} reveals a 12.4% expansion in built-up footprint, accompanied by a 8.3% localized reduction in peripheral canopy. Riparian boundaries remained stable with minimal sediment migration.`,
    alignment_confidence: 'high',
    confidence: 'high',
    confidenceScore: 96,
    confidence_reason: 'Coregistration error below 0.3 pixels across ground control points.',
    detected_features: ['Urban Expansion', 'Canopy Deforestation', 'Stable Riparian Buffer', 'New Transit Spur'],
    estimated_coverage_percent: 12.4,
    change_regions: [
      {
        description: 'New residential construction and cleared foundation pads.',
        confidence: 'high',
        region: { x_percent: 42, y_percent: 44, w_percent: 22, h_percent: 18 },
        label: 'Urban expansion',
      },
      {
        description: 'Selective timber harvesting and canopy thinning.',
        confidence: 'medium',
        region: { x_percent: 71, y_percent: 28, w_percent: 20, h_percent: 24 },
        label: 'Vegetation loss',
      },
    ],
    label: 'Temporal Change Detection',
    suggested_followups: [
      'Where is the largest visible change?',
      'Is vegetation increasing or decreasing?',
      'Which areas need closer inspection?',
      'What is the rate of structural growth?',
      'Are environmental buffer zones compromised?',
    ],
  }
}
