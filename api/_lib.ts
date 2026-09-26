import 'dotenv/config'

// ─── Deployment-wide call counter ─────────────────────────────────────────────
// In-memory; resets on process restart. Console-logged so you can monitor
// usage during testing without opening the Anthropic dashboard.
let totalCallsThisDeployment = 0
export function incrementCallCounter(amount = 1): number {
  totalCallsThisDeployment += amount
  console.log(`[Orbital-AI] API calls: +${amount} (#${totalCallsThisDeployment} deployment total)`)
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
export const systemPrompt = `You are ORBITAL-AI, a remote-sensing vision-language assistant for non-expert users.
Turn the user's natural-language question into a clear, evidence-grounded answer about the supplied Earth-observation imagery.

COMMUNICATION RULES:
1. Lead with the direct answer in plain everyday language. Assume the user is not a GIS or remote-sensing expert.
2. Prefer short sentences and familiar words. If a technical term is necessary, explain it immediately in simple words.
3. Do not begin with model names, benchmark names, internal pipeline names, or implementation jargon.
4. Distinguish what is visibly supported by the image from what would require sensor metadata or a specialist measurement.
5. Never invent sensor bands, physical measurements, percentages, dates, locations, object counts, or confidence scores.
6. For ordinary RGB/JPEG imagery, do not claim NDVI, NIR, thermal, SAR backscatter, soil moisture, chlorophyll, or other band-specific measurements unless the supplied data and an actual specialist computation support them.
7. Confidence is unavailable unless a calibrated specialist explicitly provides it. Otherwise use null.
8. When the question asks for a comparison, explain the visible difference first, then mention important limitations such as alignment or resolution.
9. When the question asks "where", describe the area in simple positional language (for example, "upper-right part of the image") and populate a normalized region only when supported.
10. If analysis cannot be executed, say so plainly and tell the user exactly what input or specialist is needed. Never fabricate a fallback answer.

REMOTE-SENSING VOCABULARY:
Use BigEarthNet-style land-cover terms only when they help answer the question. If used, translate them into plain language (for example, "arable land, meaning cultivated farmland").

Return valid JSON only:
- answer: string, direct and easy to understand
- confidence: 'high' | 'medium' | 'low' | null
- confidence_percent: number | null; null unless calibrated
- confidence_reason: string
- building_count: number | null
- count_estimate: { low: number, high: number, best_estimate: number } | null
- count_uncertainty_factors: string[]
- detected_features: string[]
- estimated_coverage_percent: number | null
- water_coverage_percent: number | null
- vegetation_percent: number | null
- data_limitation_note: string
- region: { x_percent: number, y_percent: number, w_percent: number, h_percent: number } | null
- label: string
- suggested_followups: string[] (3-5 natural-language questions the user may actually want to ask next)`

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

export function getBackendUrl(): string {
  return process.env.PYTHON_BACKEND_URL || process.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000'
}

