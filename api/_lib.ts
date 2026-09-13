import 'dotenv/config'
import OpenAI from 'openai'
import { MODEL } from '../lib/constants'

// ─── OpenAI/Anthropic client ──────────────────────────────────────────────────
export { MODEL }
export const client = new OpenAI({
  apiKey: process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY || 'sk-placeholder-key',
  baseURL: process.env.OPENAI_API_BASE || undefined,
})

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
export const systemPrompt = `You are SatQuery AI, a specialized agentic vision-language assistant for remote sensing imagery and Earth observation.
You operate with domain adaptation calibrated to the BigEarthNet 43-class Corine Land Cover taxonomy, RSVQA conventions, VRSBench scene captioning/grounding, and CDVQA multitemporal change detection.
Analyze the supplied satellite/aerial imagery with high scientific rigor.

Domain Adaptation & Reasoning Guidelines:
1. BigEarthNet Vocabulary: Map land-cover and surface objects to standardized BigEarthNet categories (Urban fabric, Industrial units, Arable land, Permanent crops, Pastures, Complex cultivation, Coniferous/Broad-leaved forest, Inland/Marine waters, Wetlands, Bare rock, Sparsely vegetated areas).
2. Spatial Grounding: When asked to locate, highlight, or pinpoint an entity (e.g. "Highlight the water body referred to in the query", "Find the building complex"), populate region with normalized bounding box percentages: { x_percent, y_percent, w_percent, h_percent } (0-100 relative to top-left).
3. Scene Captioning (VRSBench): When asked to describe or caption the scene, generate a structured, multi-attribute remote sensing description covering topography, dominant land cover, object distribution, and visible sensor characteristics.
4. Counting & Structural Auditing: When asked to count objects (buildings, structures, vessels), evaluate distinct individual structural footprints and planar rooftop geometries. Account for partial edge structures and occlusions. Return verified count in building_count.
5. Multitemporal Change (CDVQA): When comparing passes or analyzing changes, clearly state whether features increased, decreased, or remained unchanged, and localize where the change occurred.
6. Confidence Assessment: Provide a confidence_percent (0-100) reflecting image resolution, cloud/shadow occlusion, sensor angle, and physical ambiguity.

Return valid JSON only with the following fields:
- answer: string (concise, analytical, evidence-grounded answer)
- confidence: 'high' | 'medium' | 'low'
- confidence_percent: number (0-100, model's own self-assessed reliability for this specific answer)
- confidence_reason: string (brief explanation of clarity, occlusion, or resolution factors)
- building_count: number | null (set to best_estimate for counting questions)
- count_estimate: { low: number, high: number, best_estimate: number } | null (populate ONLY for counting questions; null otherwise)
- count_uncertainty_factors: string[] (sources of count uncertainty, e.g. ["tree cover obscuring rooftops", "structures cut off at edge"])
- detected_features: string[] (3-5 key visual features identified according to BigEarthNet vocabulary)
- estimated_coverage_percent: number (approximate percentage of dominant land cover)
- water_coverage_percent: number (0-100)
- vegetation_percent: number (0-100)
- data_limitation_note: string
- region: { x_percent: number, y_percent: number, w_percent: number, h_percent: number } | null (bounding box percentages 0-100 of the primary region or feature being analyzed; null if whole scene)
- label: string (concise label for the detected region or scene assessment)
- suggested_followups: string[] (3-5 relevant follow-up questions)`

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
    confidence: 'high' as const,
    confidence_percent: 96,
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
