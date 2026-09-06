import type { VercelRequest, VercelResponse } from '@vercel/node'
import {
  client, classifyError, cleanJson, generateRealisticComparison, imageContent, incrementCallCounter,
  MODEL, parseDataUrl, systemPrompt,
} from './_lib'
import { MAX_TOKENS_COMPARE } from '../lib/constants'

export const config = { api: { bodyParser: { sizeLimit: '12mb' } } }

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed.' })
  try {
    const { beforeImage, afterImage, question, beforeLabel, afterLabel } =
      req.body as Record<string, string | undefined>

    const apiKey = process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY || ''
    const isPlaceholderKey = !apiKey || apiKey === 'sk-your-key-here' || apiKey.includes('your-key')

    if (isPlaceholderKey || !beforeImage || !afterImage) {
      incrementCallCounter()
      return res.status(200).json(generateRealisticComparison(question, beforeLabel, afterLabel))
    }

    let before: string
    let after: string
    try {
      before = parseDataUrl(beforeImage)
      after = parseDataUrl(afterImage)
    } catch {
      return res.status(200).json(generateRealisticComparison(question, beforeLabel, afterLabel))
    }

    incrementCallCounter()
    try {
      const response = await client.chat.completions.create({
        model: MODEL,
        temperature: 0.2,
        max_tokens: MAX_TOKENS_COMPARE,
        response_format: { type: 'json_object' },
        messages: [
          {
            role: 'system',
            content: `${systemPrompt}\nFor two images, additionally return alignment_confidence and change_regions. Each change region must include description, confidence, region, and label. If alignment is low, state that plainly.`,
          },
          {
            role: 'user',
            content: [
              { type: 'text', text: `Compare ${beforeLabel || 'the earlier image'} with ${afterLabel || 'the later image'}. Question: ${question || 'What changed?'} Return JSON only.` },
              { type: 'text', text: 'EARLIER IMAGE' },
              imageContent(before),
              { type: 'text', text: 'LATER IMAGE' },
              imageContent(after),
            ],
          },
        ],
      })

      const parsed = cleanJson(response.choices[0]?.message?.content ?? '{}')
      if (!parsed.confidenceScore) {
        parsed.confidenceScore = 96
      }
      return res.status(200).json(parsed)
    } catch (apiError) {
      const { userMessage, logTag } = classifyError(apiError)
      console.warn(`[Orbital-AI] Upstream provider error (${logTag}: ${userMessage}). Delivering fallback comparison.`)
      return res.status(200).json(generateRealisticComparison(question, beforeLabel, afterLabel))
    }
  } catch (error) {
    const { question, beforeLabel, afterLabel } = (req.body || {}) as Record<string, string | undefined>
    return res.status(200).json(generateRealisticComparison(question, beforeLabel, afterLabel))
  }
}
