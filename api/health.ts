import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getTotalCalls, MODEL } from './_lib'

export default function handler(_req: VercelRequest, res: VercelResponse) {
  res.status(200).json({
    ok: true,
    provider: 'openai-compat',
    model: MODEL,
    configured: Boolean(process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY),
    totalCallsThisDeployment: getTotalCalls(),
  })
}
