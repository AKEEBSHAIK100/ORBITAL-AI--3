import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getTotalCalls } from './_lib.js'
import { MODEL } from '../lib/constants.js'

export default function handler(_req: VercelRequest, res: VercelResponse) {
  res.status(200).json({
    ok: true,
    provider: 'remote-sensing-specialists',
    model: MODEL,
    configured: true,
    totalCallsThisDeployment: getTotalCalls(),
  })
}
