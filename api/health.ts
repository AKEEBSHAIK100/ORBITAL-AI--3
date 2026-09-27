import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getTotalCalls, MODEL } from './_lib.js'

export default function handler(_req: VercelRequest, res: VercelResponse) {
  res.status(200).json({
    ok: true,
    provider: 'remote-sensing-specialists',
    model: MODEL,
    configured: Boolean(process.env.PYTHON_BACKEND_URL || process.env.ENABLE_HF_RS_WORKER === 'true'),
    totalCallsThisDeployment: getTotalCalls(),
  })
}
