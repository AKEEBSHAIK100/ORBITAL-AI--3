const DEFAULT_WORKER_URL = 'https://cattolatte-satquery.hf.space'

function workerUrl(): string {
  return (process.env.HF_RS_WORKER_URL || DEFAULT_WORKER_URL).replace(/\/$/, '')
}

function decodeDataUrl(value: string): { mime: string; bytes: Buffer } {
  const match = value.match(/^data:(image\/(?:jpeg|png|webp|gif));base64,([A-Za-z0-9+/=]+)$/)
  if (!match) throw new Error('Image must be a supported image data URL.')
  return { mime: match[1], bytes: Buffer.from(match[2], 'base64') }
}

async function uploadImage(dataUrl: string, filename: string): Promise<string> {
  const { mime, bytes } = decodeDataUrl(dataUrl)
  const form = new FormData()
  form.append('files', new Blob([bytes], { type: mime }), filename)

  const response = await fetch(`${workerUrl()}/gradio_api/upload`, {
    method: 'POST',
    body: form,
    signal: AbortSignal.timeout(15_000),
  })
  if (!response.ok) throw new Error(`Worker upload failed: HTTP ${response.status}`)
  const payload = await response.json()
  const path = Array.isArray(payload) ? payload[0] : payload?.path
  if (!path) throw new Error('Worker upload returned no file path.')
  return String(path)
}

async function callGradio(apiName: string, data: unknown[]): Promise<unknown> {
  const response = await fetch(`${workerUrl()}/gradio_api/call/${apiName}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data }),
    signal: AbortSignal.timeout(20_000),
  })
  if (!response.ok) throw new Error(`Worker call failed: HTTP ${response.status}`)
  const started = await response.json()
  const eventId = started?.event_id
  if (!eventId) throw new Error('Worker call returned no event id.')

  const stream = await fetch(`${workerUrl()}/gradio_api/call/${apiName}/${encodeURIComponent(eventId)}`, {
    signal: AbortSignal.timeout(130_000),
  })
  if (!stream.ok) throw new Error(`Worker result stream failed: HTTP ${stream.status}`)

  const text = await stream.text()
  const events = text.split(/\n\n+/)
  for (const block of events) {
    const event = block.match(/^event:\s*(\S+)/m)?.[1]
    const raw = block.match(/^data:\s*(.*)$/m)?.[1]
    if (!raw) continue
    if (event === 'error') throw new Error(raw)
    if (event === 'complete') {
      const parsed = JSON.parse(raw)
      return Array.isArray(parsed) && parsed.length === 1 ? parsed[0] : parsed
    }
  }
  throw new Error('Worker stream ended without a complete result.')
}

function fileRef(path: string, filename: string) {
  return {
    path,
    orig_name: filename,
    meta: { _type: 'gradio.FileData' },
  }
}

export async function runWorkerVqaOrCaption(image: string, query: string) {
  const path = await uploadImage(image, 'orbital-input.jpg')
  const raw = await callGradio('answer', [query, fileRef(path, 'orbital-input.jpg'), null])
  const values = Array.isArray(raw) ? raw : [raw]
  return { ok: true, answer: String(values[0] ?? ''), evidence: values[1] ?? '', trace: values[2] ?? '', task: /describe|caption|scene/i.test(query) ? 'caption' : 'vqa', model: 'external-zero-gpu-rs-vlm', adapter: 'external-space', provenance: 'External public Hugging Face ZeroGPU Space; not an ORBITAL-AI benchmark.' }
}

export async function runWorkerChange(beforeImage: string, afterImage: string) {
  const before = await uploadImage(beforeImage, 'orbital-before.jpg')
  const after = await uploadImage(afterImage, 'orbital-after.jpg')
  const raw = await callGradio('answer', ['What changed between these two observations, and where did the change occur?', fileRef(before, 'orbital-before.jpg'), fileRef(after, 'orbital-after.jpg')])
  const values = Array.isArray(raw) ? raw : [raw]
  return { ok: true, answer: String(values[0] ?? ''), evidence: values[1] ?? '', trace: values[2] ?? '', method: 'external_zero_gpu_rs_vlm', note: 'External public ZeroGPU specialist; no calibrated quantitative change fraction is claimed.' }
}

export async function runWorkerFusion(opticalImage: string, sarImage: string) {
  const optical = await uploadImage(opticalImage, 'orbital-optical.jpg')
  const sar = await uploadImage(sarImage, 'orbital-sar.jpg')
  const raw = await callGradio('answer', ['Use the optical and SAR images together to identify built-up and water-covered regions.', fileRef(optical, 'orbital-optical.jpg'), fileRef(sar, 'orbital-sar.jpg')])
  const values = Array.isArray(raw) ? raw : [raw]
  return { ok: true, answer: String(values[0] ?? ''), evidence: values[1] ?? '', trace: values[2] ?? '', method: 'external_zero_gpu_rs_vlm', note: 'External public ZeroGPU specialist; optical/SAR semantic result is not independently benchmarked by ORBITAL-AI.' }
}

export function isWorkerConfigured(): boolean {
  return Boolean(process.env.HF_RS_WORKER_URL?.trim()) || process.env.ENABLE_HF_RS_WORKER === 'true'
}
