const SPACE_URL = 'https://cattolatte-satquery.hf.space'
const ANSWER_ENDPOINT = '/gradio_api/call/answer'

type FileRef = {
  path: string
  orig_name: string
  meta: { _type: 'gradio.FileData' }
}

function fileRef(path: string, filename: string): FileRef {
  return { path, orig_name: filename, meta: { _type: 'gradio.FileData' } }
}

async function dataUrlToBlob(dataUrl: string): Promise<Blob> {
  const response = await fetch(dataUrl)
  if (!response.ok) throw new Error('Could not prepare the uploaded image for remote analysis.')
  return response.blob()
}

async function uploadImage(blob: Blob, filename: string): Promise<string> {
  const form = new FormData()
  form.append('files', blob, filename)
  const response = await fetch(`${SPACE_URL}/gradio_api/upload`, {
    method: 'POST',
    body: form,
  })
  if (!response.ok) throw new Error(`Remote specialist upload failed (HTTP ${response.status}).`)
  const payload = await response.json()
  const path = Array.isArray(payload) ? payload[0] : payload?.path
  if (!path) throw new Error('Remote specialist returned no uploaded-file reference.')
  return String(path)
}

async function startJob(data: unknown[]): Promise<string> {
  const response = await fetch(`${SPACE_URL}${ANSWER_ENDPOINT}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data }),
  })
  if (!response.ok) throw new Error(`Remote specialist queue request failed (HTTP ${response.status}).`)
  const payload = await response.json()
  if (!payload?.event_id) throw new Error('Remote specialist returned no job id.')
  return String(payload.event_id)
}

async function readJob(eventId: string, onStatus?: (message: string) => void): Promise<unknown[]> {
  const response = await fetch(`${SPACE_URL}${ANSWER_ENDPOINT}/${encodeURIComponent(eventId)}`)
  if (!response.ok) throw new Error(`Remote specialist result request failed (HTTP ${response.status}).`)

  const text = await response.text()
  const blocks = text.split(/\n\n+/)

  for (const block of blocks) {
    const event = block.match(/^event:\s*(\S+)/m)?.[1]
    const raw = block.match(/^data:\s*(.*)$/m)?.[1]
    if (!raw) continue

    if (event === 'heartbeat' || event === 'generating') {
      onStatus?.('Remote GPU specialist is processing the image…')
      continue
    }
    if (event === 'error') {
      throw new Error(raw)
    }
    if (event === 'complete') {
      const parsed = JSON.parse(raw)
      return Array.isArray(parsed) ? parsed : [parsed]
    }
  }

  throw new Error('Remote specialist ended without a completed result.')
}

function asText(value: unknown): string {
  if (typeof value === 'string') return value.trim()
  if (value == null) return ''
  return JSON.stringify(value)
}

/**
 * Executes the public SatQuery ZeroGPU specialist directly from the browser.
 * This avoids Vercel's serverless request timeout while the GPU Space wakes.
 * No client-side heuristic or fabricated measurement is used.
 */
export async function runBrowserRemoteAnalysis(
  question: string,
  imageA: string,
  imageB?: string,
  onStatus?: (message: string) => void,
): Promise<{ answer: string; evidence: string; trace: string }> {
  if (!imageA) throw new Error('A primary image is required.')

  onStatus?.('Connecting to remote-sensing GPU specialist…')

  const [blobA, blobB] = await Promise.all([
    dataUrlToBlob(imageA),
    imageB ? dataUrlToBlob(imageB) : Promise.resolve(null),
  ])

  const [pathA, pathB] = await Promise.all([
    uploadImage(blobA, 'orbital-image-a.jpg'),
    blobB ? uploadImage(blobB, 'orbital-image-b.jpg') : Promise.resolve(null),
  ])

  onStatus?.('Remote specialist is waking or entering the GPU queue…')

  const eventId = await startJob([
    question,
    fileRef(pathA, 'orbital-image-a.jpg'),
    pathB ? fileRef(pathB, 'orbital-image-b.jpg') : null,
  ])

  const data = await readJob(eventId, onStatus)
  const answer = asText(data[0])

  if (!answer || /^error\s*:/i.test(answer)) {
    throw new Error(answer || 'The remote-sensing specialist returned no answer.')
  }

  return {
    answer,
    evidence: asText(data[1]),
    trace: asText(data[2]),
  }
}
