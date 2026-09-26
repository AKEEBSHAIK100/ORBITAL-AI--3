const SPACE = 'cattolatte/satquery'
const ENDPOINT = '/answer'
const CLIENT_URL = 'https://cdn.jsdelivr.net/npm/@gradio/client@2.7.0/dist/index.min.js'

type GradioStatus = { status?: string; load_status?: string; message?: string }

type GradioClient = {
  predict: (endpoint: string, data: unknown[]) => Promise<{ data?: unknown[] }>
}

type GradioModule = {
  Client: {
    connect: (
      source: string,
      options?: { status_callback?: (status: GradioStatus) => void },
    ) => Promise<GradioClient>
  }
  handle_file: (file: Blob) => unknown
}

let modulePromise: Promise<GradioModule> | null = null
let clientPromise: Promise<GradioClient> | null = null

async function loadGradio(): Promise<GradioModule> {
  if (!modulePromise) {
    modulePromise = (async () => {
      // Keep the main Vite bundle lean; load the pinned client only when analysis is requested.
      // @ts-expect-error The module is intentionally loaded from a pinned CDN URL.
      return (await import(/* @vite-ignore */ CLIENT_URL)) as GradioModule
    })()
  }
  return modulePromise
}

async function getClient(onStatus?: (message: string) => void): Promise<GradioClient> {
  if (!clientPromise) {
    clientPromise = (async () => {
      const { Client } = await loadGradio()
      return Client.connect(SPACE, {
        status_callback: (status) => {
          const phase = status?.load_status || status?.status
          if (phase === 'pending' || phase === 'generating' || phase === 'running') {
            onStatus?.('Waking remote vision model…')
          } else if (phase === 'complete') {
            onStatus?.('Remote vision model is ready.')
          } else if (status?.message) {
            onStatus?.(status.message)
          }
        },
      })
    })()
  }
  return clientPromise
}

async function dataUrlToBlob(dataUrl: string): Promise<Blob> {
  const response = await fetch(dataUrl)
  if (!response.ok) throw new Error('Could not prepare the uploaded image for remote analysis.')
  return response.blob()
}

function asText(value: unknown): string {
  if (typeof value === 'string') return value.trim()
  if (value == null) return ''
  return JSON.stringify(value)
}

/**
 * Executes the same public remote-sensing Space used by the server fallback,
 * but from the browser. This avoids a Vercel serverless timeout while the
 * ZeroGPU model is waking and generating. No client-side heuristic is used.
 */
export async function runBrowserRemoteAnalysis(
  question: string,
  imageA: string,
  imageB?: string,
  onStatus?: (message: string) => void,
): Promise<{ answer: string; evidence: string; trace: string }> {
  if (!imageA) throw new Error('A primary image is required.')
  onStatus?.('Connecting to remote-sensing specialist…')

  const [{ blob: blobA, handle_file }, blobB] = await Promise.all([
    dataUrlToBlob(imageA).then(async blob => ({ blob, handle_file: (await loadGradio()).handle_file })),
    imageB ? dataUrlToBlob(imageB) : Promise.resolve(null),
  ])

  const client = await getClient(onStatus)
  onStatus?.(imageB ? 'Running two-image remote-sensing analysis…' : 'Running remote-sensing VQA…')

  const result = await client.predict(ENDPOINT, [
    question,
    handle_file(blobA),
    blobB ? handle_file(blobB) : null,
  ])

  const data = Array.isArray(result?.data) ? result.data : []
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
