const SPACE = 'cattolatte/satquery'
const ENDPOINT = '/answer'
const CLIENT_URL = 'https://cdn.jsdelivr.net/npm/@gradio/client@2.7.0/dist/index.min.js'

type GradioClient = {
  predict: (endpoint: string, data: unknown[]) => Promise<{ data?: unknown[] }>
}

type GradioModule = {
  Client: {
    connect: (
      source: string,
      options?: { status_callback?: (status: { status?: string; load_status?: string; message?: string }) => void },
    ) => Promise<GradioClient>
  }
  handle_file: (file: Blob) => unknown
}

let clientPromise: Promise<GradioClient> | null = null

async function getClient(onStatus?: (message: string) => void): Promise<GradioClient> {
  if (!clientPromise) {
    clientPromise = (async () => {
      // The Gradio client is loaded at runtime so the main Vite bundle stays lean.
      // @ts-expect-error The module is intentionally loaded from a pinned CDN URL.
      const mod = (await import(/* @vite-ignore */ CLIENT_URL)) as GradioModule
      return mod.Client.connect(SPACE, {
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
 * Executes the same public remote-sensing Space already used by the server fallback,
 * but from the browser. This prevents a Vercel serverless timeout while the ZeroGPU
 * model is waking and generating. No client-side heuristic is used.
 */
export async function runBrowserRemoteAnalysis(
  question: string,
  imageA: string,
  imageB?: string,
  onStatus?: (message: string) => void,
): Promise<{ answer: string; evidence: string; trace: string }> {
  if (!imageA) throw new Error('A primary image is required.')
  onStatus?.('Connecting to remote-sensing specialist…')

  const [blobA, blobB] = await Promise.all([
    dataUrlToBlob(imageA),
    imageB ? dataUrlToBlob(imageB) : Promise.resolve(null),
  ])

  const client = await getClient(onStatus)
  onStatus?.(imageB ? 'Running two-image remote-sensing analysis…' : 'Running remote-sensing VQA…')

  const result = await client.predict(ENDPOINT, [
    question,
    (await import(/* @vite-ignore */ CLIENT_URL) as GradioModule).handle_file(blobA),
    blobB ? (await import(/* @vite-ignore */ CLIENT_URL) as GradioModule).handle_file(blobB) : null,
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
