import { createServer } from 'vite'
import http from 'node:http'
import { spawn } from 'node:child_process'

const isWin = process.platform === 'win32'

// 1. Programmatically launch Vite on port 8443
console.log('[Orbital-AI] Launching Vite dev server on port 8443...')
const viteServer = await createServer({
  server: {
    host: '0.0.0.0',
    port: 8443,
    proxy: {
      '/api': 'http://127.0.0.1:8787',
      '/analyze': 'http://127.0.0.1:8000',
      '/classify': 'http://127.0.0.1:8000',
    },
  }
})
await viteServer.listen()
viteServer.printUrls()

// 2. Launch Express API on port 8787
console.log('[Orbital-AI] Launching API server on port 8787...')
const apiChild = spawn(isWin ? 'npx.cmd' : 'npx', ['tsx', 'watch', 'server.ts'], {
  stdio: 'inherit',
  env: { ...process.env, API_PORT: '8787' },
  shell: isWin,
})

apiChild.on('exit', code => {
  if (code && code !== 0) console.error(`[Orbital-AI API] exited with code ${code}`)
})

// 3. Launch FastAPI Building Detection backend on port 8000
console.log('[Orbital-AI] Launching FastAPI Building Detection service on port 8000...')
const pyChild = spawn('python', ['-m', 'uvicorn', 'backend.main:app', '--host', '0.0.0.0', '--port', '8000'], {
  stdio: 'inherit',
  shell: isWin,
})

pyChild.on('exit', code => {
  if (code && code !== 0) console.error(`[Orbital-AI Python Backend] exited with code ${code}`)
})

// 4. Fallback HTTP bridge on port 5173 for browsers that restrict 8443
const bridge = http.createServer((req, res) => {
  const isApi = req.url && (req.url.startsWith('/api') || req.url.startsWith('/analyze') || req.url.startsWith('/classify'))
  const isPythonAnalyze = req.url && req.url.startsWith('/analyze')
  const targetPort = isPythonAnalyze ? 8000 : isApi ? 8787 : 8443
  const connector = http.request(
    {
      hostname: '127.0.0.1',
      port: targetPort,
      path: req.url,
      method: req.method,
      headers: { ...req.headers, host: `localhost:${targetPort}` },
    },
    targetRes => {
      res.writeHead(targetRes.statusCode ?? 200, targetRes.headers)
      targetRes.pipe(res, { end: true })
    }
  )
  connector.on('error', () => {
    res.writeHead(502, { 'Content-Type': 'text/plain' })
    res.end('Connecting to Orbital-AI dev server...')
  })
  req.pipe(connector, { end: true })
})

bridge.on('error', () => {})
bridge.listen(5173, '0.0.0.0', () => {
  console.log('🌐 Seamless HTTP fallback bridge active on http://localhost:5173')
})

const shutdown = async () => {
  bridge.close()
  await viteServer.close()
  apiChild.kill('SIGTERM')
  pyChild.kill('SIGTERM')
}
process.on('SIGINT', shutdown)
process.on('SIGTERM', shutdown)
