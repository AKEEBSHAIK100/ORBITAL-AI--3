import { createServer } from 'vite'
import http from 'node:http'
import { spawn } from 'node:child_process'

const isWin = process.platform === 'win32'

// 1. Programmatically launch Vite on port 8443
console.log('[Orbital-AI] Launching Vite dev server on port 8443...')
const viteServer = await createServer({
  server: { host: '0.0.0.0', port: 8443 }
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

// 3. Fallback HTTP bridge on port 5173 for browsers that restrict 8443
const bridge = http.createServer((req, res) => {
  const isApi = req.url && req.url.startsWith('/api')
  const targetPort = isApi ? 8787 : 8443
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
}
process.on('SIGINT', shutdown)
process.on('SIGTERM', shutdown)
