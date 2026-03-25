import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendTarget = env.VITE_DEV_API_TARGET || 'http://127.0.0.1:8000'

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/health': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/ingest': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/chat': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/retrieve': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/traces': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/eval': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/diagnose': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/docs': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/openapi.json': {
          target: backendTarget,
          changeOrigin: true,
        },
      },
    },
  }
})
