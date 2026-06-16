import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  publicDir: 'public',
  server: {
    hmr: false,
    proxy: {
      '/chat/assistant': { target: 'http://localhost:8002', changeOrigin: true },
      '/chat':           { target: 'http://localhost:8000', changeOrigin: true },
      '/report':         { target: 'http://localhost:8000', changeOrigin: true },
      '/health':         { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
