import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' lets the built site be served from any folder
export default defineConfig({
  base: './',
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
})
