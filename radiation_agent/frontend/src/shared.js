import { useEffect, useRef, useState } from 'react'

export const AGENTS = [
  { id: 'baseline', name: 'Baseline DRL', short: 'Baseline', color: '#9aa4b2', desc: 'MLP on sensor vector only' },
  { id: 'cnn', name: 'CNN-DRL', short: 'CNN', color: '#4da3ff', desc: 'CNN on facility image + sensors' },
  { id: 'cnn_lstm', name: 'CNN-LSTM-DRL', short: 'CNN-LSTM', color: '#43d17a', desc: 'Adds LSTM memory over 4 frames' },
  { id: 'llm_cnn_lstm', name: 'Advisory-guided CNN-LSTM-DRL', short: 'Advisory', color: '#ff6b5e', desc: 'Adds features parsed from text advisories (rule-based parser)' },
]
export const AGENT = Object.fromEntries(AGENTS.map((a) => [a.id, a]))
export const N = 10
export const LIMIT = 1.0
export const ACTION_LABEL = ['↑ up', '↓ down', '← left', '→ right', '● wait']

export const OUTCOME = {
  success: { label: 'Reached goal', cls: 'ok' },
  overdose: { label: 'Dose limit exceeded', cls: 'bad' },
  timeout: { label: 'Timed out', cls: 'warn' },
  running: { label: 'Running', cls: 'run' },
}

// dose-rate (0..7) -> colour
export function heat(rate) {
  const v = Math.max(0, Math.min(1, rate / 7))
  const stops = [
    [0, [20, 26, 36]],
    [0.18, [58, 52, 34]],
    [0.4, [140, 92, 24]],
    [0.7, [214, 80, 36]],
    [1, [255, 60, 60]],
  ]
  for (let i = 1; i < stops.length; i++) {
    if (v <= stops[i][0]) {
      const [a, ca] = stops[i - 1]
      const [b, cb] = stops[i]
      const f = (v - a) / (b - a)
      return `rgb(${ca.map((c, k) => Math.round(c + (cb[k] - c) * f)).join(',')})`
    }
  }
  return 'rgb(255,60,60)'
}

// All data comes from the FastAPI backend (proxied to :8000 in dev, same origin in production)
const API = import.meta.env.VITE_API_BASE ?? ''
export const EVAL_SEED0 = 500000
export const N_EVAL = 200

export async function api(path) {
  let r
  try {
    r = await fetch(`${API}/api/${path}`)
  } catch {
    throw new Error('Cannot reach the backend. Start it with: uvicorn server:app --port 8000')
  }
  if (!r.ok) {
    let msg = ''
    try { msg = (await r.json()).detail } catch {}
    throw new Error(typeof msg === 'string' && msg ? msg : `Backend error ${r.status}`)
  }
  return r.json()
}

// shared playback clock: t = frame index
export function useClock(maxT, resetKey) {
  const [t, setT] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(4) // frames per second
  const maxRef = useRef(maxT)
  maxRef.current = maxT

  useEffect(() => {
    setT(0)
    setPlaying(false)
  }, [resetKey])

  // advance the clock; state updaters stay pure and the stop-at-end check lives in an effect
  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => setT((x) => Math.min(x + 1, maxRef.current)), 1000 / speed)
    return () => clearInterval(id)
  }, [playing, speed])

  useEffect(() => {
    if (playing && t >= maxRef.current) setPlaying(false)
  }, [t, playing])

  const toggle = () => {
    if (!playing && t >= maxRef.current) setT(0)
    setPlaying((p) => !p)
  }
  return { t, setT, playing, toggle, speed, setSpeed, reset: () => { setT(0); setPlaying(false) } }
}
