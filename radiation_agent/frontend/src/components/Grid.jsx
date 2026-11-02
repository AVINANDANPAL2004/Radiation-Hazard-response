import React, { useId } from 'react'
import { N, heat } from '../shared.js'

// Mirrors env.py: sensors read a (2*SENSE_R+1)^2 window around the agent; the shield
// blocks neighbours whose dose exceeds R_HARD. Mask order matches ACTIONS[:4] in env.py.
const SENSE_R = 2
const RATE_MAX = 7 // RATE_NORM in env.py: rates at or above this render at the hottest colour
const MOVES = [[-1, 0], [1, 0], [0, -1], [0, 1]] // up, down, left, right
const prefersReducedMotion = () =>
  typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

// Colour key for the dose-rate background. Sits under the map and is hidden from screen readers.
export function DoseLegend({ max = RATE_MAX }) {
  const stops = [0, 0.15, 0.3, 0.5, 0.7, 0.85, 1].map((f) => `${heat(f * max)} ${Math.round(f * 100)}%`)
  return (
    <div className="dose-legend" aria-hidden="true">
      <span className="mono small dim">0</span>
      <div className="dose-bar" style={{ background: `linear-gradient(90deg, ${stops.join(', ')})` }} />
      <span className="mono small dim">{max}+</span>
      <span className="small dim">dose rate</span>
    </div>
  )
}

// SVG map of one scenario at frame t for one agent run.
// show: { trail, sense, blocked } overlays. compact: smaller strokes, no overlays (used in Compare).
export default function Grid({ scen, run, t, color = '#4da3ff', compact = false, show = {} }) {
  const { trail = true, sense = true, blocked = true } = show
  const uid = useId().replace(/:/g, '')
  const hatch = `hatch-${uid}`, glow = `glow-${uid}`, arrow = `arrow-${uid}`

  const fi = Math.min(t, run.frames.length - 1)
  const fr = run.frames[fi]
  const ri = Math.min(t, scen.rates.length - 1)
  const rates = scen.rates[ri]
  const sources = scen.sources[ri]
  const [pr, pc] = fr.pos
  const reduce = prefersReducedMotion()
  const cx = (c) => c * 10 + 5
  const isWall = (r, c) => scen.walls[r * N + c] === 1
  const sw = compact ? 0.8 : 1.1
  const showOverlays = !compact

  // trail: segments that fade from the start (faint) to the current step (solid)
  const segs = []
  if (trail && fi > 0) {
    for (let i = 0; i < fi; i++) {
      const [r0, c0] = run.frames[i].pos
      const [r1, c1] = run.frames[i + 1].pos
      segs.push(
        <line key={i} x1={cx(c0)} y1={cx(r0)} x2={cx(c1)} y2={cx(r1)} stroke={color} strokeWidth={sw}
          strokeLinecap="round" opacity={0.12 + 0.88 * ((i + 1) / fi)} />
      )
    }
  }

  // sensing window: the square of cells the agent's sensors can read this step
  const sr0 = Math.max(0, pr - SENSE_R), sr1 = Math.min(N, pr + SENSE_R + 1)
  const sc0 = Math.max(0, pc - SENSE_R), sc1 = Math.min(N, pc + SENSE_R + 1)

  // neighbours the shield forbids (in bounds, not walls): these are the "blocked" moves
  const blockedCells = []
  if (showOverlays && blocked && fr.mask) {
    MOVES.forEach(([dr, dc], i) => {
      const nr = pr + dr, nc = pc + dc
      if (fr.mask[i] || nr < 0 || nc < 0 || nr >= N || nc >= N || isWall(nr, nc)) return
      blockedCells.push([nr, nc])
    })
  }

  return (
    <svg viewBox="0 0 100 100" className="grid-svg" role="img"
      aria-label={`Facility map at step ${fi}, dose ${fr.exposure.toFixed(2)}`}>
      <defs>
        <pattern id={hatch} width="2" height="2" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="2" height="2" fill="#0b0e13" />
          <line x1="0" y1="0" x2="0" y2="2" stroke="#2a313c" strokeWidth="0.8" />
        </pattern>
        <filter id={glow}><feGaussianBlur stdDeviation="1.2" /></filter>
        <marker id={arrow} viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4" markerHeight="4" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#ffd166" />
        </marker>
      </defs>

      {rates.map((r, i) => {
        const row = Math.floor(i / N), col = i % N
        return (
          <rect key={i} x={col * 10} y={row * 10} width="10" height="10" fill={heat(r)} stroke="#0e1117" strokeWidth="0.25">
            <title>{`row ${row}, col ${col}: dose rate ${r.toFixed(2)}`}</title>
          </rect>
        )
      })}

      {scen.walls.map((w, i) => w ? (
        <rect key={'w' + i} x={(i % N) * 10 + 0.4} y={Math.floor(i / N) * 10 + 0.4} width="9.2" height="9.2" rx="1"
          fill={`url(#${hatch})`} stroke="#3a424f" strokeWidth="0.4" />
      ) : null)}

      {blockedCells.map(([r, c]) => (
        <g key={`b${r}-${c}`}>
          <rect x={c * 10 + 1} y={r * 10 + 1} width="8" height="8" rx="1" fill="rgba(255,107,94,0.12)"
            stroke="#ff6b5e" strokeWidth="0.6" strokeDasharray="1 1" />
          <text x={cx(c)} y={cx(r) + 1.8} fontSize="5" textAnchor="middle" fill="#ff6b5e">✕</text>
        </g>
      ))}

      {showOverlays && sense && (
        <rect x={sc0 * 10} y={sr0 * 10} width={(sc1 - sc0) * 10} height={(sr1 - sr0) * 10} rx="1"
          fill="none" stroke={color} strokeWidth="0.5" strokeDasharray="1.5 1.5" opacity="0.75" />
      )}

      {sources.map((src, k) => {
        const [r, c] = src
        const heading = src[3]
        const moving = heading && (heading[0] || heading[1])
        return (
          <g key={k}>
            <circle cx={cx(c)} cy={cx(r)} r="4.2" fill="none" stroke="#ffd166" strokeWidth="0.5" strokeDasharray="1.2 1.2">
              {!reduce && <animate attributeName="r" values="3.4;5;3.4" dur="2s" repeatCount="indefinite" />}
            </circle>
            <text x={cx(c)} y={cx(r) + 2} fontSize="5.5" textAnchor="middle" fill="#ffd166">☢</text>
            {showOverlays && moving && (
              <line x1={cx(c)} y1={cx(r)} x2={cx(c) + heading[1] * 8} y2={cx(r) + heading[0] * 8}
                stroke="#ffd166" strokeWidth="0.7" markerEnd={`url(#${arrow})`} opacity="0.9" />
            )}
          </g>
        )
      })}

      <text x={cx(scen.goal[1])} y={cx(scen.goal[0]) + 2.4} fontSize="7" textAnchor="middle" fill="#5dffa0">★</text>
      <text x={cx(scen.start[1])} y={cx(scen.start[0]) + 2.2} fontSize="6" textAnchor="middle" fill="#67e8f9">◆</text>

      {segs}

      <circle cx={cx(pc)} cy={cx(pr)} r="3.6" fill={color} opacity="0.5" filter={`url(#${glow})`} />
      <circle cx={cx(pc)} cy={cx(pr)} r="2.3" fill={color} stroke="#fff" strokeWidth="0.6" />
      {!reduce && fi < run.frames.length - 1 && (
        <circle cx={cx(pc)} cy={cx(pr)} r="2.3" fill="none" stroke={color} strokeWidth="0.4">
          <animate attributeName="r" values="2.6;5;2.6" dur="1.6s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.8;0;0.8" dur="1.6s" repeatCount="indefinite" />
        </circle>
      )}
    </svg>
  )
}
