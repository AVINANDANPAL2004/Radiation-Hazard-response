import React from 'react'
import { N, heat } from '../shared.js'

// SVG map of one scenario at frame t for one agent run
export default function Grid({ scen, run, t, color = '#4da3ff', showTrail = true, showQ = false }) {
  const fi = Math.min(t, run.frames.length - 1)
  const fr = run.frames[fi]
  const ri = Math.min(t, scen.rates.length - 1)
  const rates = scen.rates[ri]
  const sources = scen.sources[ri]
  const trail = run.frames.slice(0, fi + 1).map((f) => `${f.pos[1] * 10 + 5},${f.pos[0] * 10 + 5}`).join(' ')
  const cx = (c) => c * 10 + 5

  return (
    <svg viewBox="0 0 100 100" className="grid-svg" role="img" aria-label="Facility map">
      <defs>
        <pattern id="hatch" width="2" height="2" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="2" height="2" fill="#0b0e13" />
          <line x1="0" y1="0" x2="0" y2="2" stroke="#2a313c" strokeWidth="0.8" />
        </pattern>
        <filter id="glow"><feGaussianBlur stdDeviation="1.2" /></filter>
      </defs>
      {rates.map((r, i) => {
        const row = Math.floor(i / N), col = i % N
        return <rect key={i} x={col * 10} y={row * 10} width="10" height="10" fill={heat(r)} stroke="#0e1117" strokeWidth="0.25" />
      })}
      {scen.walls.map((w, i) => w ? <rect key={'w' + i} x={(i % N) * 10 + 0.4} y={Math.floor(i / N) * 10 + 0.4} width="9.2" height="9.2" rx="1" fill="url(#hatch)" stroke="#3a424f" strokeWidth="0.4" /> : null)}

      {sources.map(([r, c], k) => (
        <g key={k}>
          <circle cx={cx(c)} cy={cx(r)} r="4.2" fill="none" stroke="#ffd166" strokeWidth="0.5" strokeDasharray="1.2 1.2">
            <animate attributeName="r" values="3.4;5;3.4" dur="2s" repeatCount="indefinite" />
          </circle>
          <text x={cx(c)} y={cx(r) + 2} fontSize="5.5" textAnchor="middle" fill="#ffd166">☢</text>
        </g>
      ))}

      <text x={cx(scen.goal[1])} y={cx(scen.goal[0]) + 2.4} fontSize="7" textAnchor="middle" fill="#5dffa0">★</text>
      <text x={cx(scen.start[1])} y={cx(scen.start[0]) + 2.2} fontSize="6" textAnchor="middle" fill="#67e8f9">◆</text>

      {showTrail && fi > 0 && <polyline points={trail} fill="none" stroke={color} strokeWidth="1.1" strokeLinejoin="round" strokeLinecap="round" opacity="0.9" />}
      <circle cx={cx(fr.pos[1])} cy={cx(fr.pos[0])} r="3.6" fill={color} opacity="0.5" filter="url(#glow)" />
      <circle cx={cx(fr.pos[1])} cy={cx(fr.pos[0])} r="2.3" fill={color} stroke="#fff" strokeWidth="0.6" />
    </svg>
  )
}
