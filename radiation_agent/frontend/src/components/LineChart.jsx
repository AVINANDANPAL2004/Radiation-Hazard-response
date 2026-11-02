import React from 'react'

// series: [{name,color,x[],mean[],std[]}]
export default function LineChart({ series, yLabel, yMin, yMax, height = 260, pct = false }) {
  const W = 600, H = height, m = { l: 46, r: 12, t: 10, b: 30 }
  const xs = series.flatMap((s) => s.x)
  const xMax = Math.max(...xs)
  const lo = yMin ?? Math.min(...series.flatMap((s) => s.mean.map((v, i) => v - (s.std?.[i] || 0))))
  const hi = yMax ?? Math.max(...series.flatMap((s) => s.mean.map((v, i) => v + (s.std?.[i] || 0))))
  const X = (v) => m.l + (v / xMax) * (W - m.l - m.r)
  const Y = (v) => H - m.b - ((v - lo) / (hi - lo || 1)) * (H - m.t - m.b)
  const yTicks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4)
  const xTicks = Array.from({ length: 5 }, (_, i) => Math.round((xMax * i) / 4))
  const fmt = (v) => (pct ? `${Math.round(v * 100)}%` : Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1))

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={yLabel}>
      {yTicks.map((v, i) => (
        <g key={i}>
          <line x1={m.l} x2={W - m.r} y1={Y(v)} y2={Y(v)} stroke="#222a35" />
          <text x={m.l - 6} y={Y(v) + 3} textAnchor="end" className="tick">{fmt(v)}</text>
        </g>
      ))}
      {xTicks.map((v, i) => (
        <text key={i} x={X(v)} y={H - m.b + 16} textAnchor="middle" className="tick">{v}</text>
      ))}
      <text x={(W + m.l) / 2} y={H - 3} textAnchor="middle" className="tick">episode</text>
      {series.map((s) => {
        const line = s.x.map((x, i) => `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(s.mean[i]).toFixed(1)}`).join('')
        const band = s.x.map((x, i) => `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(s.mean[i] + (s.std?.[i] || 0)).toFixed(1)}`).join('') +
          [...s.x].reverse().map((x, j) => { const i = s.x.length - 1 - j; return `L${X(x).toFixed(1)},${Y(s.mean[i] - (s.std?.[i] || 0)).toFixed(1)}` }).join('') + 'Z'
        return (
          <g key={s.name}>
            <path d={band} fill={s.color} opacity="0.14" />
            <path d={line} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" />
          </g>
        )
      })}
    </svg>
  )
}
