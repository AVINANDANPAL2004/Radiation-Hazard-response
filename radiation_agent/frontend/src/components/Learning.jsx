import React, { useState } from 'react'
import LineChart from './LineChart.jsx'
import { AGENTS } from '../shared.js'

const CHARTS = [
  { key: 'train_success', title: 'Training success rate', sub: '100-episode moving average', pct: true, yMin: 0, yMax: 1 },
  { key: 'train_reward', title: 'Training reward', sub: '100-episode moving average' },
  { key: 'eval_success', title: 'Greedy success on 20 fixed held-out scenarios', sub: 'checked every 100 episodes', pct: true, yMin: 0, yMax: 1 },
]

export default function Learning({ curves }) {
  const [on, setOn] = useState(Object.fromEntries(AGENTS.map((a) => [a.id, true])))
  const ckpts = curves.baseline.eval_success.x
  const total = ckpts[ckpts.length - 1]
  const decayEnd = Math.round(0.6 * total) // epsilon schedule in train.py decays over the first 60%
  return (
    <div className="stack">
      <section className="card">
        <div className="row wrap gap">
          <span className="dim">Show agents:</span>
          {AGENTS.map((a) => (
            <button key={a.id} className={`chip ${on[a.id] ? 'on' : ''}`} style={on[a.id] ? { borderColor: a.color, color: a.color } : null}
              onClick={() => setOn({ ...on, [a.id]: !on[a.id] })} aria-pressed={on[a.id]}>
              <i style={{ background: a.color }} /> {a.name}
            </button>
          ))}
        </div>
        <p className="dim small">Lines show the mean over 3 seeds, shaded bands show ± 1 std. Exploration (epsilon) decays over the first {decayEnd} of {total} episodes.</p>
      </section>
      <div className="grid2">
        {CHARTS.map((c) => (
          <section key={c.key} className={`card ${c.key === 'eval_success' ? 'span2' : ''}`}>
            <h3>{c.title}</h3>
            <p className="dim small">{c.sub}</p>
            <LineChart
              pct={c.pct} yMin={c.yMin} yMax={c.yMax} yLabel={c.title}
              series={AGENTS.filter((a) => on[a.id]).map((a) => ({ name: a.name, color: a.color, ...curves[a.id][c.key] }))}
            />
          </section>
        ))}
      </div>
    </div>
  )
}
