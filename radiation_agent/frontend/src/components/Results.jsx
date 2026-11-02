import React from 'react'
import { AGENTS } from '../shared.js'

const ROWS = [
  { k: 'success', label: 'Task success rate', unit: '%', mult: 100, dp: 1, better: 'high' },
  { k: 'reward', label: 'Mean evaluation reward', unit: '', mult: 1, dp: 2, better: 'high' },
  { k: 'exposure', label: 'Mean dose (budget 1.0)', unit: '', mult: 1, dp: 2, better: 'low' },
  { k: 'overdose', label: 'Dose-limit failures', unit: '%', mult: 100, dp: 1, better: 'low' },
  { k: 'travel_steps', label: 'Travel time (steps, successes)', unit: '', mult: 1, dp: 1, better: 'low' },
  { k: 'interventions', label: 'Shield interventions / episode', unit: '', mult: 1, dp: 1, better: 'low' },
  { k: 'latency_ms', label: 'Inference latency (ms / decision)', unit: '', mult: 1, dp: 2, better: 'low' },
]

export default function Results({ summary }) {
  const best = (r) => {
    const vals = AGENTS.map((a) => summary[a.id][r.k].mean)
    return r.better === 'high' ? Math.max(...vals) : Math.min(...vals)
  }
  return (
    <div className="stack">
      <section className="card">
        <h3>Held-out evaluation: 200 scenarios, mean ± std over 3 training seeds</h3>
        <div className="table-wrap">
          <table className="tbl">
            <thead>
              <tr><th>Metric</th>{AGENTS.map((a) => <th key={a.id} style={{ color: a.color }}>{a.short}</th>)}</tr>
            </thead>
            <tbody>
              {ROWS.map((r) => {
                const b = best(r)
                return (
                  <tr key={r.k}>
                    <td>{r.label}</td>
                    {AGENTS.map((a) => {
                      const s = summary[a.id][r.k]
                      const isBest = Math.abs(s.mean - b) < 1e-9
                      return (
                        <td key={a.id} className={isBest ? 'best' : ''}>
                          <span className="mono">{(s.mean * r.mult).toFixed(r.dp)}{r.unit}</span>
                          <span className="dim mono small"> ± {(s.std * r.mult).toFixed(r.dp)}</span>
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
              <tr>
                <td>Model parameters</td>
                {AGENTS.map((a) => <td key={a.id} className="mono">{summary[a.id].params.toLocaleString()}</td>)}
              </tr>
              <tr>
                <td>Training time (min / run)</td>
                {AGENTS.map((a) => <td key={a.id} className="mono">{(summary[a.id].train_seconds / 60).toFixed(1)}</td>)}
              </tr>
            </tbody>
          </table>
        </div>
        <p className="dim small">Highlighted cell = best agent on that metric.</p>
      </section>

      <section className="card">
        <h3>Success rate per seed</h3>
        <div className="bars">
          {AGENTS.map((a) => {
            const s = summary[a.id].success
            return (
              <div key={a.id} className="barrow">
                <span className="barname">{a.short}</span>
                <div className="bartrack">
                  <div className="barfill" style={{ width: `${s.mean * 100}%`, background: a.color }} />
                  {s.per_seed.map((v, i) => <span key={i} className="seeddot" style={{ left: `${v * 100}%` }} title={`seed ${i}: ${(v * 100).toFixed(1)}%`} />)}
                </div>
                <span className="mono barval">{(s.mean * 100).toFixed(1)}%</span>
              </div>
            )
          })}
        </div>
        <p className="dim small">White dots are individual seeds. The LLM-guided agent varies a lot between seeds (13% to 41%), so its improvement over CNN-LSTM is suggestive, not established.</p>
      </section>

      <section className="card">
        <h3>How to read this</h3>
        <ul className="plain">
          <li>The <b>simple baseline wins</b> in this compute budget. It gets explicit goal-relative coordinates and learns fastest. The image and recurrent agents were still improving when training stopped at 2000 episodes, so their numbers are a lower bound.</li>
          <li><b>LLM guidance helped the recurrent agent</b> (about 10% to 26% success), but dose and dose-limit failures did not improve.</li>
          <li>The <b>LLM is a stand-in</b>: a deterministic parser of the advisory text, not a real language model. Its latency is the parser's, not an LLM's.</li>
          <li>Safety shield is identical for all agents and cannot be overridden. Dose-limit failures come from accumulated dose on longer routes, which the shield does not prevent.</li>
        </ul>
      </section>
    </div>
  )
}
