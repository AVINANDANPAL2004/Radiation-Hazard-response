import React from 'react'
import Grid from './Grid.jsx'
import Controls from './Controls.jsx'
import { ExposureBar } from './Replay.jsx'
import { AGENTS, OUTCOME, useClock } from '../shared.js'

export default function Compare({ scen }) {
  const maxT = Math.max(...AGENTS.map((a) => scen.runs[a.id].frames.length - 1))
  const clock = useClock(maxT, `${scen.seed}-${scen.model_seed}`)
  const wins = scen.runs
  const nSucc = AGENTS.filter((a) => wins[a.id].outcome === 'success').length

  return (
    <div className="stack">
      <section className="card">
        <div className="row wrap gap between">
          <span className="dim small">{nSucc} of 4 agents reach the goal in this scenario</span>
        </div>
        <Controls clock={clock} maxT={maxT} />
      </section>
      <div className="grid4">
        {AGENTS.map((a) => {
          const run = wins[a.id]
          const fi = Math.min(clock.t, run.frames.length - 1)
          const done = clock.t >= run.frames.length - 1
          return (
            <section key={a.id} className="card compact" style={{ borderTop: `3px solid ${a.color}` }}>
              <div className="row between">
                <h4>{a.name}</h4>
                {done ? <span className={`badge sm ${OUTCOME[run.outcome].cls}`}>{OUTCOME[run.outcome].label}</span> : null}
              </div>
              <Grid scen={scen} run={run} t={clock.t} color={a.color} />
              <div className="row between small">
                <span className="dim">dose</span><span className="mono">{run.frames[fi].exposure.toFixed(2)} / 1.00</span>
              </div>
              <ExposureBar value={run.frames[fi].exposure} />
              <div className="row between small"><span className="dim">steps</span><span className="mono">{Math.min(clock.t, run.steps)}</span></div>
            </section>
          )
        })}
      </div>
      <p className="dim small">Each agent is run live on the backend using the selected trained model. A single scenario is an anecdote; see the Results tab for the 200-scenario averages over 3 seeds.</p>
    </div>
  )
}
