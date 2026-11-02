import React, { useMemo, useState } from 'react'
import Grid from './Grid.jsx'
import Controls from './Controls.jsx'
import { AGENTS, AGENT, OUTCOME, ACTION_LABEL, LIMIT, useClock } from '../shared.js'

const QUADS = ['north-west', 'north-east', 'south-west', 'south-east']
const LEVEL = { strong: 1.0, moderate: 0.6, weak: 0.3 }

// mirrors advisory.py (rule-based stand-in for the LLM)
export function parseAdvisory(text) {
  const hz = [0, 0, 0, 0]
  const re = /(strong|moderate|weak) radiation reported in the (north-west|north-east|south-west|south-east) sector/g
  let m
  while ((m = re.exec(text))) hz[QUADS.indexOf(m[2])] = Math.max(hz[QUADS.indexOf(m[2])], LEVEL[m[1]])
  return { hz, caution: text.includes('WARNING') }
}

export function ExposureBar({ value }) {
  const f = Math.min(1, value / LIMIT)
  const hue = 130 - 130 * f
  return (
    <div className="expo" title={`Dose ${value.toFixed(2)} of ${LIMIT.toFixed(1)}`}>
      <div className="expo-fill" style={{ width: `${f * 100}%`, background: `hsl(${hue} 80% 52%)` }} />
    </div>
  )
}

export default function Replay({ scen }) {
  const [aid, setAid] = useState('llm_cnn_lstm')
  const run = scen.runs[aid]
  const maxT = run.frames.length - 1
  const clock = useClock(maxT, `${scen.seed}-${scen.model_seed}-${aid}`)
  const fi = Math.min(clock.t, maxT)
  const fr = run.frames[fi]
  const done = fi >= maxT
  const parsed = useMemo(() => parseAdvisory(fr.advisory), [fr.advisory])
  const a = AGENT[aid]

  const qs = fr.q || []
  const vis = qs.filter((_, i) => fr.mask?.[i])
  const qMin = Math.min(...(vis.length ? vis : qs), 0), qMax = Math.max(...(vis.length ? vis : qs), 0.01)
  const barW = (v) => 6 + 94 * ((v - qMin) / (qMax - qMin || 1))

  return (
    <div className="two-col">
      <section className="card map-card">
        <div className="row wrap gap">
          <div className="seg" role="tablist" aria-label="Agent">
            {AGENTS.map((g) => (
              <button key={g.id} role="tab" aria-selected={g.id === aid} className={g.id === aid ? 'on' : ''}
                style={g.id === aid ? { borderColor: g.color, color: g.color } : null} onClick={() => setAid(g.id)}>
                {g.short}
              </button>
            ))}
          </div>
        </div>
        <div className="map-wrap"><Grid scen={scen} run={run} t={clock.t} color={a.color} /></div>
        <Controls clock={clock} maxT={maxT} />
        <div className="legend dim">
          <span><b style={{ color: '#67e8f9' }}>◆</b> entrance</span>
          <span><b style={{ color: '#5dffa0' }}>★</b> inspection point</span>
          <span><b style={{ color: '#ffd166' }}>☢</b> drifting source</span>
          <span>background = dose rate (dark → red)</span>
        </div>
      </section>

      <aside className="stack">
        <section className="card">
          <div className="row between">
            <h3>{a.name}</h3>
            {done ? <span className={`badge ${OUTCOME[run.outcome].cls}`}>{OUTCOME[run.outcome].label}</span> : <span className="badge run">Step {fi}</span>}
          </div>
          <p className="dim small">{a.desc}</p>
          <div className="label">Accumulated dose <span className="mono">{fr.exposure.toFixed(2)} / {LIMIT.toFixed(1)}</span></div>
          <ExposureBar value={fr.exposure} />
        </section>

        <section className="card">
          <h3>Decision at this step</h3>
          {fr.q ? (
            <>
              <div className="qlist">
                {qs.map((v, i) => {
                  const blocked = !fr.mask[i]
                  const chosen = i === fr.action
                  return (
                    <div key={i} className={`qrow ${chosen ? 'chosen' : ''} ${blocked ? 'blocked' : ''}`}>
                      <span className="qname">{ACTION_LABEL[i]}</span>
                      <div className="qbar"><div style={{ width: `${barW(v)}%`, background: blocked ? 'repeating-linear-gradient(45deg,#3a2326,#3a2326 4px,#2a1a1d 4px,#2a1a1d 8px)' : a.color }} /></div>
                      <span className="mono qv">{v.toFixed(1)}</span>
                      <span className="qtag">{blocked ? 'shield' : chosen ? 'chosen' : ''}</span>
                    </div>
                  )
                })}
              </div>
              {fr.blocked && <p className="note warnbox">The agent's top-valued action was <b>blocked by the safety shield</b>; it took the best allowed action instead.</p>}
            </>
          ) : <p className="dim">Episode finished.</p>}
        </section>

        <section className="card">
          <h3>Advisory feed {aid === 'llm_cnn_lstm' ? <span className="badge ok sm">used by this agent</span> : <span className="badge sm">not used by this agent</span>}</h3>
          <p className="advisory mono">{fr.advisory}</p>
          <div className="label">Parsed hazard by sector <span className="dim">(rule-based stand-in for the LLM)</span></div>
          <div className="quads">
            {parsed.hz.map((h, i) => (
              <div key={i} className="quad" style={{ background: `rgba(255,107,94,${0.08 + h * 0.6})` }}>
                <span>{QUADS[i]}</span><b className="mono">{h ? h.toFixed(1) : '–'}</b>
              </div>
            ))}
          </div>
          {parsed.caution && <p className="note warnbox">Dose-budget warning raised.</p>}
        </section>
      </aside>
    </div>
  )
}
