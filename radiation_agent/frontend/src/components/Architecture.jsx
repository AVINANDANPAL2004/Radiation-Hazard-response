import React from 'react'
import { AGENTS } from '../shared.js'

const BLOCKS = [
  { t: 'CNN', s: 'Spatial perception', d: 'Reads the 4-channel facility image: local radiation readings (radius 2, noisy), wall blueprint, agent and goal.' },
  { t: 'LSTM', s: 'Temporal memory', d: 'Remembers the last 4 frames, so the agent can infer where sources are drifting even when they leave sensor range.' },
  { t: 'LLM (stand-in)', s: 'Contextual reasoning', d: 'Turns the text advisory into 5 bounded features: hazard per sector plus a dose-budget caution flag. Here a rule-based parser plays this role.' },
]

export default function Architecture() {
  return (
    <div className="stack">
      <section className="card">
        <h3>Pipeline</h3>
        <div className="flow">
          <div className="fbox env">Simulated radiation facility<span>image · sensors · history · text advisory</span></div>
          <div className="arrow">↓</div>
          <div className="frow">
            {BLOCKS.map((b) => <div key={b.t} className="fbox"><b>{b.t}</b><span>{b.s}</span></div>)}
          </div>
          <div className="arrow">↓</div>
          <div className="fbox fuse">Multimodal feature fusion + Double DQN<span>Q-values for up · down · left · right · wait</span></div>
          <div className="arrow">↓</div>
          <div className="fbox shield">Safety shield (rule-based, cannot be overridden)<span>masks walls and cells above the per-step dose limit</span></div>
          <div className="arrow">↓</div>
          <div className="fbox env">Action → reward → policy improvement</div>
        </div>
      </section>

      <div className="grid3">
        {BLOCKS.map((b) => (
          <section key={b.t} className="card"><h4>{b.t}</h4><p className="dim">{b.d}</p></section>
        ))}
      </div>

      <section className="card">
        <h3>The four compared agents</h3>
        <div className="table-wrap">
          <table className="tbl">
            <thead><tr><th>Agent</th><th>CNN</th><th>LSTM</th><th>Advisory</th><th>Sees</th></tr></thead>
            <tbody>
              {AGENTS.map((a, i) => (
                <tr key={a.id}>
                  <td style={{ color: a.color }}><b>{a.name}</b></td>
                  <td>{i >= 1 ? '✓' : '–'}</td><td>{i >= 2 ? '✓' : '–'}</td><td>{i === 3 ? '✓' : '–'}</td>
                  <td className="dim">{a.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="grid2">
        <section className="card">
          <h3>Environment</h3>
          <ul className="plain">
            <li>10×10 maze, about 12% walls, random each episode.</li>
            <li>Two radiation sources of random strength that random-walk (20% chance to move per step).</li>
            <li>Dose per step = 0.04 × dose rate of the cell. Budget = 1.0, exceeding it fails the episode. Max 50 steps.</li>
            <li>15% of advisory reports name the wrong sector.</li>
          </ul>
        </section>
        <section className="card">
          <h3>Reward</h3>
          <ul className="plain mono small">
            <li>−0.02 per step</li>
            <li>+0.1 per cell closer to the goal</li>
            <li>−6 × dose received</li>
            <li>+10 reaching the inspection point</li>
            <li>−5 exceeding the dose budget</li>
          </ul>
        </section>
      </div>

      <section className="card warnbox">
        <h3>Safety and scope</h3>
        <p>Everything here is a <b>simulation</b>. The shield is applied identically to every agent, and the language component only supplies bounded features; it never picks actions or edits the shield. These agents must not be used for real emergency or nuclear operations without independent validation and authorized supervision.</p>
      </section>
    </div>
  )
}
