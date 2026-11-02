import React, { useCallback, useEffect, useRef, useState } from 'react'
import Replay from './components/Replay.jsx'
import Compare from './components/Compare.jsx'
import Results from './components/Results.jsx'
import Learning from './components/Learning.jsx'
import Architecture from './components/Architecture.jsx'
import ScenarioPicker from './components/ScenarioPicker.jsx'
import { api, EVAL_SEED0 } from './shared.js'

const TABS = [
  ['replay', 'Live run'],
  ['compare', 'Compare agents'],
  ['results', 'Results'],
  ['learning', 'Learning curves'],
  ['arch', 'Architecture'],
]

export default function App() {
  const [tab, setTab] = useState('replay')
  const [meta, setMeta] = useState(null)          // /api/health
  const [stats, setStats] = useState(null)        // summary + curves
  const [sel, setSel] = useState({ seed: EVAL_SEED0 + 3, modelSeed: 0 })
  const [scen, setScen] = useState(null)          // last successfully loaded scenario
  const [loading, setLoading] = useState(false)
  const [scenErr, setScenErr] = useState(null)
  const [fatal, setFatal] = useState(null)
  const reqId = useRef(0)

  useEffect(() => {
    Promise.all([api('health'), api('summary'), api('curves')])
      .then(([h, summary, curves]) => { setMeta(h); setStats({ summary, curves }) })
      .catch((e) => setFatal(e.message))
  }, [])

  const run = useCallback((next) => {
    const id = ++reqId.current
    setSel(next); setLoading(true); setScenErr(null)
    api(`compare?seed=${next.seed}&model_seed=${next.modelSeed}`)
      .then((d) => { if (id === reqId.current) setScen(d) })
      .catch((e) => { if (id === reqId.current) setScenErr(e.message) })
      .finally(() => { if (id === reqId.current) setLoading(false) })
  }, [])

  useEffect(() => { if (meta) run(sel) }, [meta]) // first load once the backend answers

  const needsScenario = tab === 'replay' || tab === 'compare'

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1><span className="rad">☢</span> Radiation-Zone Autonomous Agent</h1>
          <p className="dim">CNN + LSTM + LLM + deep reinforcement learning · navigating a drifting radiation field</p>
        </div>
        <span className="badge sim">SIMULATION ONLY</span>
      </header>

      <nav className="tabs" role="tablist">
        {TABS.map(([id, label]) => (
          <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? 'on' : ''} onClick={() => setTab(id)}>{label}</button>
        ))}
      </nav>

      <main>
        {fatal && (
          <div className="card warnbox">
            <h3>Backend not reachable</h3>
            <p>{fatal}</p>
            <p className="dim small">From the project folder: <code className="mono">pip install -r requirements.txt</code> then <code className="mono">uvicorn server:app --port 8000</code>.</p>
          </div>
        )}
        {!fatal && !meta && <div className="card dim">Connecting to backend…</div>}

        {meta && needsScenario && (
          <div className="stack">
            <ScenarioPicker seed={sel.seed} modelSeed={sel.modelSeed} modelSeeds={meta.model_seeds}
              onChange={run} loading={loading} computeMs={scen?.compute_ms} error={scenErr} />
            {scen && tab === 'replay' && <Replay scen={scen} />}
            {scen && tab === 'compare' && <Compare scen={scen} />}
            {!scen && !scenErr && <div className="card dim">Running the agents…</div>}
          </div>
        )}
        {stats && tab === 'results' && <Results summary={stats.summary} />}
        {stats && tab === 'learning' && <Learning curves={stats.curves} />}
        {tab === 'arch' && <Architecture />}
      </main>

      <footer className="dim small">Agents run live on the backend from the trained models, in a simulated facility. The language component is a rule-based advisory parser, not a real LLM.</footer>
    </div>
  )
}
