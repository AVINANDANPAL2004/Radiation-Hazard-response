import React, { useState } from 'react'
import { EVAL_SEED0, N_EVAL } from '../shared.js'

const PRESETS = Array.from({ length: 16 }, (_, i) => EVAL_SEED0 + i)

export default function ScenarioPicker({ seed, modelSeed, modelSeeds, onChange, loading, computeMs, error }) {
  const [text, setText] = useState(String(seed))
  const [err, setErr] = useState(null)
  const submit = (e) => {
    e.preventDefault()
    const v = Number(text)
    if (Number.isInteger(v) && v >= 0 && v <= 2147483647) { setErr(null); onChange({ seed: v, modelSeed }) }
    else setErr('Enter a whole number from 0 to 2147483647.')
  }
  const random = () => {
    const v = Math.floor(Math.random() * 1_000_000_000)
    setText(String(v)); onChange({ seed: v, modelSeed })
  }
  const inEval = seed >= EVAL_SEED0 && seed < EVAL_SEED0 + N_EVAL
  return (
    <section className="card picker">
      <div className="row wrap gap">
        <label className="field">
          <span className="dim">Held-out scenario</span>
          <select value={PRESETS.includes(seed) ? seed : ''} onChange={(e) => { setText(e.target.value); onChange({ seed: +e.target.value, modelSeed }) }}>
            {!PRESETS.includes(seed) && <option value="">custom</option>}
            {PRESETS.map((s, i) => <option key={s} value={s}>#{i + 1} (seed {s})</option>)}
          </select>
        </label>
        <form className="field" onSubmit={submit}>
          <span className="dim">Any seed → a brand-new layout</span>
          <div className="row">
            <input className="num" inputMode="numeric" value={text} onChange={(e) => setText(e.target.value)} aria-label="Scenario seed" />
            <button className="btn" type="submit">Run</button>
            <button className="btn" type="button" onClick={random}>🎲 Random</button>
          </div>
          {err && <span className="badge bad">{err}</span>}
        </form>
        <label className="field">
          <span className="dim">Trained model (training seed)</span>
          <select value={modelSeed} onChange={(e) => onChange({ seed, modelSeed: +e.target.value })}>
            {modelSeeds.map((m) => <option key={m} value={m}>seed {m}</option>)}
          </select>
        </label>
        <div className="status">
          {loading ? <span className="badge run">running agents…</span>
            : error ? <span className="badge bad">{error}</span>
            : <span className="badge ok">live inference{computeMs != null ? ` · ${computeMs} ms` : ''}</span>}
          <span className="dim small">{inEval ? 'part of the 200-scenario evaluation set' : 'not in the evaluation set (unseen layout)'}</span>
        </div>
      </div>
    </section>
  )
}
