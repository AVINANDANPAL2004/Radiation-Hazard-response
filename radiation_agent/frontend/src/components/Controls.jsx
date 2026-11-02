import React from 'react'

export default function Controls({ clock, maxT }) {
  const { t, setT, playing, toggle, speed, setSpeed, reset } = clock
  return (
    <div className="controls">
      <button className="btn primary" onClick={toggle}>{playing ? '❚❚ Pause' : t >= maxT ? '↺ Replay' : '▶ Play'}</button>
      <button className="btn" onClick={() => setT(Math.max(0, t - 1))} aria-label="Step back">◀</button>
      <button className="btn" onClick={() => setT(Math.min(maxT, t + 1))} aria-label="Step forward">▶</button>
      <button className="btn" onClick={reset}>Reset</button>
      <input type="range" min="0" max={maxT} value={Math.min(t, maxT)} onChange={(e) => setT(+e.target.value)} className="scrub" aria-label="Time step" />
      <span className="mono dim">t = {Math.min(t, maxT)}/{maxT}</span>
      <label className="speed">
        <span className="dim">speed</span>
        <select value={speed} onChange={(e) => setSpeed(+e.target.value)}>
          <option value="2">0.5×</option>
          <option value="4">1×</option>
          <option value="8">2×</option>
          <option value="16">4×</option>
        </select>
      </label>
    </div>
  )
}
