import React, { useEffect } from 'react'

export default function Controls({ clock, maxT }) {
  const { t, setT, playing, toggle, speed, setSpeed, reset } = clock

  // Space = play/pause, arrows = step; ignored while focus is on a form control
  useEffect(() => {
    const onKey = (e) => {
      if (e.target instanceof Element && e.target.matches('input, select, textarea, button')) return
      if (e.code === 'Space') { e.preventDefault(); toggle() }
      else if (e.key === 'ArrowLeft') setT(Math.max(0, t - 1))
      else if (e.key === 'ArrowRight') setT(Math.min(maxT, t + 1))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [t, maxT, toggle, setT])

  return (
    <div className="controls">
      <button className="btn primary" onClick={toggle} title="Play / pause (Space)">{playing ? '❚❚ Pause' : t >= maxT ? '↺ Replay' : '▶ Play'}</button>
      <button className="btn" onClick={() => setT(Math.max(0, t - 1))} aria-label="Previous step" title="Previous step (←)">‹</button>
      <button className="btn" onClick={() => setT(Math.min(maxT, t + 1))} aria-label="Next step" title="Next step (→)">›</button>
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
