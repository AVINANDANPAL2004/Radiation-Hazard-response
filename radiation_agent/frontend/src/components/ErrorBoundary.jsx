import React from 'react'

// Keeps one failing view from blanking the whole dashboard. App keys it by tab, so switching tabs resets it.
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('View failed to render:', error, info)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="card warnbox">
          <h3>This view hit an error</h3>
          <p className="small mono">{String(this.state.error?.message || this.state.error)}</p>
          <button className="btn" onClick={() => this.setState({ error: null })}>Try again</button>
        </div>
      )
    }
    return this.props.children
  }
}
