import React, { useState, useEffect } from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import PrivacyPage from './pages/PrivacyPage'
import TermsPage from './pages/TermsPage'
import './index.css'

function RootRouter() {
  const [pathname, setPathname] = useState(() => window.location.pathname.replace(/\/+$/, '') || '/')

  useEffect(() => {
    const handlePopState = () => {
      setPathname(window.location.pathname.replace(/\/+$/, '') || '/')
    }
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  if (pathname === '/privacy') {
    return <PrivacyPage />
  }
  if (pathname === '/terms') {
    return <TermsPage />
  }
  return <App />
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RootRouter />
  </React.StrictMode>,
)
