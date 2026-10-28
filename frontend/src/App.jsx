import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const API_URL = (import.meta.env.VITE_API_URL || 'https://brandguard-ai-x0h8.onrender.com').replace(/\/+$/, '')
const TODAY_LABEL = new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' }).toUpperCase()
const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard', icon: 'grid' },
  { id: 'guidelines', label: 'Brand guidelines', icon: 'book' },
  { id: 'review', label: 'Content review', icon: 'scan' },
  { id: 'history', label: 'Review history', icon: 'clock' },
  { id: 'drift', label: 'Brand DNA drift', icon: 'spark' },
  { id: 'reports', label: 'Reports', icon: 'chart' },
]
const EMPTY_GUIDELINES = {
  brand_name: '',
  voice_tone: '',
  preferred_terms: [],
  prohibited_terms: [],
  approved_facts: [],
  instructions: '',
  is_sample: false,
}

async function api(path, options = {}) {
  let response
  try {
    const headers = { ...options.headers }
    if (!(typeof FormData !== 'undefined' && options.body instanceof FormData)) {
      headers['Content-Type'] = headers['Content-Type'] || 'application/json'
    }
    response = await fetch(`${API_URL}${path}`, {
      ...options,
      headers,
    })
  } catch {
    throw new Error(`Cannot reach the BrandGuard API at ${API_URL}. Start the backend and retry.`)
  }
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map((item) => item.msg || 'Invalid request').join(' ')
      : body.detail
    const error = new Error(detail || `Request failed (${response.status}).`)
    error.status = response.status
    throw error
  }
  return body
}

function Icon({ name, size = 20 }) {
  const paths = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></>,
    book: <><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v17H6.5A2.5 2.5 0 0 0 4 22z" /><path d="M4 5.5v14A2.5 2.5 0 0 1 6.5 17H20" /><path d="M8 7h8M8 11h7" /></>,
    scan: <><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" /><path d="M7 12h10M12 7v10" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    chart: <><path d="M4 19V5M4 19h17" /><path d="m7 15 4-4 3 2 6-7" /><path d="M16 6h4v4" /></>,
    plus: <><path d="M12 5v14M5 12h14" /></>,
    arrow: <><path d="M5 12h14M13 6l6 6-6 6" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    alert: <><path d="M10.3 3.9 2.5 17.4A1.8 1.8 0 0 0 4 20h16a1.8 1.8 0 0 0 1.5-2.6L13.7 3.9a2 2 0 0 0-3.4 0Z" /><path d="M12 9v4m0 3h.01" /></>,
    file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6M8 13h8m-8 4h8" /></>,
    download: <><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="m7 10 5 5 5-5m-5 5V3" /></>,
    menu: <><path d="M4 6h16M4 12h16M4 18h16" /></>,
    close: <><path d="m18 6-12 12M6 6l12 12" /></>,
    chat: <><path d="M20 11.5a7.5 7.5 0 0 1-7.5 7.5H6l-3 2v-6a7.5 7.5 0 1 1 17-3.5Z" /><path d="M8 11h.01M12 11h.01M16 11h.01" /></>,
    spark: <><path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3Z" /><path d="m19 14 1 2 2 1-2 1-1 2-1-2-2-1 2-1 1-2Z" /></>,
    shield: <><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z" /><path d="m9 12 2 2 4-4" /></>,
  }
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">{paths[name] || paths.file}</svg>
}

function PageHeading({ eyebrow, title, description, action }) {
  return <div className="page-heading"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1>{description && <p className="page-description">{description}</p>}</div>{action}</div>
}

function Score({ value, size = 'normal' }) {
  const tone = value >= 85 ? 'good' : value >= 65 ? 'fair' : 'poor'
  return <span className={`score ${tone} ${size}`}><strong>{value}</strong><small>/100</small></span>
}

function StatusBadge({ status }) {
  const value = (status || 'Pending').toLowerCase()
  return <span className={`status-badge ${value}`}>{status}</span>
}

function FieldList({ id, value, onChange, placeholder }) {
  return <textarea id={id} value={value.join('\n')} onChange={(event) => onChange(event.target.value.split('\n').map((item) => item.trim()).filter(Boolean))} placeholder={placeholder} rows={4} />
}

function App() {
  const [page, setPage] = useState('dashboard')
  const [reviews, setReviews] = useState([])
  const [guidelines, setGuidelines] = useState(EMPTY_GUIDELINES)
  const [documentState, setDocumentState] = useState({ status: 'empty', documents: [], chunk_count: 0 })
  const [selectedReview, setSelectedReview] = useState(null)
  const [backendError, setBackendError] = useState('')
  const [notice, setNotice] = useState('')
  const [loading, setLoading] = useState(true)
  const [mobileNav, setMobileNav] = useState(false)

  const fetchWorkspace = useCallback(() => Promise.all([
    api('/api/guidelines'),
    api('/api/reviews'),
    api('/api/documents'),
  ]), [])

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const [savedGuidelines, savedReviews, savedDocuments] = await fetchWorkspace()
      setGuidelines(savedGuidelines)
      setReviews(savedReviews)
      setDocumentState(savedDocuments)
      setBackendError('')
    } catch (error) {
      setBackendError(error.message)
    } finally {
      setLoading(false)
    }
  }, [fetchWorkspace])

  useEffect(() => {
    let active = true
    fetchWorkspace()
      .then(([savedGuidelines, savedReviews, savedDocuments]) => {
        if (active) {
          setGuidelines(savedGuidelines)
          setReviews(savedReviews)
          setDocumentState(savedDocuments)
          setBackendError('')
        }
      })
      .catch((error) => {
        if (active) setBackendError(error.message)
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => { active = false }
  }, [fetchWorkspace])

  const pendingCount = reviews.filter((review) => review.status === 'Pending').length
  const approvedCount = reviews.filter((review) => review.status === 'Approved').length
  const issueCount = reviews.reduce((total, review) => total + review.issues.length, 0)
  const averageScore = reviews.length ? Math.round(reviews.reduce((total, review) => total + review.score, 0) / reviews.length) : null
  const navigate = (nextPage) => {
    setPage(nextPage)
    setMobileNav(false)
    setNotice('')
    if (nextPage !== 'review') setSelectedReview(null)
  }
  const openReview = async (review) => {
    try {
      const current = await api(`/api/reviews/${review.id}`)
      setSelectedReview(current)
      setPage('review')
      setNotice('')
      setMobileNav(false)
    } catch (error) {
      setBackendError(error.message)
    }
  }
  const onDecision = async (action, review = selectedReview, reason = null) => {
    if (!review) return
    try {
      const updated = await api(`/api/reviews/${review.id}/${action}`, {
        method: 'POST',
        body: JSON.stringify(action === 'reject' ? { reason } : {}),
      })
      setSelectedReview(updated)
      await refresh()
      setNotice(`Review ${action === 'approve' ? 'approved' : 'rejected'} successfully.`)
    } catch (error) {
      setBackendError(error.message)
    }
  }
  const issueTotals = useMemo(() => reviews.flatMap((review) => review.issues).reduce((totals, issue) => {
    totals[issue.severity] = (totals[issue.severity] || 0) + 1
    return totals
  }, { High: 0, Medium: 0, Low: 0 }), [reviews])

  return (
    <div className="app-shell">
      {mobileNav && <button className="mobile-backdrop" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}
      <aside className={`sidebar ${mobileNav ? 'sidebar-open' : ''}`}>
        <div className="brand-lockup"><span className="brand-mark"><Icon name="shield" size={23} /></span><span>brandguard<span className="brand-ai">.ai</span><small>BRAND GOVERNANCE</small></span></div>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="main-nav" aria-label="Main navigation">
          {NAV_ITEMS.map((item) => <button key={item.id} className={`nav-item ${page === item.id ? 'active' : ''}`} onClick={() => navigate(item.id)}><Icon name={item.icon} /><span>{item.label}</span>{item.id === 'history' && reviews.length > 0 && <span className="nav-count">{reviews.length}</span>}</button>)}
        </nav>
        <div className="sidebar-bottom">
          <div className="help-card"><span className="help-icon"><Icon name="spark" size={18} /></span><strong>Keep your brand on point</strong><p>Review every message before it reaches your audience.</p><button onClick={() => navigate('review')}>Start a review <Icon name="arrow" size={15} /></button></div>
          <div className="user-profile"><div className="avatar">S</div><div><strong>SYNAPSE</strong><small>CODEX 48 workspace</small></div><span className="online-dot" /></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar"><button className="mobile-menu" aria-label="Open navigation" onClick={() => setMobileNav(true)}><Icon name="menu" /></button><div className="breadcrumb">Workspace <span>/</span> <strong>{NAV_ITEMS.find((item) => item.id === page)?.label || 'Content review'}</strong></div><div className="topbar-right"><span className="team-pill"><span /> SYNAPSE</span><span className="top-avatar">S</span></div></header>
        {backendError && <div className="global-alert" role="alert"><Icon name="alert" size={18} /><span>{backendError}</span><button className="retry-button" onClick={refresh} disabled={loading}>{loading ? 'Checking…' : 'Retry connection'}</button></div>}
        {notice && <div className="global-notice" role="status"><Icon name="check" size={18} />{notice}<button aria-label="Dismiss message" onClick={() => setNotice('')}>×</button></div>}
        {loading && <div className="loading-strip"><span /> Connecting to your workspace…</div>}
        {page === 'dashboard' && <Dashboard reviews={reviews} pendingCount={pendingCount} approvedCount={approvedCount} issueCount={issueCount} navigate={navigate} openReview={openReview} />}
        {page === 'guidelines' && <GuidelinesPage guidelines={guidelines} documents={documentState} onDocumentsChanged={setDocumentState} onSaved={(saved) => { setGuidelines(saved); setNotice('Brand guidelines saved.') }} setBackendError={setBackendError} />}
        {page === 'review' && <ReviewPage selectedReview={selectedReview} reviews={reviews} onOpenVersion={openReview} onCreated={(review) => { setReviews((current) => [review, ...current]); setSelectedReview(review); setNotice('Content reviewed and saved to history.') }} onDecision={onDecision} onRecheckStart={() => setSelectedReview(null)} guidelines={guidelines} />}
        {page === 'history' && <HistoryPage reviews={reviews} openReview={openReview} />}
        {page === 'drift' && <DriftPage reviews={reviews} guidelines={guidelines} setBackendError={setBackendError} />}
        {page === 'reports' && <ReportsPage reviews={reviews} issueTotals={issueTotals} averageScore={averageScore} openReview={openReview} />}
        <footer className="app-footer"><span>BrandGuard AI <span className="footer-dot">·</span> CODEX 48</span><span>Built for thoughtful, consistent communication</span></footer>
      </main>
      <BrandAssistant reviewId={page === 'review' ? selectedReview?.id : null} />
    </div>
  )
}

function BrandAssistant({ reviewId }) {
  const [open, setOpen] = useState(false)
  const [question, setQuestion] = useState('')
  const [sending, setSending] = useState(false)
  const [messages, setMessages] = useState([{
    id: 'welcome',
    role: 'assistant',
    text: 'Hi! Ask me anything—from general questions and writing help to your brand guidelines and review history.',
    sources: [],
  }])
  const messageList = useRef(null)

  useEffect(() => {
    if (open && messageList.current) messageList.current.scrollTop = messageList.current.scrollHeight
  }, [open, messages])

  const ask = async (event) => {
    event.preventDefault()
    const text = question.trim()
    if (!text || sending) return
    const pendingId = `assistant-${Date.now()}`
    setQuestion('')
    setSending(true)
    setMessages((current) => [
      ...current,
      { id: `user-${pendingId}`, role: 'user', text },
      { id: pendingId, role: 'assistant', text: '', sources: [], pending: true },
    ])
    try {
      const now = new Date()
      const localToday = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
      const response = await api('/api/assistant', {
        method: 'POST',
        body: JSON.stringify({
          question: text,
          review_id: reviewId || null,
          local_today: localToday,
          timezone_offset_minutes: now.getTimezoneOffset(),
          history: messages
            .filter((message) => message.id !== 'welcome' && !message.pending && !message.error)
            .slice(-10)
            .map(({ role, text: messageText }) => ({ role, text: messageText.slice(0, 1800) })),
        }),
      })
      setMessages((current) => current.map((message) => message.id === pendingId
        ? { ...message, text: response.answer, sources: response.sources || [], pending: false }
        : message))
    } catch (error) {
      const errorText = error.status === 404
        ? 'The Brand Assistant endpoint is not available on the backend yet. Please try again after the backend update is deployed.'
        : error.message
      setMessages((current) => current.map((message) => message.id === pendingId
        ? { ...message, text: errorText, pending: false, error: true }
        : message))
    } finally {
      setSending(false)
    }
  }

  return <div className="brand-assistant">
    {open && <section className="assistant-panel" aria-label="Brand assistant chat">
      <header className="assistant-header">
        <span className="assistant-mark"><Icon name="spark" size={17} /></span>
        <div><strong>Brand Assistant</strong><small>Gemini chat with brand context</small></div>
        <button className="assistant-close" aria-label="Close brand assistant" onClick={() => setOpen(false)}><Icon name="close" size={17} /></button>
      </header>
      <div className="assistant-messages" ref={messageList} aria-live="polite">
        {messages.map((message) => <article className={`assistant-message ${message.role} ${message.error ? 'error' : ''}`} key={message.id}>
          <p>{message.pending ? <span className="assistant-typing">Thinking…</span> : message.text}</p>
          {message.sources?.length > 0 && <div className="assistant-sources"><strong>Sources</strong>{message.sources.map((source) => <div className="assistant-source" key={source.source_id}>
            <span>{source.filename ? `${source.filename} · page ${source.page_number}` : source.title}</span>
            {source.source_type === 'review_content'
              ? <details><summary>View saved content</summary><small>{source.excerpt}</small></details>
              : <small>{source.excerpt}</small>}
          </div>)}</div>}
        </article>)}
      </div>
      <form className="assistant-form" onSubmit={ask}>
        <textarea aria-label="Ask the Brand Assistant" rows={2} maxLength={2000} value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit() } }} placeholder="Ask anything…" />
        <button className="button button-primary" type="submit" disabled={sending || !question.trim()} aria-label={sending ? 'Sending question' : 'Send question'}><Icon name="arrow" size={17} /></button>
      </form>
      <p className="assistant-disclaimer">Brand and review answers use saved workspace evidence; general answers come from Gemini.</p>
    </section>}
    <button className={`assistant-toggle ${open ? 'open' : ''}`} aria-label={open ? 'Close Brand Assistant' : 'Open Brand Assistant'} aria-expanded={open} onClick={() => setOpen((current) => !current)}>
      <Icon name={open ? 'close' : 'chat'} size={21} />
      <span>{open ? 'Close' : 'Ask brand assistant'}</span>
    </button>
  </div>
}

function Dashboard({ reviews, pendingCount, approvedCount, issueCount, navigate, openReview }) {
  const recent = reviews.slice(0, 5)
  const chartData = reviews.slice(0, 7).reverse()
  return <div className="content-wrap dashboard-page">
    <PageHeading eyebrow={TODAY_LABEL} title="Good to see you, SYNAPSE." description="A clear view of how every message measures up to your brand." action={<button className="button button-primary" onClick={() => navigate('review')}><Icon name="plus" size={18} /> Review content</button>} />
    <section className="welcome-strip"><div className="welcome-symbol"><Icon name="spark" size={22} /></div><div><strong>Your brand, consistently understood.</strong><p>Check copy against your guidelines before it goes live.</p></div><button className="text-button" onClick={() => navigate('guidelines')}>Manage guidelines <Icon name="arrow" size={16} /></button></section>
    <section className="stat-grid" aria-label="Review metrics">
      <StatCard label="Content reviewed" value={reviews.length} detail={reviews.length ? 'Across all review types' : 'No reviews yet'} icon="file" tone="purple" />
      <StatCard label="Pending approvals" value={pendingCount} detail={pendingCount ? 'Waiting for a decision' : 'You’re all caught up'} icon="clock" tone="amber" />
      <StatCard label="Brand issues found" value={issueCount} detail={issueCount ? 'From stored review findings' : 'No issues recorded yet'} icon="alert" tone="rose" />
      <StatCard label="Approved content" value={approvedCount} detail={approvedCount ? 'Approved review versions' : 'No approvals yet'} icon="check" tone="green" />
    </section>
    <div className="dashboard-grid">
      <section className="panel recent-panel"><div className="panel-heading"><div><h2>Recent reviews</h2><p>Your latest content checks</p></div><button className="link-button" onClick={() => navigate('history')}>View all <Icon name="arrow" size={15} /></button></div>
        {recent.length ? <ReviewTable reviews={recent} openReview={openReview} /> : <EmptyState icon="file" title="Your review history starts here" message="Submit a piece of marketing content and its findings will appear here." action={<button className="button button-secondary" onClick={() => navigate('review')}><Icon name="plus" size={16} /> Start your first review</button>} />}
      </section>
      <section className="panel overview-panel"><div className="panel-heading"><div><h2>Compliance overview</h2><p>Score by recent review</p></div><span className="overview-icon"><Icon name="chart" size={17} /></span></div>
        {chartData.length ? <div className="chart-area"><div className="chart-grid"><span>100</span><span>75</span><span>50</span><span>25</span></div><div className="bar-chart">{chartData.map((review) => <div className="bar-column" key={review.id}><div className="bar-value" style={{ height: `${Math.max(review.score, 4)}%` }} title={`${review.score}%`} /><span>{review.title.length > 8 ? `${review.title.slice(0, 8)}…` : review.title}</span></div>)}</div></div> : <div className="chart-empty"><div className="empty-chart-icon"><Icon name="chart" size={22} /></div><strong>No compliance data yet</strong><p>Run your first review to see scores here.</p></div>}
        <div className="overview-footer"><span><i className="legend-dot" /> Compliance score</span><span>{reviews.length} {reviews.length === 1 ? 'review' : 'reviews'}</span></div>
      </section>
    </div>
    <section className="quick-start"><div><span className="quick-icon"><Icon name="book" size={19} /></span><span><strong>Make every review more useful</strong><small>Keep your brand voice, terminology, and facts up to date.</small></span></div><button className="button button-secondary" onClick={() => navigate('guidelines')}>Set up guidelines <Icon name="arrow" size={15} /></button></section>
  </div>
}

function StatCard({ label, value, detail, icon, tone }) {
  return <article className="stat-card"><div className="stat-top"><span>{label}</span><span className={`stat-icon ${tone}`}><Icon name={icon} size={18} /></span></div><strong className="stat-value">{value}</strong><small>{detail}</small></article>
}

function EmptyState({ icon, title, message, action }) {
  return <div className="empty-state"><span className="empty-icon"><Icon name={icon} size={21} /></span><strong>{title}</strong><p>{message}</p>{action}</div>
}

function ReviewTable({ reviews, openReview }) {
  return <div className="table-scroll"><table className="review-table"><thead><tr><th>CONTENT</th><th>TYPE</th><th>SCORE</th><th>STATUS</th><th /></tr></thead><tbody>
    {reviews.map((review) => <tr key={review.id} onClick={() => openReview(review)} tabIndex={0} onKeyDown={(event) => event.key === 'Enter' && openReview(review)}><td><strong>{review.title}</strong><small>{new Date(review.created_at).toLocaleDateString()}</small></td><td>{review.content_type}</td><td><Score value={review.score} /></td><td><StatusBadge status={review.status} /></td><td><button className="row-arrow" aria-label={`Open ${review.title}`} onClick={(event) => { event.stopPropagation(); openReview(review) }}><Icon name="arrow" size={16} /></button></td></tr>)}
  </tbody></table></div>
}

function GuidelinesPage({ guidelines, documents, onDocumentsChanged, onSaved, setBackendError }) {
  const [form, setForm] = useState(guidelines)
  const [saving, setSaving] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const [draggingFile, setDraggingFile] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const update = (key, value) => setForm((current) => ({ ...current, [key]: value }))
  const selectUploadFile = (file) => {
    setUploadError('')
    setUploadStatus('')
    if (!file) {
      setSelectedFile(null)
      return
    }
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setSelectedFile(null)
      setUploadError('Unsupported file type. Choose a PDF document (.pdf).')
      return
    }
    if (file.size === 0) {
      setSelectedFile(null)
      setUploadError('The selected PDF is empty. Choose a non-empty PDF document.')
      return
    }
    if (file.size > 15 * 1024 * 1024) {
      setSelectedFile(null)
      setUploadError('This PDF exceeds the 15 MB upload limit.')
      return
    }
    setSelectedFile(file)
  }
  const save = async (event) => {
    event?.preventDefault()
    setSaving(true)
    try {
      const saved = await api('/api/guidelines', { method: 'PUT', body: JSON.stringify({ ...form, is_sample: false }) })
      onSaved(saved)
    } catch (error) {
      setBackendError(error.message)
    } finally {
      setSaving(false)
    }
  }
  const uploadPdf = async (event) => {
    event.preventDefault()
    if (!selectedFile) {
      setUploadError('Choose a PDF before uploading.')
      return
    }
    setUploading(true)
    setUploadError('')
    try {
      const payload = new FormData()
      payload.append('file', selectedFile)
      const uploaded = await api('/api/documents', { method: 'POST', body: payload })
      const currentDocuments = await api('/api/documents')
      onDocumentsChanged(currentDocuments)
      setSelectedFile(null)
      event.target.reset()
      setUploadStatus(uploaded.duplicate
        ? `${uploaded.filename} is already indexed (${uploaded.page_count} pages, ${uploaded.chunk_count} chunks).`
        : `${uploaded.filename} indexed: ${uploaded.page_count} pages, ${uploaded.extracted_characters.toLocaleString()} extracted characters, ${uploaded.chunk_count} chunks.`)
    } catch (error) {
      setUploadError(error.message)
    } finally {
      setUploading(false)
    }
  }
  const [uploadStatus, setUploadStatus] = useState('')
  return <div className="content-wrap">
    <PageHeading eyebrow="BRAND GOVERNANCE" title="Brand guidelines" description="Define the rules and reference documents that keep every content review aligned with your brand." action={<button className="button button-primary page-save-button" onClick={save} disabled={saving}><Icon name="check" size={17} /> {saving ? 'Saving…' : 'Save guidelines'}</button>} />
    {guidelines.is_sample && <div className="sample-banner"><span className="sample-badge">SAMPLE DATA</span><span>These Northstar guidelines are an example to help you explore. Replace them with your brand’s approved guidance.</span></div>}
    <section className="panel document-panel">
      <div className="panel-heading"><div><h2>Guideline documents</h2><p>Upload searchable PDF guidance. Text is indexed by page and kept in this local workspace.</p></div><span className={`document-count ${documents.status}`}>{documents.chunk_count} chunks</span></div>
      <form className="document-upload" onSubmit={uploadPdf}>
        <div
          className={`upload-dropzone ${draggingFile ? 'dragging' : ''} ${selectedFile ? 'has-file' : ''}`}
          onDragOver={(event) => { event.preventDefault(); setDraggingFile(true) }}
          onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setDraggingFile(false) }}
          onDrop={(event) => { event.preventDefault(); setDraggingFile(false); selectUploadFile(event.dataTransfer.files?.[0]) }}
        >
          <span className="pdf-picker-icon"><Icon name="file" size={18} /></span>
          <div className="upload-dropzone-copy"><strong>{selectedFile ? selectedFile.name : 'Drop your guideline PDF here'}</strong><small>{selectedFile ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB · Ready to upload` : 'PDF files · selectable text · up to 15 MB · multi-page supported'}</small></div>
          <label className="button button-secondary browse-button" htmlFor="guideline-pdf">Browse files</label>
          <input id="guideline-pdf" className="upload-file-input" type="file" accept="application/pdf,.pdf" onChange={(event) => selectUploadFile(event.target.files?.[0] || null)} />
        </div>
        <button className="button button-primary" type="submit" disabled={uploading || !selectedFile}><Icon name={uploading ? 'spark' : 'plus'} size={16} />{uploading ? 'Uploading & indexing…' : 'Upload and index'}</button>
      </form>
      {uploadError && <p className="upload-feedback error" role="alert"><Icon name="alert" size={15} />{uploadError}</p>}
      {uploadStatus && <p className="upload-feedback success" role="status"><Icon name="check" size={15} />{uploadStatus}</p>}
      {uploading && <p className="upload-feedback progress" role="status"><span className="upload-progress-dot" />Uploading PDF, extracting page text, and creating searchable embeddings. This may take a little while.</p>}
      {documents.error && <p className="upload-feedback error" role="alert"><Icon name="alert" size={15} />Knowledge index issue: {documents.error}</p>}
      {documents.documents.length ? <div className="document-list">{documents.documents.map((document) => <div className="document-row" key={document.document_id}><span className="document-file-icon"><Icon name="file" size={17} /></span><div className="document-row-main"><strong>{document.filename}</strong><small>{document.page_count} {document.page_count === 1 ? 'page' : 'pages'} · {document.chunk_count} indexed chunks · {(document.extracted_characters || 0).toLocaleString()} extracted characters · Added {new Date(document.uploaded_at).toLocaleDateString()}</small></div><span className={`ingestion-status ${document.ingestion_status === 'Indexed' || documents.status === 'ready' ? 'indexed' : 'failed'}`}>{document.ingestion_status || (documents.status === 'ready' ? 'Indexed' : 'Index issue')}</span></div>)}</div> : <div className="document-empty"><Icon name="book" size={17} /><span>No PDFs indexed yet. Structured brand guidelines below remain available.</span></div>}
    </section>
    <form className="guidelines-form" onSubmit={save}>
      <section className="panel form-panel"><div className="form-section-head"><span className="section-number">01</span><div><h2>Brand identity</h2><p>Start with the name reviewers should use.</p></div></div><label className="field-label">Brand name <span>*</span><input required maxLength={120} value={form.brand_name} onChange={(event) => update('brand_name', event.target.value)} placeholder="e.g. Northstar" /></label></section>
      <section className="panel form-panel"><div className="form-section-head"><span className="section-number">02</span><div><h2>Voice & tone</h2><p>Describe the personality and language of your brand.</p></div></div><label className="field-label">Voice and tone<textarea maxLength={3000} rows={4} value={form.voice_tone} onChange={(event) => update('voice_tone', event.target.value)} placeholder="e.g. Warm, direct, and optimistic. Keep the language human and practical." /></label></section>
      <div className="guideline-two-col">
        <section className="panel form-panel"><div className="form-section-head"><span className="section-number">03</span><div><h2>Preferred terminology</h2><p>One term or phrase per line.</p></div></div><label className="field-label sr-only" htmlFor="preferred-terms">Preferred terminology</label><FieldList id="preferred-terms" value={form.preferred_terms} onChange={(value) => update('preferred_terms', value)} placeholder={'e.g. your customers\\nproduct materials\\nservice plan'} /></section>
        <section className="panel form-panel"><div className="form-section-head"><span className="section-number">04</span><div><h2>Prohibited language</h2><p>Words or claims your brand should avoid.</p></div></div><label className="field-label sr-only" htmlFor="prohibited-terms">Prohibited language</label><FieldList id="prohibited-terms" value={form.prohibited_terms} onChange={(value) => update('prohibited_terms', value)} placeholder={'guaranteed\\nbest in the world\\nrevolutionary'} /></section>
      </div>
      <section className="panel form-panel"><div className="form-section-head"><span className="section-number">05</span><div><h2>Approved company facts</h2><p>Verified statements reviewers can reference. One fact per line.</p></div></div><label className="field-label sr-only" htmlFor="approved-facts">Approved company facts</label><FieldList id="approved-facts" value={form.approved_facts} onChange={(value) => update('approved_facts', value)} placeholder={'Designed in Portland.\\nTwo-year limited warranty introduced in 2022.'} /></section>
      <section className="panel form-panel"><div className="form-section-head"><span className="section-number">06</span><div><h2>Additional instructions</h2><p>Any messaging or formatting guidance specific to your brand.</p></div></div><label className="field-label">Brand-specific instructions<textarea maxLength={5000} rows={4} value={form.instructions} onChange={(event) => update('instructions', event.target.value)} placeholder="e.g. Lead with the customer benefit. Keep claims specific and supportable." /></label></section>
      <div className="form-actions"><span>Changes are saved to this manager workspace.</span></div>
    </form>
  </div>
}

function DriftPage({ reviews }) {
  const [content, setContent] = useState('')
  const [candidateId, setCandidateId] = useState('')
  const approvedReviews = reviews.filter((review) => review.status === 'Approved')
  const [approvedReviewId, setApprovedReviewId] = useState(null)
  const selectedApprovedReviewId = approvedReviewId === null ? approvedReviews[0]?.id || '' : approvedReviewId
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [checking, setChecking] = useState(false)
  const checkDrift = async (event) => {
    event.preventDefault()
    setError('')
    setResult(null)
    setChecking(true)
    try {
      const drift = await api('/api/drift', {
        method: 'POST',
        body: JSON.stringify({ content, approved_review_id: selectedApprovedReviewId || null }),
      })
      setResult(drift)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setChecking(false)
    }
  }
  const selectCandidate = (id) => {
    setCandidateId(id)
    const selected = reviews.find((review) => review.id === id)
    if (selected) setContent(selected.content)
  }
  const differences = result?.approved_content_differences || []
  return <div className="content-wrap">
    <PageHeading eyebrow="BRAND DNA" title="Brand drift detector" description="Compare a new draft with saved guidance and a manager-selected approved version." />
    <div className="drift-layout">
      <form className="panel drift-form" onSubmit={checkDrift}>
        <div className="panel-heading form-panel-heading"><div><span className="step-label">RULE-BASED COMPARISON</span><h2>Choose content to compare</h2><p>Review saved submissions over time or paste a fresh draft.</p></div><span className="review-form-mark"><Icon name="spark" size={21} /></span></div>
        <label className="field-label">Existing submission (optional)<select value={candidateId} onChange={(event) => selectCandidate(event.target.value)}><option value="">Paste a new draft below</option>{reviews.map((review) => <option value={review.id} key={review.id}>v{review.version_number || 1} · {review.title} · {new Date(review.created_at).toLocaleDateString()}</option>)}</select></label>
        <label className="field-label">Approved content to compare against<select value={selectedApprovedReviewId} onChange={(event) => setApprovedReviewId(event.target.value)}><option value="">No approved-copy comparison</option>{approvedReviews.map((review) => <option value={review.id} key={review.id}>v{review.version_number || 1} · {review.title} · {new Date(review.created_at).toLocaleDateString()}</option>)}</select></label>
        {!approvedReviews.length && <div className="document-empty"><Icon name="clock" size={16} /><span>No approved submissions yet. Save an approval from Review results to enable approved-copy comparison.</span></div>}
        <label className="field-label">Candidate content <span>*</span><textarea required maxLength={20000} rows={11} value={content} onChange={(event) => setContent(event.target.value)} placeholder="Paste the new content you want to compare with the brand guidelines…" /><span className="character-count">{content.length.toLocaleString()} / 20,000 characters</span></label>
        {error && <div className="inline-error" role="alert"><Icon name="alert" size={16} />{error}</div>}
        <button type="submit" className="button button-primary review-submit" disabled={checking || !content.trim()}><Icon name={checking ? 'spark' : 'scan'} size={18} />{checking ? 'Comparing content…' : 'Check brand drift'}<Icon name="arrow" size={16} /></button>
      </form>
      {result ? <section className="drift-results">
        <section className="panel drift-score-panel"><div className="drift-score"><Score value={result.score} size="large" /><span>rule-based<br />consistency estimate</span></div><div><h2>Consistency estimate</h2><p>{result.findings.length ? `${result.findings.length} detectable deviation${result.findings.length === 1 ? '' : 's'} against the saved guidance.` : 'No configured rule-based deviations detected.'}</p><small>This score is a transparent heuristic, not an objective measure of brand voice.</small></div></section>
        <section className="panel drift-explanation"><div className="panel-heading"><div><h2>Score rules</h2><p>Every penalty is listed below; score is 100 minus penalties, floored at 0.</p></div></div><p>{result.score_method}</p>{result.score_breakdown.length ? <ul>{result.score_breakdown.map((rule, index) => <li key={`${rule.rule}-${index}`}><span>{rule.rule}</span><strong>−{rule.penalty}</strong><small>Evidence: {rule.evidence}</small></li>)}</ul> : <div className="document-empty"><Icon name="check" size={16} /><span>No score penalties applied.</span></div>}</section>
        <section className="panel drift-findings"><div className="panel-heading"><div><h2>Detected deviations</h2><p>Only configured text rules are reported here; there is no subjective AI tone score.</p></div><span className="engine-badge">Rule-based</span></div>
          {result.findings.length ? <div className="findings-list">{result.findings.map((finding, index) => <article className="finding-card" key={`${finding.category}-${index}`}><div className="finding-category"><strong>{finding.category}</strong><span className={`severity-heading ${finding.severity.toLowerCase()}`}>{finding.severity}</span>{finding.problematic_text && <code>"{finding.problematic_text}"</code>}</div><p>{finding.explanation}</p><div className="evidence"><strong>Evidence</strong><span>{finding.evidence}</span></div><small className="claim-method">{finding.method}</small></article>)}</div> : <div className="clear-result"><span><Icon name="check" size={19} /></span><div><strong>No configured deviations detected</strong><p>Rule-based checks do not assess every possible tone, terminology, or messaging difference.</p></div></div>}
        </section>
        {result.approved_content_title && <section className="panel drift-comparison"><div className="panel-heading"><div><h2>Compared with approved content</h2><p>{result.approved_content_title} · wording differences are evidence of change, not automatic violations.</p></div></div>
          {differences.length ? <div className="difference-list">{differences.map((difference, index) => <article className={`difference-row ${difference.change}`} key={`${difference.change}-${index}`}><strong>{difference.change}</strong>{difference.approved_excerpt && <p><span>Approved:</span> {difference.approved_excerpt}</p>}{difference.candidate_excerpt && <p><span>Candidate:</span> {difference.candidate_excerpt}</p>}</article>)}</div> : <div className="clear-result"><span><Icon name="check" size={19} /></span><div><strong>Text is identical</strong><p>The candidate matches the selected approved content exactly.</p></div></div>}
        </section>}
      </section> : <aside className="panel drift-empty"><span className="aside-icon"><Icon name="spark" size={19} /></span><h3>How the drift check works</h3><p>It checks exact prohibited and preferred terminology, explicit required phrases, and a small set of hype/urgency markers only when your voice guidance asks for a measured tone.</p><p>If you select approved content, exact wording changes are shown separately and do not automatically reduce the score.</p><div className="drift-empty-meta"><Icon name="shield" size={15} /> No subjective or random scores</div></aside>}
    </div>
  </div>
}

function ReviewPage({ selectedReview, reviews, onOpenVersion, onCreated, onDecision, onRecheckStart, guidelines }) {
  const [form, setForm] = useState({ title: '', content_type: 'Social media', content: '', previous_review_id: null })
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const submit = async (event) => {
    event.preventDefault()
    setError('')
    setSubmitting(true)
    try {
      const result = await api('/api/reviews', { method: 'POST', body: JSON.stringify(form) })
      onCreated(result)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setSubmitting(false)
    }
  }
  const review = selectedReview
  return <div className="content-wrap">
    <PageHeading eyebrow="BEFORE IT GOES LIVE" title={review ? 'Review results' : 'Content review'} description={review ? `Review the findings for “${review.title}” and decide what happens next.` : 'Check a draft against your saved brand guidance.'} />
    {review ? <ReviewResults review={review} reviews={reviews} onOpenVersion={onOpenVersion} onDecision={onDecision} onRecheck={(content) => { setForm({ title: `${review.title} - revised`, content_type: review.content_type, content, previous_review_id: review.id }); onRecheckStart(); window.scrollTo({ top: 0, behavior: 'smooth' }) }} /> : <div className="review-layout">
      <form className="panel review-form" onSubmit={submit}>
        <div className="panel-heading form-panel-heading"><div><span className="step-label">NEW REVIEW</span><h2>Share your draft</h2><p>We’ll compare it with your saved guidelines.</p></div><span className="review-form-mark"><Icon name="scan" size={21} /></span></div>
        <label className="field-label">Content title <span>*</span><input required maxLength={200} value={form.title} onChange={(event) => setForm({ ...form, title: event.target.value })} placeholder="e.g. Spring launch announcement" /></label>
        <label className="field-label">Content type <span>*</span><select value={form.content_type} onChange={(event) => setForm({ ...form, content_type: event.target.value })}><option>Social media</option><option>Advertisement</option><option>Blog</option><option>Email</option></select></label>
        <label className="field-label">Marketing copy <span>*</span><textarea required maxLength={20000} rows={11} value={form.content} onChange={(event) => setForm({ ...form, content: event.target.value })} placeholder="Paste the copy you’d like to check…" /><span className="character-count">{form.content.length.toLocaleString()} / 20,000 characters</span></label>
        {error && <div className="inline-error" role="alert"><Icon name="alert" size={16} />{error}</div>}
        <button type="submit" className="button button-primary review-submit" disabled={submitting}><Icon name={submitting ? 'spark' : 'scan'} size={18} />{submitting ? 'Checking your content…' : 'Review content'}<Icon name="arrow" size={16} /></button>
        <p className="privacy-note"><Icon name="shield" size={14} /> Your draft is stored in this local workspace.</p>
      </form>
      <aside className="review-aside">
        <section className="panel review-aside-card"><span className="aside-icon"><Icon name="book" size={19} /></span><h3>What we check</h3><ul><li><Icon name="check" size={15} /> Brand voice and tone</li><li><Icon name="check" size={15} /> Preferred and prohibited terms</li><li><Icon name="check" size={15} /> Claims against approved facts</li><li><Icon name="check" size={15} /> Messaging clarity and accessibility</li></ul></section>
        <section className="engine-note"><span className="engine-dot" /><div><strong>{guidelines.brand_name || 'Brand'} guidelines connected</strong><p>Review engine is shown with every result. If Gemini is not configured, a rule-based fallback is clearly labelled.</p></div></section>
      </aside>
    </div>}
  </div>
}

function ReviewResults({ review, reviews, onOpenVersion, onDecision, onRecheck }) {
  const [rejectOpen, setRejectOpen] = useState(false)
  const [reason, setReason] = useState('')
  const grouped = ['High', 'Medium', 'Low'].map((severity) => ({ severity, issues: review.issues.filter((issue) => issue.severity === severity) })).filter((group) => group.issues.length)
  const sources = review.sources || []
  const retrievalStatus = review.retrieval_status || 'empty'
  const versionGroupId = review.version_group_id || review.id
  const versions = reviews.filter((item) => (item.version_group_id || item.id) === versionGroupId).sort((a, b) => a.version_number - b.version_number)
  const previousVersion = reviews.find((item) => item.id === review.previous_review_id)
  const issueIdentity = (issue) => `${issue.category}|${(issue.problematic_text || issue.explanation).trim().toLowerCase()}`
  const currentIssueKeys = new Set(review.issues.map(issueIdentity))
  const resolvedIssues = (previousVersion?.issues || []).filter((issue) => !currentIssueKeys.has(issueIdentity(issue)))
  const remainingIssues = (previousVersion?.issues || []).filter((issue) => currentIssueKeys.has(issueIdentity(issue)))
  const versionChanges = review.content_differences || []
  const retrievalMessage = retrievalStatus === 'retrieved'
    ? review.context_used_by_model
      ? `${sources.length} relevant passage${sources.length === 1 ? '' : 's'} retrieved and supplied to Gemini for this review.${review.retrieval_error ? ` Some claim retrieval requests failed: ${review.retrieval_error}` : ''}`
      : `${sources.length} passage${sources.length === 1 ? '' : 's'} retrieved, but Gemini was unavailable. The review used the rule-based fallback; the model did not use this evidence.${review.retrieval_error ? ` Some claim retrieval requests failed: ${review.retrieval_error}` : ''}`
    : retrievalStatus === 'empty'
      ? 'No PDF guidelines are indexed. This review used structured brand guidelines only.'
      : retrievalStatus === 'no_relevant_sources'
        ? 'No sufficiently relevant passages were found. No document evidence was supplied to the model.'
        : retrievalStatus === 'unavailable'
          ? review.retrieval_error || 'Retrieval is unavailable. Configure GEMINI_API_KEY to create embeddings.'
          : `Retrieval failed; no document passages were supplied to the model. ${review.retrieval_error || ''}`.trim()
  return <div className="results-wrap">
    <section className="result-summary panel"><div className="result-score"><Score value={review.score} size="large" /><span>compliance<br />score</span></div><div className="result-summary-copy"><div className="result-tags"><StatusBadge status={review.status} /><span className={`engine-badge ${review.engine.includes('Gemini') ? 'gemini' : ''}`}><Icon name="spark" size={14} /> {review.engine}</span></div><h2>{review.summary}</h2><p>Reviewed {new Date(review.created_at).toLocaleString()} <span>·</span> {review.content_type}</p></div></section>
    <div className={`retrieval-banner ${retrievalStatus}`} role="status"><Icon name={retrievalStatus === 'retrieved' ? 'book' : 'alert'} size={17} /><span><strong>Document retrieval: {retrievalStatus.replaceAll('_', ' ')}</strong><small>{retrievalMessage}</small></span></div>
    <div className="result-caveat"><Icon name="alert" size={17} /><span>Automated review is guidance, not proof of factual accuracy. Claims not found in approved facts may be unverified; confirm with a trusted source.</span></div>
    <section className="panel claims-panel">
      <div className="panel-heading"><div><h2>Evidence-backed claim verification</h2><p>Claims are extracted with rule-based sentence patterns. Missing evidence is not proof of falsity.</p></div><span className="history-total">{(review.claim_assessments || []).length} claims</span></div>
      {(review.claim_assessments || []).length ? <div className="claim-list">{review.claim_assessments.map((claim, index) => <article className="claim-card" key={`${claim.claim}-${index}`}>
        <div className="claim-heading"><strong>{claim.claim}</strong><span className={`claim-status ${claim.classification.toLowerCase()}`}>{claim.classification}</span></div>
        <p>{claim.explanation}</p><small className="claim-method">Assessment: {claim.method}</small>
        {claim.evidence.length ? <div className="claim-evidence-list">{claim.evidence.map((evidence) => <div className="claim-evidence" key={evidence.source_id}>
          <strong>{evidence.source_type === 'document' ? `${evidence.filename} · p. ${evidence.page_number}` : 'Saved approved fact'}</strong>
          <span>{evidence.excerpt}</span>
          {evidence.source_type === 'document' && evidence.similarity !== null && <small>{Math.round(evidence.similarity * 100)}% retrieval similarity · not a truth score</small>}
        </div>)}</div> : <div className="claim-no-evidence">No approved fact or retrieved passage was available for this claim.</div>}
      </article>)}</div> : <div className="clear-result"><span><Icon name="check" size={19} /></span><div><strong>No factual claims matched the extractor</strong><p>Sentence-pattern extraction can miss claims; review the submitted copy manually.</p></div></div>}
    </section>
    {versions.length > 1 && <section className="panel versions-panel"><div className="panel-heading"><div><h2>Content versions</h2><p>Each decision is stored against the exact submitted text.</p></div></div><div className="version-list">{versions.map((version) => <button className={`version-row ${version.id === review.id ? 'current' : ''}`} key={version.id} onClick={() => onOpenVersion(version)}>
      <span><strong>Version {version.version_number}</strong><small>{new Date(version.created_at).toLocaleString()}</small></span><Score value={version.score} /><StatusBadge status={version.status} />
    </button>)}</div></section>}
    {previousVersion && <section className="panel version-comparison"><div className="panel-heading"><div><h2>Version {review.version_number} compared with version {previousVersion.version_number}</h2><p>Compliance and findings are compared with the version this revision is linked to.</p></div></div>
      <div className="version-score-change"><span>Previous <Score value={previousVersion.score} /></span><span>Current <Score value={review.score} /></span><strong>{review.score - previousVersion.score > 0 ? '+' : ''}{review.score - previousVersion.score} points</strong></div>
      <div className="version-issue-summary"><div><strong>{resolvedIssues.length}</strong><span>previous issue{resolvedIssues.length === 1 ? '' : 's'} resolved</span></div><div><strong>{remainingIssues.length}</strong><span>previous issue{remainingIssues.length === 1 ? '' : 's'} remaining</span></div><div><strong>{Math.max(0, review.issues.length - remainingIssues.length)}</strong><span>new finding{Math.max(0, review.issues.length - remainingIssues.length) === 1 ? '' : 's'}</span></div></div>
      <div className="version-text-changes"><div className="version-changes-heading"><strong>Copy changes</strong><small>Exact wording diff against the selected parent version</small></div>
        {versionChanges.length ? <div className="difference-list">{versionChanges.map((difference, index) => <article className={`difference-row ${difference.change}`} key={`${difference.change}-${index}`}><strong>{difference.change}</strong>{difference.approved_excerpt && <p><span>Previous:</span> {difference.approved_excerpt}</p>}{difference.candidate_excerpt && <p><span>Current:</span> {difference.candidate_excerpt}</p>}</article>)}</div> : <p className="version-no-changes">No wording changes from the selected parent.</p>}
      </div>
    </section>}
    <section className="panel findings-panel"><div className="panel-heading"><div><h2>Findings</h2><p>{review.issues.length ? `${review.issues.length} item${review.issues.length === 1 ? '' : 's'} to review` : 'No issues identified'}</p></div></div>
      {review.issues.length ? <div className="findings-list">
        {grouped.map((group) => <div className="finding-group" key={group.severity}>
          <div className={`severity-heading ${group.severity.toLowerCase()}`}><span className="severity-mark" />{group.severity} severity <span>{group.issues.length}</span></div>
          {group.issues.map((issue, index) => <article className="finding-card" key={`${issue.category}-${index}`}>
            <div className="finding-category"><strong>{issue.category}</strong>{issue.problematic_text && <code>"{issue.problematic_text}"</code>}</div>
            <p>{issue.explanation}</p>
            {issue.evidence && <div className="evidence"><strong>Guideline evidence</strong><span>{issue.evidence}</span></div>}
            {issue.source_chunk_ids?.length > 0 && <div className="issue-sources"><strong>Supporting document source</strong>{issue.source_chunk_ids.map((chunkId) => { const source = sources.find((item) => item.chunk_id === chunkId); return source ? <span className="source-chip" key={chunkId}>{source.filename} · p. {source.page_number}</span> : null })}</div>}
            <div className="suggestion"><Icon name="spark" size={16} /><div><strong>Suggested correction</strong><p>{issue.suggested_correction}</p></div></div>
          </article>)}
        </div>)}
      </div> : <div className="clear-result"><span><Icon name="check" size={19} /></span><div><strong>Looking good against your guidelines</strong><p>No issues were identified. Continue to verify any claims that require external evidence.</p></div></div>}
    </section>
    {sources.length > 0 && <section className="panel evidence-sources"><div className="panel-heading"><div><h2>Retrieved document evidence</h2><p>Source excerpts returned by semantic search. Similarity is a retrieval score, not proof of a claim.</p></div></div><div className="retrieved-source-list">{sources.map((source) => <article className="retrieved-source" key={source.chunk_id}><div className="retrieved-source-heading"><strong>{source.filename}</strong><span>Page {source.page_number}</span><small>{Math.round(source.similarity * 100)}% match</small></div><blockquote>{source.excerpt}</blockquote><code>Chunk {source.chunk_id}</code></article>)}</div></section>}
    <section className="panel copy-comparison"><div className="panel-heading"><div><h2>Content comparison</h2><p>Original draft and review suggestions</p></div></div><div className="comparison-grid"><div><span className="comparison-label">ORIGINAL CONTENT</span><p>{review.content}</p></div><div><span className="comparison-label suggested-label">SUGGESTED REVISION</span><p>{review.revised_content || review.content}</p>{review.revision_notice && <p className="revision-notice" role="status">{review.revision_notice}</p>}<button className="text-button copy-button" onClick={() => onRecheck(review.revised_content || review.content)}>Review revised copy <Icon name="arrow" size={15} /></button></div></div></section>
    {review.status === 'Pending' && <div className="decision-bar"><div><strong>Ready to make a decision?</strong><span>Approval status will be saved to review history.</span></div><div><button className="button button-secondary reject-button" onClick={() => setRejectOpen(true)}>Reject</button><button className="button button-primary" onClick={() => onDecision('approve', review)}><Icon name="check" size={17} /> Approve</button></div></div>}
    {review.status !== 'Pending' && <div className="decision-bar resolved"><div><strong>This review was {review.status.toLowerCase()}.</strong><span>{review.rejection_reason ? `Reason: ${review.rejection_reason}` : 'This decision is saved to your review history.'}</span></div><StatusBadge status={review.status} /></div>}
    {rejectOpen && <div className="modal-backdrop" role="presentation"><form className="confirm-modal" onSubmit={(event) => { event.preventDefault(); onDecision('reject', review, reason); setRejectOpen(false) }}><button type="button" className="modal-close" aria-label="Close" onClick={() => setRejectOpen(false)}><Icon name="close" size={19} /></button><span className="modal-icon"><Icon name="alert" size={20} /></span><h2>Reject this content?</h2><p>The decision will be saved with this review.</p><label className="field-label">Reason (optional)<textarea maxLength={1000} rows={3} value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Add a note for the review history" /></label><div className="modal-actions"><button type="button" className="button button-secondary" onClick={() => setRejectOpen(false)}>Cancel</button><button type="submit" className="button button-danger">Reject content</button></div></form></div>}
  </div>
}

function HistoryPage({ reviews, openReview }) {
  return <div className="content-wrap"><PageHeading eyebrow="YOUR WORKSPACE" title="Review history" description="Every check and decision, all in one place." action={<span className="history-total">{reviews.length} {reviews.length === 1 ? 'review' : 'reviews'}</span>} />
    <section className="panel history-panel">{reviews.length ? <ReviewTable reviews={reviews} openReview={openReview} /> : <EmptyState icon="clock" title="No reviews in your history" message="Once you review content, it will be saved here with its score and approval status." />}</section>
  </div>
}

function ReportsPage({ reviews, issueTotals, averageScore, openReview }) {
  const exportCsv = () => {
    const header = ['Title', 'Content type', 'Score', 'Status', 'Engine', 'Issue count', 'High issues', 'Medium issues', 'Low issues', 'Reviewed at']
    const rows = reviews.map((review) => [review.title, review.content_type, review.score, review.status, review.engine, review.issues.length, ...['High', 'Medium', 'Low'].map((severity) => review.issues.filter((issue) => issue.severity === severity).length), review.created_at])
    const csv = [header, ...rows].map((row) => row.map((field) => `"${String(field ?? '').replaceAll('"', '""')}"`).join(',')).join('\r\n')
    const url = URL.createObjectURL(new Blob([`\uFEFF${csv}`], { type: 'text/csv;charset=utf-8' }))
    const link = document.createElement('a')
    link.href = url
    link.download = 'brandguard-review-report.csv'
    document.body.append(link)
    link.click()
    link.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  return <div className="content-wrap"><PageHeading eyebrow="MEASURE WHAT MATTERS" title="Reports" description="A snapshot calculated from reviews saved in this workspace." action={<button className="button button-secondary" onClick={exportCsv} disabled={!reviews.length}><Icon name="download" size={17} /> Export CSV</button>} />
    {!reviews.length && <div className="sample-banner report-empty-banner"><Icon name="alert" size={18} /><span>Reports will populate after your first saved content review. No sample statistics are included.</span></div>}
    <section className="report-stat-grid"><StatCard label="Total reviews" value={reviews.length} detail="Stored in this workspace" icon="file" tone="purple" /><StatCard label="Average score" value={averageScore === null ? '—' : `${averageScore}%`} detail="Mean compliance score" icon="chart" tone="green" /><StatCard label="Pending decision" value={reviews.filter((review) => review.status === 'Pending').length} detail="Awaiting approval or rejection" icon="clock" tone="amber" /><StatCard label="Total findings" value={Object.values(issueTotals).reduce((sum, count) => sum + count, 0)} detail="Across stored reviews" icon="alert" tone="rose" /></section>
    <section className="panel report-panel"><div className="panel-heading"><div><h2>Findings by severity</h2><p>Counted from saved review results</p></div></div><div className="severity-report">{['High', 'Medium', 'Low'].map((severity) => <div className={`severity-report-row ${severity.toLowerCase()}`} key={severity}><div><span className="severity-mark" /><strong>{severity}</strong></div><strong className="severity-count">{issueTotals[severity] || 0}</strong><div className="severity-track"><span style={{ width: `${issueTotals[severity] ? Math.max((issueTotals[severity] / Math.max(Object.values(issueTotals).reduce((a, b) => a + b, 0), 1)) * 100, 4) : 0}%` }} /></div></div>)}</div></section>
    <section className="panel history-panel report-history"><div className="panel-heading"><div><h2>Review activity</h2><p>Download a CSV with the review-level details.</p></div></div>{reviews.length ? <ReviewTable reviews={reviews.slice(0, 10)} openReview={openReview} /> : <EmptyState icon="chart" title="Nothing to report yet" message="Your actual review data will appear here after you complete a review." />}</section>
  </div>
}

export default App
