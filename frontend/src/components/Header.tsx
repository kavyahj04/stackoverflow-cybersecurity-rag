export default function Header() {
  return (
    <header className="topbar">
      <div className="topbar-inner">
        <a href="/" className="brand" aria-label="Security Q&A Research, home">
          <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden="true">
            <path d="M16 2 4 7v8c0 7.5 5.1 13.6 12 15 6.9-1.4 12-7.5 12-15V7L16 2z" fill="var(--orange)" />
            <path d="m11 16 3.5 3.5L21.5 12" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round"
              strokeLinejoin="round" />
          </svg>
          <span>
            security<b>q&amp;a</b> <span className="brand-sub">research</span>
          </span>
        </a>
        <nav className="topbar-links">
          <a href="/docs">API</a>
          <a href="https://security.stackexchange.com" target="_blank" rel="noopener noreferrer">Source site</a>
        </nav>
      </div>
    </header>
  );
}
