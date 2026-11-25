import { useState } from "react";
import type { Source } from "../api";

interface Props {
  source: Source;
  highlighted?: boolean;
  idPrefix?: string;
}

/** One answer the LLM read, shown like a post summary: votes on the left, thread title, then the text. */
export default function SourceCard({ source: s, highlighted, idPrefix = "source" }: Props) {
  const [open, setOpen] = useState(false);
  const oldYears = new Date().getFullYear() - s.year;
  return (
    <article id={`${idPrefix}-${s.n}`} className={`source ${highlighted ? "is-highlighted" : ""}`}>
      <div className="source-stats">
        <div className="source-n">[{s.n}]</div>
        <div className={`votes ${s.accepted ? "is-accepted" : ""}`} title={s.accepted ? "Accepted answer" : "Votes"}>
          {s.accepted && <span aria-hidden="true">✓ </span>}
          {s.votes}
          <span className="votes-label">votes</span>
        </div>
      </div>
      <div className="source-body">
        <h3>
          <a href={s.url} target="_blank" rel="noopener noreferrer">{s.thread_title}</a>
        </h3>
        <div className="meta">
          {s.accepted && <span className="badge badge-success">accepted</span>}
          <span className={`badge ${oldYears >= 6 ? "badge-warning" : ""}`}>
            {s.year}{oldYears >= 6 ? ` · ${oldYears} years old` : ""}
          </span>
          <a className="small" href={s.thread_url} target="_blank" rel="noopener noreferrer">whole thread</a>
          <button type="button" className="link-button small" onClick={() => setOpen(!open)} aria-expanded={open}>
            {open ? "hide answer text" : "read answer text"}
          </button>
        </div>
        {open && <div className="source-text">{s.text}</div>}
      </div>
    </article>
  );
}
