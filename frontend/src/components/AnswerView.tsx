import { useEffect, useMemo, useState, type ReactNode } from "react";
import Markdown from "react-markdown";
import type { AnswerResult } from "../api";
import { answerToMarkdown, downloadJson, slug } from "../exportAnswer";
import SourceCard from "./SourceCard";

/** Turn [3] into a link to source 3 so react-markdown renders it as a clickable citation. */
function linkCitations(text: string): string {
  return text.replace(/\[(\d+)\]/g, "[\\[$1\\]](#source-$1)");
}

export default function AnswerView({ answer: a }: { answer: AnswerResult }) {
  const [highlight, setHighlight] = useState<number | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const titles = useMemo(() => new Map(a.sources.map((s) => [s.n, s.thread_title])), [a]);

  useEffect(() => {
    if (highlight === null) return;
    const t = setTimeout(() => setHighlight(null), 2000);
    return () => clearTimeout(t);
  }, [highlight]);

  const goTo = (n: number) => {
    document.getElementById(`source-${n}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    setHighlight(n);
  };

  const copy = async (what: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(what);
      setTimeout(() => setCopied(null), 1500);
    } catch {
      setCopied("blocked");
    }
  };

  if (!a.found) {
    return (
      <section className="post">
        <h2 className="post-title">{a.question}</h2>
        <div className="notice">
          <strong>No relevant discussions found</strong>
          <p>{a.text}</p>
          <p className="muted small">
            Nothing in the data was close enough to this question, so no answer was generated (this avoids an
            answer made up without sources). Try rephrasing, or use “Search threads only” to see the nearest threads.
          </p>
        </div>
      </section>
    );
  }

  const citation = ({ href, children }: { href?: string; children?: ReactNode }) => {
    const match = href?.match(/^#source-(\d+)$/);
    if (match) {
      const n = Number(match[1]);
      return (
        <a href={href} className="cite" title={titles.get(n)}
          onClick={(e) => { e.preventDefault(); goTo(n); }}>
          {n}
        </a>
      );
    }
    return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
  };

  return (
    <section className="post">
      <h2 className="post-title">{a.question}</h2>

      <div className="post-text">
        <Markdown components={{ a: citation }}>{linkCitations(a.text)}</Markdown>
      </div>

      {a.age_note && (
        <div className="notice notice-warning">
          <strong>Check the dates</strong>
          <p>{a.age_note}</p>
        </div>
      )}
      {a.usage.cut_off && (
        <div className="notice notice-warning">The answer reached the length limit and may be incomplete.</div>
      )}

      <div className="post-actions">
        <button type="button" className="btn-outline" onClick={() => copy("answer", answerToMarkdown(a))}>
          {copied === "answer" ? "Copied" : "Copy as Markdown"}
        </button>
        <button type="button" className="btn-outline" onClick={() => copy("link", window.location.href)}>
          {copied === "link" ? "Copied" : "Copy link"}
        </button>
        <button type="button" className="btn-outline" onClick={() => downloadJson(a, `${slug(a.question)}.json`)}>
          Download JSON
        </button>
        {copied === "blocked" && <span className="small text-danger">The browser blocked the clipboard.</span>}
      </div>

      <h2 className="section-title">
        {a.sources.length} {a.sources.length === 1 ? "Source" : "Sources"}
        <span className="muted small"> cited in the answer</span>
      </h2>
      {a.sources.length === 0 ? (
        <p className="muted">The answer cites no sources.</p>
      ) : (
        a.sources.map((s) => <SourceCard key={s.n} source={s} highlighted={highlight === s.n} />)
      )}

      <Details answer={a} />
    </section>
  );
}

/** For researchers who want to judge the result: what search found and what the model was given. */
function Details({ answer: a }: { answer: AnswerResult }) {
  const r = a.retrieval;
  const u = a.usage;
  const uncited = a.context_sources.filter((s) => !a.sources.some((c) => c.n === s.n));
  return (
    <details className="details">
      <summary>How this answer was made</summary>
      <dl>
        <dt>Search</dt>
        <dd>
          {r.threads_searched} threads found, top {r.threads_in_context} given to the model
          {r.exact_match ? " · exact identifier match (keyword search used)" : ""}
          {r.best_distance !== null ? ` · closest match distance ${r.best_distance.toFixed(3)} (0 = same meaning)` : ""}
        </dd>
        <dt>Model</dt>
        <dd>
          {u.model ?? "-"} · {u.prompt_tokens.toLocaleString()} tokens in, {u.completion_tokens.toLocaleString()} out
          {u.seconds !== null ? ` · ${u.seconds}s` : ""}
        </dd>
        <dt>Citation check</dt>
        <dd>
          {a.citations.invalid_removed.length
            ? `Removed citations to sources that don't exist: ${a.citations.invalid_removed.join(", ")}`
            : "All citations point to real sources"}
          {` · ${a.citations.uncited_sentences} longer sentence${a.citations.uncited_sentences === 1 ? "" : "s"} without a citation`}
        </dd>
      </dl>
      {uncited.length > 0 && (
        <>
          <h3 className="details-title">Given to the model but not cited ({uncited.length})</h3>
          {uncited.map((s) => <SourceCard key={s.n} source={s} idPrefix="context" />)}
        </>
      )}
    </details>
  );
}
