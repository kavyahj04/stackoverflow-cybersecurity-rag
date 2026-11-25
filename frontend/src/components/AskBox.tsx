import type { FormEvent, KeyboardEvent } from "react";
import type { Mode } from "../App";

// Real questions from different areas, including an identifier (the exact-match path)
const EXAMPLES = [
  "How do I prevent SQL injection in PHP?",
  "Is bcrypt still a good choice for hashing passwords?",
  "What is CVE-2014-0160 and how do I check if my server is affected?",
  "Should JWTs be stored in localStorage or in cookies?",
];

interface Props {
  question: string;
  mode: Mode;
  loading: boolean;
  maxChars: number;
  threads?: number;
  onChange: (q: string) => void;
  onModeChange: (m: Mode) => void;
  onSubmit: () => void;
  showExamples: boolean;
  onExample: (q: string) => void;
}

export default function AskBox(p: Props) {
  const tooLong = p.question.length > p.maxChars;
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!tooLong) p.onSubmit();
  };
  // Enter sends, Shift+Enter makes a new line
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) submit(e);
  };

  return (
    <section className="ask">
      <h1>{p.mode === "answer" ? "Ask a security question" : "Search security discussions"}</h1>
      <p className="muted">
        {p.mode === "answer"
          ? "Get an answer written from Security Stack Exchange discussions, with every claim linked to its source."
          : "Find the most relevant threads without generating an answer."}
        {p.threads ? ` ${p.threads.toLocaleString()} threads indexed.` : ""}
      </p>

      <form className="ask-card" onSubmit={submit}>
        <div className="segmented" role="tablist" aria-label="Mode">
          {(["answer", "search"] as const).map((m) => (
            <button key={m} type="button" role="tab" aria-selected={p.mode === m}
              className={p.mode === m ? "is-selected" : ""} onClick={() => p.onModeChange(m)}>
              {m === "answer" ? "Answer with sources" : "Search threads only"}
            </button>
          ))}
        </div>
        <label htmlFor="question" className="label">
          {p.mode === "answer" ? "Your question" : "Search for"}
        </label>
        <textarea
          id="question"
          value={p.question}
          onChange={(e) => p.onChange(e.target.value)}
          onKeyDown={onKey}
          rows={3}
          placeholder="e.g. Is it safe to store session tokens in localStorage?"
          aria-describedby="question-help"
        />
        <div className="ask-footer">
          <span id="question-help" className={`small ${tooLong ? "text-danger" : "muted"}`}>
            {tooLong
              ? `${p.question.length} / ${p.maxChars} characters: please shorten the question.`
              : "Enter to send · Shift+Enter for a new line"}
          </span>
          <button type="submit" className="btn-primary" disabled={p.loading || !p.question.trim() || tooLong}>
            {p.loading ? "Working…" : p.mode === "answer" ? "Get answer" : "Search"}
          </button>
        </div>
      </form>

      {p.showExamples && (
        <div className="examples">
          <span className="muted small">Try:</span>
          {EXAMPLES.map((q) => (
            <button key={q} type="button" className="tag" onClick={() => p.onExample(q)}>
              {q}
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
