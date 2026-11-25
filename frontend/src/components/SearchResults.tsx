import type { SearchResult } from "../api";

interface Props {
  result: SearchResult;
  onAsk: (q: string) => void;
}

export default function SearchResults({ result, onAsk }: Props) {
  const relevant = result.threads.filter((t) => t.relevant).length;
  return (
    <section className="post">
      <div className="results-header">
        <h2 className="post-title">
          {result.threads.length} threads for “{result.query}”
        </h2>
        <button type="button" className="btn-primary" onClick={() => onAsk(result.query)}>
          Get a cited answer
        </button>
      </div>
      {relevant === 0 && (
        <div className="notice">
          None of these is a close match, so “Get a cited answer” will report that nothing relevant was found.
        </div>
      )}
      <ol className="results">
        {result.threads.map((t) => (
          <li key={t.question_id} className={`result ${t.relevant ? "" : "is-weak"}`}>
            <div className="result-stats">
              <div><b>{t.votes}</b> votes</div>
              <div className={`answers ${t.has_accepted ? "has-accepted" : t.answers ? "has-answers" : ""}`}>
                {t.has_accepted && "✓ "}<b>{t.answers}</b> answers
              </div>
            </div>
            <div className="result-body">
              <h3>
                <a href={t.url} target="_blank" rel="noopener noreferrer">{t.title}</a>
              </h3>
              <p className="excerpt">{t.excerpt}</p>
              <div className="meta">
                {t.match === "exact" ? (
                  <span className="badge badge-success" title="The identifier in your query appears in this thread">
                    exact identifier match
                  </span>
                ) : (
                  <span className="tag" title="Cosine distance between your query and the question (0 = same meaning)">
                    distance {t.distance?.toFixed(3)}
                  </span>
                )}
                {!t.relevant && <span className="badge">weak match</span>}
                <span className="muted small">asked {t.year}</span>
              </div>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
