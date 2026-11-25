import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, ask, getInfo, search, type AnswerResult, type Info, type SearchResult } from "./api";
import AnswerView from "./components/AnswerView";
import AskBox from "./components/AskBox";
import Header from "./components/Header";
import SearchResults from "./components/SearchResults";
import Sidebar from "./components/Sidebar";
import { addToHistory, clearHistory, loadHistory, type HistoryItem } from "./history";

export type Mode = "answer" | "search";

type Result = { kind: "answer"; data: AnswerResult } | { kind: "search"; data: SearchResult };

/** The question in the address bar (?q=...&mode=search), so a result can be shared as a link. */
function readUrl(): HistoryItem | null {
  const params = new URLSearchParams(window.location.search);
  const q = params.get("q")?.trim();
  return q ? { q, mode: params.get("mode") === "search" ? "search" : "answer" } : null;
}

function writeUrl(item: HistoryItem) {
  const params = new URLSearchParams({ q: item.q });
  if (item.mode === "search") params.set("mode", "search");
  const url = `?${params}`;
  if (url !== window.location.search) window.history.pushState(null, "", url);
}

export default function App() {
  const initial = readUrl();
  const [question, setQuestion] = useState(initial?.q ?? "");
  const [mode, setMode] = useState<Mode>(initial?.mode ?? "answer");
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState<Mode | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>(loadHistory);
  const [info, setInfo] = useState<Info | null>(null);
  const inFlight = useRef<AbortController | null>(null);

  const run = useCallback(async (item: HistoryItem, updateUrl = true) => {
    inFlight.current?.abort(); // a new question replaces one still running
    const controller = new AbortController();
    inFlight.current = controller;
    setQuestion(item.q);
    setMode(item.mode);
    setLoading(item.mode);
    setError(null);
    setResult(null);
    if (updateUrl) writeUrl(item);
    try {
      const next: Result =
        item.mode === "answer"
          ? { kind: "answer", data: await ask(item.q, controller.signal) }
          : { kind: "search", data: await search(item.q, controller.signal) };
      setResult(next);
      setHistory((h) => addToHistory(h, item));
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setError(e instanceof ApiError ? e : new ApiError("Something went wrong.", 0));
      }
    } finally {
      if (inFlight.current === controller) setLoading(null);
    }
  }, []);

  useEffect(() => {
    getInfo().then(setInfo).catch(() => setInfo(null));
    if (initial) run(initial, false);
    // Back/forward buttons: show the question from that address
    const onPop = () => {
      const item = readUrl();
      if (item) run(item, false);
      else {
        setResult(null);
        setQuestion("");
      }
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <Header />
      <div className="container">
        <Sidebar
          history={history}
          onPick={(item) => run(item)}
          onClear={() => setHistory(clearHistory())}
        />
        <main className="content">
          <AskBox
            question={question}
            mode={mode}
            loading={loading !== null}
            maxChars={info?.max_question_chars ?? 1000}
            threads={info?.threads}
            onChange={setQuestion}
            onModeChange={setMode}
            onSubmit={() => question.trim() && run({ q: question.trim(), mode })}
            showExamples={!result && !loading && !error}
            onExample={(q) => run({ q, mode })}
          />

          {loading && <Loading mode={loading} />}

          {error && (
            <div className="notice notice-danger" role="alert">
              <strong>{error.status === 429 ? "Slow down" : "Couldn't get a result"}</strong>
              <p>{error.message}</p>
              {error.requestId && <p className="muted small">Reference: {error.requestId}</p>}
            </div>
          )}

          {result?.kind === "answer" && <AnswerView answer={result.data} />}
          {result?.kind === "search" && (
            <SearchResults result={result.data} onAsk={(q) => run({ q, mode: "answer" })} />
          )}
        </main>
      </div>
      <footer className="footer">
        <div className="footer-inner">
          <p>
            Answers are written by a language model from{" "}
            <a href="https://security.stackexchange.com" target="_blank" rel="noopener noreferrer">
              Security Stack Exchange
            </a>{" "}
            posts and can be wrong. Check the cited sources before relying on them. Content is licensed{" "}
            <a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noopener noreferrer">
              CC BY-SA 4.0
            </a>
            .
          </p>
          {info && (
            <p className="muted">
              {info.threads.toLocaleString()} threads · answers by {info.llm_model} from the {info.context} ·{" "}
              <a href="/docs">API docs</a>
            </p>
          )}
        </div>
      </footer>
    </>
  );
}

function Loading({ mode }: { mode: Mode }) {
  return (
    <div className="loading" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      {mode === "answer"
        ? "Searching the discussions and writing a cited answer. This usually takes 5–15 seconds."
        : "Searching the discussions…"}
    </div>
  );
}
