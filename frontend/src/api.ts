// Types mirror rag/api/schemas.py. Every call goes through post()/get() so errors look the same everywhere.

export interface Source {
  n: number;
  answer_id: number;
  question_id: number;
  thread_title: string;
  url: string;
  thread_url: string;
  votes: number;
  accepted: boolean;
  year: number;
  text: string;
}

export interface AnswerResult {
  question: string;
  found: boolean;
  text: string;
  age_note: string | null;
  sources: Source[];
  context_sources: Source[];
  retrieval: {
    exact_match: boolean;
    best_distance: number | null;
    threads_searched: number;
    threads_in_context: number;
  };
  citations: { invalid_removed: number[]; uncited_sentences: number };
  usage: {
    model: string | null;
    prompt_tokens: number;
    completion_tokens: number;
    cut_off: boolean;
    seconds: number | null;
  };
}

export interface Thread {
  question_id: number;
  title: string;
  url: string;
  excerpt: string;
  votes: number;
  answers: number;
  has_accepted: boolean;
  year: number;
  match: "exact" | "dense";
  distance: number | null;
  relevant: boolean;
}

export interface SearchResult {
  query: string;
  threads: Thread[];
}

export interface Info {
  threads: number;
  llm_model: string;
  embedding_model: string;
  context: string;
  max_question_chars: number;
}

// Where the API lives. Empty: the same address as the page (local dev proxy, or the API serving the page).
// On Vercel: set VITE_API_URL to the Render address, e.g. https://secqa-api.onrender.com
export const API_URL = (import.meta.env.VITE_API_URL ?? "").replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(message: string, public status: number, public requestId?: string) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(API_URL + path, init);
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("Can't reach the server. If nobody has used it for a while it may be starting up: " +
      "please try again in a minute.", 0);
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof body?.detail === "string" ? body.detail : `Request failed (${response.status}).`;
    throw new ApiError(detail, response.status, body?.request_id ?? response.headers.get("X-Request-ID") ?? undefined);
  }
  return body as T;
}

function post<T>(path: string, data: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
    signal,
  });
}

export const ask = (question: string, signal?: AbortSignal) =>
  post<AnswerResult>("/api/ask", { question }, signal);

export const search = (query: string, signal?: AbortSignal) =>
  post<SearchResult>("/api/search", { query }, signal);

export const getInfo = () => request<Info>("/api/info");
