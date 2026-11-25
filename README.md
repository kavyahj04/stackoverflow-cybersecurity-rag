# Security Q&A Research

Cited answers to security questions, written from 57,141 Security Stack Exchange threads.

## Run it

Needs `rag-dev/.env` with `DATABASE_URL` and `OPENAI_API_KEY`.

**One server (what researchers use):**

```bash
cd frontend && npm install && npm run build && cd ..   # once, and after frontend changes
.venv/bin/uvicorn rag.api.app:app --port 8000
```

Open http://localhost:8000. The API docs are at http://localhost:8000/docs. Startup takes about 10 s, because the keyword index is built in memory.

**While changing the frontend** (the page reloads on save):

```bash
.venv/bin/uvicorn rag.api.app:app --port 8000 --reload   # terminal 1
cd frontend && npm run dev                               # terminal 2: open http://localhost:5173
```

**From the terminal:** `.venv/bin/python -m rag.ask "How do I prevent SQL injection in PHP?"`

## Production

The API runs on Render (Docker), and the page runs on Vercel.

| Piece | File | Settings |
|---|---|---|
| API | `Dockerfile`, `render.yaml` | `DATABASE_URL`, `OPENAI_API_KEY`, `CORS_ORIGINS` (the Vercel address), `CLIENT_IP_HEADER=True-Client-IP`, `ASK_PER_MINUTE`, `ASK_PER_DAY` |
| Page | `frontend/vercel.json` | `VITE_API_URL` (the Render address) |

Build and run the production image locally:
`docker build -t secqa . && docker run --env-file .env -p 8080:8080 -e PORT=8080 secqa`

## Tests

```bash
.venv/bin/python -m pytest                               # unit + API tests, no database or OpenAI
RUN_DB_TESTS=1 .venv/bin/python -m pytest -m db          # checks against the real database
```

## Layout

| Path | What it does |
|---|---|
| `rag/retrieval/` | dense search (pgvector HNSW), BM25 for exact identifiers, hybrid routing, thread store |
| `rag/context.py` | picks the best answers per thread and fits the token budget |
| `rag/generation/` | prompt, LLM call, citation check, age note |
| `rag/api/` | FastAPI: `/api/ask`, `/api/search`, `/api/health`, `/api/info`; serves the built frontend |
| `frontend/` | React + Vite page |
| `rag/eval/` | retrieval and answer evaluations; reports in `reports/` |
