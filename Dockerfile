# The production image: the React page is built first, then copied next to the Python server,
# so one server answers both the page and /api on one address.

# 1) Build the React page into static files
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# 2) The Python server
FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# Libraries first (this layer is reused while only our code changes). --frozen: exactly the
# versions in uv.lock. --no-default-groups: no notebooks, data scripts or test tools.
COPY pyproject.toml uv.lock ./
COPY src/pyproject.toml src/pyproject.toml
RUN uv sync --frozen --no-default-groups --no-install-project --no-install-workspace

COPY rag/ rag/
COPY --from=frontend /app/frontend/dist frontend/dist

# Don't run as root: if the app were ever compromised, the attacker gets a powerless user
RUN useradd --create-home app
USER app

# The host tells us the port in $PORT; one process, because each one holds its own BM25 index
CMD ["sh", "-c", "exec /app/.venv/bin/uvicorn rag.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
