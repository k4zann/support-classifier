FROM python:3.12.14-slim

# uv binary from the official image, same version as used locally.
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /bin/uv

WORKDIR /app

# Install dependencies first, so this layer is cached until the lockfile changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# App code and prompts; .dockerignore keeps .env and other local files out.
COPY . .

EXPOSE 8000
# --no-sync: dependencies were installed at build time, don't re-sync (or add dev deps) on start.
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
