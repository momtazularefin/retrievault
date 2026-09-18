# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Pinned so an image build cannot pick up a different resolver than the one that wrote uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.11.23 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    # fastembed defaults to the system temp directory, which is cleared between runs; keep the
    # downloaded ONNX models on a volume instead.
    FASTEMBED_CACHE_PATH=/models

# Dependencies first, from the lockfile only, so the layer is cached and the image matches
# what CI resolved. --no-dev leaves out pytest, ruff, and the evaluation stack.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY retrievault ./retrievault
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 retrievault \
    && mkdir -p /models && chown -R retrievault:retrievault /models /app
USER retrievault

EXPOSE 8000
# No --reload: that is a development flag and it doubles the process count in a container.
CMD ["uvicorn", "retrievault.api:app", "--host", "0.0.0.0", "--port", "8000"]
