FROM python:3.12-slim

# Install uv from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies and the package itself in editable mode
# For local dev, we mount the volume so changes are reflected.
COPY pyproject.toml uv.lock ./
# We copy the source so the editable install works (volume mount will override this later)
COPY . .
RUN uv pip install --system -e .[dev]

CMD ["uvicorn", "retrievault.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
