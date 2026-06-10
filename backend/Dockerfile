FROM python:3.12-slim

WORKDIR /app

# Install dependencies and the package itself in editable mode
# For local dev, we mount the volume so changes are reflected.
COPY pyproject.toml .
RUN pip install --no-cache-dir hatchling
# We copy the source so pip install -e . works, but the volume mount will override it later
COPY . .
RUN pip install --no-cache-dir -e .[dev]

CMD ["uvicorn", "retrievault.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
