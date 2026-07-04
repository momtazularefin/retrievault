# retrievault frontend

Next.js chat UI for the retrievault FastAPI-codebase RAG backend.

## Local Development

Run the backend first, then start the frontend from this directory:

```bash
npm install
npm run dev
```

The UI uses `http://localhost:8000` as its default API base URL. For a non-local backend, set `NEXT_PUBLIC_API_URL` in `frontend/.env.local`.

## Checks

```bash
npm run lint
npm run build
```

See the root [README](../README.md) and [Getting Started guide](../docs/getting-started.md) for the full local workflow.
