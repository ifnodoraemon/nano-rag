# Nano RAG Frontend

React + Vite frontend for exercising the Nano RAG backend during engineering testing.

## Local Development

```bash
cd frontend
npm install
npm run dev
```

By default the Vite dev server proxies API traffic to `http://127.0.0.1:8000`.
If your backend is elsewhere, override the target:

```bash
cd frontend
VITE_DEV_API_TARGET=http://your-backend-host:8000 npm run dev
```

## Production Build

```bash
cd frontend
npm run build
```

The Docker production build serves the static frontend through nginx and proxies API routes back to the backend service.
