# FinanceBench RAG — Frontend

Minimal text-only chat UI for the LangGraph CRAG pipeline. Vite + React + TypeScript + Tailwind, shadcn project structure (`@/` → `src/`, UI in `src/components/ui/`). No file/image input — sends text, renders the answer + source documents, and shows live pipeline status (analyzing → retrieving → reranking → grading → generating) streamed over SSE.

## Run

**1. Backend (FastAPI streaming wrapper):**

```bash
# from repo root — Qdrant must be up (docker compose up -d) and OPENAI_API_KEY set
cd backend/CoreRefactoryLangGraph
uv run uvicorn api:api --reload --port 8000
```

**2. Frontend:**

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Backend URL defaults to `http://localhost:8000`. Override with `VITE_API_URL` (see `.env.example`).

## Notes

- Latency is ~40s/query (GPT-4o generation). The status line updates per graph node while you wait.
- CORS in `api.py` allows `localhost:5173`. Change it there if you serve the frontend elsewhere.
