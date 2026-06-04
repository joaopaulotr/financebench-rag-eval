import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graph import app as graph_app

api = FastAPI(title="FinanceBench RAG — CRAG API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

NODE_LABELS = {
    "query_analysis": "Analyzing query…",
    "retrieve": "Retrieving documents…",
    "rerank": "Reranking results…",
    "grade_documents": "Grading relevance…",
    "relax_filter": "Relaxing filters, retrying…",
    "generate": "Generating answer…",
}


class ChatRequest(BaseModel):
    query: str


def _initial_state(query: str) -> dict:
    return {
        "query": query,
        "filter_token": "",
        "company_filter": "",
        "context": "",
        "sources": [],
        "grade": "",
        "answer": "",
        "retry_count": 0,
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _event_stream(query: str):
    answer = ""
    sources: list[str] = []
    try:
        for step in graph_app.stream(_initial_state(query)):
            for node, update in step.items():
                label = NODE_LABELS.get(node, node)
                yield _sse({"type": "status", "node": node, "label": label})
                if isinstance(update, dict):
                    if update.get("sources"):
                        sources = update["sources"]
                    if update.get("answer"):
                        answer = update["answer"]
        yield _sse({"type": "done", "answer": answer, "sources": sources})
    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})


@api.post("/chat")
def chat(req: ChatRequest):
    return StreamingResponse(
        _event_stream(req.query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api.get("/health")
def health():
    return {"status": "ok"}
