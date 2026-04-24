from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from rag.pipeline import RAGPipeline

router = APIRouter(prefix="/rag", tags=["rag"])


class AskRequest(BaseModel):
    query: str
    history: list[dict] | None = None


@router.post("/ask")
async def ask_question(request: Request, ask_request: AskRequest):
    rag_pipeline: RAGPipeline | None = getattr(request.app.state, "rag_pipeline", None)

    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not ready")

    answer = rag_pipeline.ask(ask_request.query, ask_request.history)
    return {
        "status": "success",
        "answer": answer,
    }