from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/ask")
async def ask_question(request: Request, query: str):
	rag_pipeline = getattr(request.app.state, "rag_pipeline", None)

	if rag_pipeline is None:
		raise HTTPException(status_code=503, detail="RAG pipeline is not ready")

	answer = rag_pipeline.ask(query)
	return {
		"query": query,
		"answer": answer,
	}