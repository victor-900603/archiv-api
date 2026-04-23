from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path

from rag.pipeline import build_rag_pipeline, build_ingestion_pipeline
from configs.logging import configure_logging
from routers.rag import router as rag_router



@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    ingestion_pipeline = build_ingestion_pipeline()
    example_pdf = Path(__file__).resolve().parent / "docs" / "example" / "example.pdf"

    if example_pdf.exists():
        ingestion_pipeline.ingest(str(example_pdf))

    app.state.rag_pipeline = build_rag_pipeline()
    yield
    
app = FastAPI(lifespan=lifespan)
app.include_router(rag_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Welcome to the Archive API!"}