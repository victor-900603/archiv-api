from fastapi import FastAPI
from contextlib import asynccontextmanager

from .configs.logging import configure_logging



@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    yield
    
app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Welcome to the Archive API!"}