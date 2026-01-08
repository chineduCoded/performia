from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from typing import AsyncGenerator

from app.config import initialize_directories
from app.api.v1.routers import risk, metadata
from app.core.dependencies import load_models

logger = logging.getLogger("uvicorn")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        await load_models()
        logger.info("Models loaded successfully")

        initialize_directories()
        logger.info("Required directories initialized")
    except Exception as e:
        logger.error("Failed during initialization: %s", e)
        raise
    yield

app = FastAPI(
    title="Performia API",
    version="0.1.0",
    description="API for predicting academic risk based on student records.",
    lifespan=lifespan,
)

app.include_router(risk.router, prefix="/api/v1")
app.include_router(metadata.router, prefix="/api/v1")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Performia API!"}