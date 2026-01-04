from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from typing import AsyncGenerator

from app.api.v1.routes import risk, metadata
from app.services.model_loader import ModelLoader

logger = logging.getLogger("uvicorn")

@asynccontextmanager
async def load_models(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        ModelLoader.load_model("risk", "risk_model.joblib")
        logger.info("Risk model loaded successfully")
    except Exception as e:
        logger.error("Failed to load risk model: %s", e)
        raise
    yield

app = FastAPI(
    title="Performia API",
    version="0.1.0",
    description="API for predicting academic risk based on student records.",
    lifespan=load_models,
)

app.include_router(risk.router, prefix="/api/v1")
app.include_router(metadata.router, prefix="/api/v1")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Performia API!"}