from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from typing import AsyncGenerator

from app.config import initialize_directories
from app.api.v1.routers import risk, metadata
from app.core.dependencies import load_models
from app.utils.format_validation_error import format_validation_error

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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    formatted_errors = format_validation_error(exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "status": "error",
            "message": "Invalid input data",
            "errors": formatted_errors
        }
    )

app.include_router(risk.router, prefix="/api/v1")
app.include_router(metadata.router, prefix="/api/v1")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Performia API!"}