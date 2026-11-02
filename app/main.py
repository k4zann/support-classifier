import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.classifier import Classifier, ClassifierError, build_client
from app.config import Settings, load_prompt
from app.schemas import (
    BatchError,
    BatchItem,
    BatchRequest,
    BatchResponse,
    ClassifyRequest,
    ClassifyResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    app.state.settings = settings
    prompt = load_prompt(settings.prompt_path)
    client = build_client(settings)
    app.state.classifier = Classifier(client, prompt, settings)
    yield
    await client.aio.aclose()


app = FastAPI(title="Support classifier", lifespan=lifespan)


@app.exception_handler(ClassifierError)
async def classifier_error_handler(
    request: Request, exc: ClassifierError
) -> JSONResponse:
    # The traceback (with the chained provider error) goes to the log only;
    # the client gets a fixed message so no provider text or secrets leak.
    logger.error("classification failed", exc_info=exc)
    return JSONResponse(status_code=502, content={"detail": "classification failed"})


def get_classifier(request: Request) -> Classifier:
    # The single test seam: tests replace this via app.dependency_overrides.
    return request.app.state.classifier


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify")
async def classify(
    body: ClassifyRequest, classifier: Annotated[Classifier, Depends(get_classifier)]
) -> ClassifyResponse:
    return await classifier.classify(body.text)


@app.post("/classify/batch")
async def classify_batch(
    body: BatchRequest,
    classifier: Annotated[Classifier, Depends(get_classifier)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> BatchResponse:
    semaphore = asyncio.Semaphore(settings.batch_concurrency)

    async def classify_one(text: str) -> BatchItem:
        async with semaphore:
            try:
                return await classifier.classify(text)
            except ClassifierError as exc:
                # Same contract as the 502 handler: log the cause, return fixed text.
                logger.error("classification failed", exc_info=exc)
                return BatchError(error="classification failed")

    # gather keeps the input order; per-item errors never fail the whole batch.
    results = await asyncio.gather(*(classify_one(text) for text in body.texts))
    return BatchResponse(results=list(results))
