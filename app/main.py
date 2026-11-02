import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai

from app.classifier import Classifier, ClassifierError
from app.config import Settings, load_prompt
from app.schemas import ClassifyRequest, ClassifyResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    prompt = load_prompt(settings.prompt_path)
    client = genai.Client(api_key=settings.gemini_api_key)
    app.state.classifier = Classifier(client, prompt, settings)
    yield


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify")
async def classify(
    body: ClassifyRequest, classifier: Annotated[Classifier, Depends(get_classifier)]
) -> ClassifyResponse:
    return await classifier.classify(body.text)
