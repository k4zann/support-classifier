from enum import Enum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field


# str Enums: Gemini's schema conversion maps them reliably to a JSON `enum`.
class Category(str, Enum):
    auth = "auth"
    payment = "payment"
    delivery = "delivery"
    technical = "technical"
    other = "other"


class Urgency(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


def _not_blank(text: str) -> str:
    if not text.strip():
        raise ValueError("text must not be empty")
    return text


# Shared by ClassifyRequest and each item of BatchRequest.
# The 4000-character limit applies to the text as sent, before any stripping.
SupportText = Annotated[str, Field(max_length=4000), AfterValidator(_not_blank)]


class ClassifyRequest(BaseModel):
    text: SupportText


class ClassifyResponse(BaseModel):
    category: Category
    urgency: Urgency
    summary: str = Field(min_length=1, max_length=150)


class BatchRequest(BaseModel):
    texts: list[SupportText] = Field(min_length=1, max_length=50)


class BatchError(BaseModel):
    error: str


BatchItem = ClassifyResponse | BatchError


class BatchResponse(BaseModel):
    results: list[BatchItem]


class ModelOutput(BaseModel):
    """Schema sent to Gemini as `response_schema`.

    `summary` has no max length on purpose: an overlong summary must still parse,
    so it can be trimmed instead of failing the whole classification.
    """

    category: Category
    urgency: Urgency
    summary: str
