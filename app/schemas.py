from enum import Enum

from pydantic import BaseModel, Field


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


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    category: Category
    urgency: Urgency
    summary: str = Field(min_length=1, max_length=150)


class ModelOutput(BaseModel):
    """Schema sent to Gemini as `response_schema`.

    `summary` has no max length on purpose: an overlong summary must still parse,
    so it can be trimmed instead of failing the whole classification.
    """

    category: Category
    urgency: Urgency
    summary: str
