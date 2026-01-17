from pydantic import BaseModel, Field

from .config import (
    DEFAULT_MAX_NEW_TOKENS,
    MAX_MAX_NEW_TOKENS,
    MIN_MAX_NEW_TOKENS,
    DEFAULT_NUM_RETURN_SEQUENCES,
    MIN_NUM_RETURN_SEQUENCES,
    MAX_NUM_RETURN_SEQUENCES,
    DEFAULT_TEMP,
    MIN_TEMP,
    MAX_TEMP,
    DEFAULT_TOP_P,
    MIN_TOP_P,
    MAX_TOP_P,
    MIN_TEXT_LENGTH,
    MIN_TOKENS_LENGTH,
    MAX_TOKENS_LENGTH,
)


class GenerationModel(BaseModel):
    """Model for text generation requests."""
    prompt: str = Field(
        ..., min_length=MIN_TEXT_LENGTH, description="User prompt"
    )
    max_new_tokens: int = Field(
        default=DEFAULT_MAX_NEW_TOKENS,
        gt=MIN_MAX_NEW_TOKENS - 1,
        le=MAX_MAX_NEW_TOKENS,
        description="Max output token count :input prompt + generated output"
    )
    num_return_sequences: int = Field(
        default=DEFAULT_NUM_RETURN_SEQUENCES,
        ge=MIN_NUM_RETURN_SEQUENCES,
        le=MAX_NUM_RETURN_SEQUENCES,
        description="How many new tokens the LLM can add"
    )
    temp: float = Field(
        default=DEFAULT_TEMP,
        ge=MIN_TEMP,
        le=MAX_TEMP,
        description="Creativity level"
    )
    top_p: float = Field(
        default=DEFAULT_TOP_P,
        ge=MIN_TOP_P,
        le=MAX_TOP_P,
        description="Creativity level"
    )


class EncoderModel(BaseModel):
    """Model for text encoding requests."""
    text: str = Field(..., min_length=MIN_TEXT_LENGTH, description="Input text for encoding")


class DecoderModel(BaseModel):
    """Model for token decoding requests."""
    tokens: list[int] = Field(
        ...,
        min_length=MIN_TOKENS_LENGTH,
        max_length=MAX_TOKENS_LENGTH,
        description="Input list embeddings for decoding"
    )