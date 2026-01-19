from fastapi import FastAPI

from .data_classes import GenerationModel, EncoderModel, DecoderModel
from .llm_service import llm_service

app = FastAPI(title="GPT-2 API")


@app.get("/health")
def main_test() -> dict[str, str]:
    """Health check"""
    return {"status": "HEALTHY"}


@app.post("/encode")
async def encode(data: EncoderModel):
    """Encode text into token embeddings."""
    tokens = llm_service.encode(data.text)
    return {"tokens": tokens, "count": len(tokens)}


@app.post("/decode")
async def decode(data: DecoderModel):
    """Decode token embeddings to text."""
    text = llm_service.decode(data.tokens)
    return {"text": text}


@app.post("/generate")
async def generate(data: GenerationModel):
    """Generate text using the language model."""
    text = llm_service.generate(
        data.prompt,
        temp=data.temp,
        top_p=data.top_p,
        max_new_tokens=data.max_new_tokens,
        num_return_sequences=data.num_return_sequences
    )
    return {"result": text}


# poetry run uvicorn src.ml_ops_exercise.main:app --reload
