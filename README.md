# ML Ops Exercise

A FastAPI service providing GPT-2 language model capabilities for text encoding, decoding, and generation.

## Features

- **Encode**: Convert text to token embeddings
- **Decode**: Convert token embeddings back to text
- **Generate**: Generate text from prompts with configurable parameters

## Quick Start

### Local Development

1. Install dependencies:
```bash
poetry install
```

2. Run the service:
```bash
poetry run uvicorn src.ml_ops_exercise.main:app --reload
```

The API will be available at `http://localhost:8000`

### Docker

Build and run with Docker:
```bash
docker build -t gpt2-llm-service .
docker run -p 8000:8000 gpt2-llm-service
```

## API Endpoints

- `GET /health` - Health check
- `POST /encode` - Encode text to tokens
- `POST /decode` - Decode tokens to text
- `POST /generate` - Generate text from a prompt

API documentation available at `http://localhost:8000/docs`

## Example Usage

**Generate text:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI", "max_new_tokens": 50}'
```

**Encode text:**
```bash
curl -X POST http://localhost:8000/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

## Requirements

- Python 3.10-3.14
- Poetry for dependency management

