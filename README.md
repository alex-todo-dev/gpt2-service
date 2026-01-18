# GTP2 service docker 

GITHUB: https://github.com/alex-todo-dev/gpt2-service 
DOCKER HUB (cpu version): https://hub.docker.com/repository/docker/alextododev/gpt2-service/general

A FastAPI service providing GPT-2 language model capabilities for text encoding, decoding, and generation.

## Features

- **Encode**: Convert text to token embeddings
- **Decode**: Convert token embeddings back to text
- **Generate**: Generate text from prompts with configurable parameters

## API Endpoints

- `GET /health` - Health check
- `POST /encode` - Encode text to tokens
- `POST /decode` - Decode tokens to text
- `POST /generate` - Generate text from a prompt

## Installation and Setup

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alex-todo-dev/gpt2-service.git
   cd gpt2-service
   ```

2. **Install dependencies with Poetry:**
   ```bash
   poetry install
   ```

3. **Start the service:**
   ```bash
   poetry run uvicorn src.ml_ops_exercise.main:app --reload
   ```

### Docker Build and Run

The project provides two Dockerfiles:

- **`Dockerfile`** - CPU-only version (optimized, smaller size ~3-5GB)
- **`Dockerfile.gpu`** - GPU version with CUDA 12.1 support (requires NVIDIA Docker runtime)

#### CPU Version (Recommended for most use cases)

```bash
# Build the image
docker build -t gpt2-llm-service-cpu .

# Run the container
docker run -it --rm -p 8000:8000 gpt2-llm-service-cpu

```

#### GPU Version (Requires NVIDIA GPU and nvidia-docker)

```bash
# Build the GPU image
docker build -f Dockerfile.gpu -t gpt2-llm-service:gpu .

# Run with GPU support
docker run --gpus all -it --rm -p 8000:8000 gpt2-llm-service:gpu
```

**Note:** Both Docker images:
- Pre-download the model during build time
- Run in offline mode (`TRANSFORMERS_OFFLINE=1`) after build
- Cache models in `/app/model_cache` (set via `HF_HOME`)

**Custom Model:** You can specify a different model during build:
```bash
docker build --build-arg MODEL_NAME=your-model-name -t gpt2-llm-service-cpu .
```

## API Documentation

API documentation is available at `http://localhost:8000/docs` when the service is running.

## Example Usage

**Generate text:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "FAST API IS",
    "temp": 0.2,
    "top_p": 0.95,
    "max_new_tokens": 50,
    "num_return_sequences": 2
  }'
```

**Encode text:**
```bash
curl -X POST http://localhost:8000/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

**Decode tokens:**
```bash
curl -X POST http://localhost:8000/decode \
  -H "Content-Type: application/json" \
  -d '{"tokens": [2504, 318, 616, 2420]}'
```
