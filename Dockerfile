FROM python:3.11-slim

WORKDIR /app

# Force Poetry 2.0+ to support the [project] section
RUN pip install "poetry>=2.0.0"

# model
ARG MODEL_NAME=gpt2

# env var setup 
ENV MODEL_NAME=${MODEL_NAME}
ENV PYTHONPATH=/app/src
ENV HF_HOME=/app/model_cache

# Copy files
COPY pyproject.toml poetry.lock* README.md* ./

# Install dependencies with poetry
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

# Replace PyTorch with CPU-only version (much smaller than CUDA version)
# Note: For GPU version, use Dockerfile.gpu
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# pre-download model
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained('${MODEL_NAME}', cache_dir='${HF_HOME}'); \
    AutoModelForCausalLM.from_pretrained('${MODEL_NAME}', cache_dir='${HF_HOME}')"

COPY . .

ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8000 

CMD ["uvicorn", "ml_ops_exercise.main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t gpt2-llm-service-cpu .
# docker run --network none -it --rm -p 8000:8000 gpt2-llm-service-cpu