FROM python:3.11-slim

WORKDIR /app

# model
ARG MODEL_NAME=gpt2

# env var setup 
ENV MODEL_NAME=${MODEL_NAME}
ENV PYTHONPATH=/app/src
ENV HF_HOME=/app/model_cache
ENV PIP_NO_CACHE_DIR=1

# Install Poetry first
RUN pip install --no-cache-dir "poetry>=2.0.0"

# Copy dependency files
COPY pyproject.toml poetry.lock* ./



# Install all dependencies EXCEPT torch using pip directly no GPU
RUN pip install --no-cache-dir \
    "fastapi>=0.128.0,<0.129.0" \
    "uvicorn>=0.40.0,<0.41.0" \
    "pydantic>=2.0.0" \
    "transformers>=4.0.0" \
    "protobuf"


RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Verify torch is CPU-only and no CUDA packages remain
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Pre-download model
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained('${MODEL_NAME}', cache_dir='${HF_HOME}'); \
    AutoModelForCausalLM.from_pretrained('${MODEL_NAME}', cache_dir='${HF_HOME}')"

COPY . .

ENV TRANSFORMERS_OFFLINE=1

# Final cleanup: remove all caches and temporary files
RUN find /root/.cache -type f -name "*.lock" -delete 2>/dev/null || true && \
    rm -rf /root/.cache/pip /root/.cache/poetry /tmp/* /var/tmp/*

EXPOSE 8000 

CMD ["uvicorn", "ml_ops_exercise.main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t gpt2-llm-service-cpu .
# docker run --network none -it --rm -p 8000:8000 gpt2-llm-service-cpu
# docker run -it --rm -p 8000:8000 gpt2-llm-service-cpu