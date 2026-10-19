
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .

# Install PyTorch CPU-only agar tidak mengunduh paket CUDA/NVIDIA
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependensi proyek
RUN pip install -r requirements.txt

COPY main.py .
COPY app/ ./app/
COPY data/raw/ ./data/raw/

EXPOSE 8000

CMD ["sh", "-c", "python -m app.vector_store && python -m uvicorn main:app --host 0.0.0.0 --port 8000"]
