FROM python:3.12-slim
WORKDIR /app
ENV NO_ALBUMENTATIONS_UPDATE=1
ENV OMP_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libglib2.0-0 libgl1 libportaudio2 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
