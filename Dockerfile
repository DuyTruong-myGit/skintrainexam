FROM python:3.10-slim

WORKDIR /app

# Cài lib cần cho opencv / torch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Chỉ cài dependency inference
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy đúng những gì cần
COPY best.pt .
COPY infer.py .
COPY api.py .

EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
