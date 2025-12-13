# Sử dụng Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal for OpenCV + PyTorch CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (caching layer)
COPY requirements.txt .

# Install Python dependencies with CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py .
COPY infer.py .

# Create model directory
RUN mkdir -p /app/model

# Copy model file
COPY model/best.pt /app/model/best.pt

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]