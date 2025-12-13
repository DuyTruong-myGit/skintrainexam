# Sử dụng Python slim image để giảm dung lượng
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Fixed for Debian Trixie)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    libgl1 \
    libglx0 \
    libegl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (caching layer)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py .
COPY infer.py .

# Create model directory
RUN mkdir -p /app/model

# Copy model file
COPY model/best.pt /app/model/best.pt

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]