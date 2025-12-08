# Stage 1: Builder
FROM python:3.10-slim as builder

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install Runtime Libs (OpenCV, PDF)
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy Code
COPY . .

# Security: Run as non-root
RUN useradd -m appuser
USER appuser

# Environment
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app