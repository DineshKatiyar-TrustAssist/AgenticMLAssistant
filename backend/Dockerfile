# Backend Dockerfile for FastAPI application (Google Cloud Run optimized)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user (Cloud Run best practice)
RUN useradd -m -u 1000 appuser

# Cloud Run sets PORT environment variable
# Default to 8080 if not set (Cloud Run default)
ENV PORT=8080

# Expose port (Cloud Run uses PORT env var, but EXPOSE needs a literal)
EXPOSE 8080

# Create startup script to handle PORT env var dynamically
RUN echo '#!/bin/sh\n\
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --timeout-keep-alive 300 --workers 1\n\
' > /app/start.sh && chmod +x /app/start.sh && \
    chown -R appuser:appuser /app

USER appuser

# Run the application
# Cloud Run requires listening on 0.0.0.0 and using PORT env var
# Increased timeout for long-running ML tasks
CMD ["/app/start.sh"]
