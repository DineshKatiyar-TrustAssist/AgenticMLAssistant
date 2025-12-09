# Combined Dockerfile for Frontend and Backend
# This Dockerfile builds both the Next.js frontend  and FastAPI backend in a single container

# Stage 1: Build Frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy frontend package files
COPY frontend/package*.json ./

# Install frontend dependencies
RUN npm ci --only=production=false

# Copy frontend source code
COPY frontend/ .

# Build Next.js application
RUN npm run build

# Stage 2: Build Backend and Combine
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for both Python and Node.js
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for running the frontend
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install Python dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

# Copy backend application code
COPY backend/ ./backend/

# Copy frontend build from builder stage
# Next.js standalone output structure: .next/standalone contains server.js and .next/static
COPY --from=frontend-builder /app/frontend/.next/standalone ./
COPY --from=frontend-builder /app/frontend/.next/static ./.next/static

# Create startup script to run both services
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Function to handle shutdown\n\
cleanup() {\n\
    echo "Shutting down services..."\n\
    kill -TERM $BACKEND_PID $FRONTEND_PID 2>/dev/null || true\n\
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true\n\
    exit 0\n\
}\n\
\n\
# Trap signals for graceful shutdown\n\
trap cleanup SIGTERM SIGINT\n\
\n\
# Start backend in background (only accessible from within container)\n\
cd /app/backend\n\
echo "Starting backend on port 8000 (localhost only)..."\n\
uvicorn main:app --host 127.0.0.1 --port 8000 &\n\
BACKEND_PID=$!\n\
echo "Backend started with PID: $BACKEND_PID"\n\
\n\
# Wait for backend to be ready\n\
echo "Waiting for backend to be ready..."\n\
for i in {1..30}; do\n\
  if python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" 2>/dev/null; then\n\
    echo "Backend is ready!"\n\
    break\n\
  fi\n\
  sleep 1\n\
done\n\
\n\
# Start frontend in background\n\
cd /app\n\
NODE_ENV=production PORT=3000 HOSTNAME="0.0.0.0" node server.js &\n\
FRONTEND_PID=$!\n\
echo "Frontend started with PID: $FRONTEND_PID"\n\
\n\
# Wait for both processes\n\
wait $BACKEND_PID $FRONTEND_PID\n\
' > /app/start.sh && chmod +x /app/start.sh

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose ports (only frontend needs external access)
EXPOSE 3000

# Health check for backend (using localhost since backend is internal)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

# Run startup script
CMD ["/app/start.sh"]

