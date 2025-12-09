# Docker Deployment Guide

This guide explains how to deploy the Agentic ML Assistant application using Docker.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+

## Quick Start

### Option 1: Combined Container (Single Container)

Deploy both frontend and backend in a single container:

```bash
# Build and run combined container
docker build -t agentic-ml-assistant .
docker run -d -p 8000:8000 -p 3000:3000 --name agentic-ml agentic-ml-assistant

# Or using docker-compose
docker-compose -f docker-compose.combined.yml up -d
```

### Option 2: Separate Containers (Docker Compose)

Deploy frontend and backend as separate containers:

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000

## Deployment Options

### Combined Container (Recommended for Simple Deployments)

The `Dockerfile` in the root directory builds both frontend and backend in a single container. This is ideal for:
- Simple deployments
- Single-server setups
- Reduced resource usage
- Easier management

**Build and run:**
```bash
docker build -t agentic-ml-assistant .
docker run -d -p 8000:8000 -p 3000:3000 \
  -e CORS_ORIGINS=http://localhost:3000 \
  -e NEXT_PUBLIC_API_URL=http://localhost:8000 \
  --name agentic-ml \
  agentic-ml-assistant
```

**Using docker-compose:**
```bash
docker-compose -f docker-compose.combined.yml up -d
```

### Separate Containers (Recommended for Production)

Use separate containers for better scalability and resource management:

**Build and run:**
```bash
docker-compose up -d
```

## Building Individual Services

### Backend Only

```bash
cd backend
docker build -t agentic-ml-backend .
docker run -p 8000:8000 agentic-ml-backend
```

### Frontend Only

```bash
cd frontend
docker build -t agentic-ml-frontend .
docker run -p 3000:3000 agentic-ml-frontend
```

## Docker Compose Configuration

The `docker-compose.yml` file orchestrates both services:

- **Backend**: FastAPI application running on port 8000
- **Frontend**: Next.js application running on port 3000

### Environment Variables

#### Backend
- `GOOGLE_GENAI_USE_VERTEXAI`: Set to `FALSE` (default)
- `CORS_ORIGINS`: Comma-separated list of allowed origins (default: `http://localhost:3000,http://localhost:3001`)

#### Frontend
- `NODE_ENV`: Set to `production`
- `NEXT_PUBLIC_API_URL`: Backend API URL (default: `http://localhost:8000`)

## Production Deployment

### Using Docker Compose

1. **Update environment variables** in `docker-compose.yml`:
   ```yaml
   environment:
     - CORS_ORIGINS=https://yourdomain.com
     - NEXT_PUBLIC_API_URL=https://api.yourdomain.com
   ```

2. **Build for production**:
   ```bash
   docker-compose -f docker-compose.yml build
   ```

3. **Run in detached mode**:
   ```bash
   docker-compose up -d
   ```

### Using Individual Dockerfiles

#### Backend Production

```bash
docker build -t agentic-ml-backend:latest ./backend
docker run -d \
  -p 8000:8000 \
  -e CORS_ORIGINS=https://yourdomain.com \
  --name agentic-ml-backend \
  agentic-ml-backend:latest
```

#### Frontend Production

```bash
docker build -t agentic-ml-frontend:latest ./frontend
docker run -d \
  -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=https://api.yourdomain.com \
  --name agentic-ml-frontend \
  agentic-ml-frontend:latest
```

## Health Checks

The backend includes a health check endpoint at `/api/health`. Docker Compose uses this to ensure the backend is ready before starting the frontend.

## Volumes

The `docker-compose.yml` includes a volume mount for the backend to enable hot-reloading during development. Remove this in production:

```yaml
# Remove this line for production
volumes:
  - ./backend:/app
```

## Troubleshooting

### Backend won't start

1. Check logs: `docker-compose logs backend`
2. Verify Python dependencies are installed
3. Ensure port 8000 is not already in use

### Frontend can't connect to backend

1. Verify `NEXT_PUBLIC_API_URL` is set correctly
2. Check CORS configuration in backend
3. Ensure backend is healthy: `curl http://localhost:8000/api/health`

### Build fails

1. Clear Docker cache: `docker system prune -a`
2. Rebuild without cache: `docker-compose build --no-cache`
3. Check Dockerfile syntax

## Development Mode

For development with hot-reloading:

```bash
# Backend (with volume mount)
docker-compose up backend

# Frontend (run locally for hot-reload)
cd frontend && npm run dev
```

## Security Considerations

1. **Never commit API keys** - Use environment variables or secrets management
2. **Update CORS origins** - Restrict to your domain in production
3. **Use HTTPS** - Configure reverse proxy (nginx/traefik) for SSL
4. **Limit resources** - Add resource limits in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '1'
         memory: 2G
   ```

## Scaling

To scale the backend:

```bash
docker-compose up -d --scale backend=3
```

Use a load balancer (nginx, traefik) to distribute traffic across multiple backend instances.

## Monitoring

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
```

### Check container status
```bash
docker-compose ps
```

### Resource usage
```bash
docker stats
```

