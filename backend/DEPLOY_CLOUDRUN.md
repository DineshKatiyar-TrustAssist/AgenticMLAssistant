# Deploying Backend to Google Cloud Run

This guide explains how to deploy the Agentic ML Assistant backend to Google Cloud Run.

## Prerequisites

1. **Google Cloud Project**: Create or select a Google Cloud project
2. **Enable APIs**: Enable the following APIs:
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```
3. **Authentication**: Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   gcloud auth configure-docker
   gcloud config set project YOUR_PROJECT_ID
   ```

## Deployment Methods

### Method 1: Build and Deploy with Docker (Recommended)

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Build the Docker image**:
   ```bash
   docker build -t gcr.io/YOUR_PROJECT_ID/agentic-ml-backend .
   ```

3. **Push to Container Registry**:
   ```bash
   docker push gcr.io/YOUR_PROJECT_ID/agentic-ml-backend
   ```

4. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy agentic-ml-backend \
     --image gcr.io/YOUR_PROJECT_ID/agentic-ml-backend \
     --region us-central1 \
     --platform managed \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 2 \
     --timeout 3600 \
     --max-instances 10 \
     --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=FALSE
   ```

### Method 2: Direct Deploy (Simplest - Builds Docker for you)

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Deploy directly** (Cloud Run will build the Docker image):
   ```bash
   gcloud run deploy agentic-ml-backend \
     --source . \
     --region us-central1 \
     --platform managed \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 2 \
     --timeout 3600 \
     --max-instances 10 \
     --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=FALSE
   ```

## Configuration Options

### Resource Allocation

- **Memory**: 2Gi (recommended for ML workloads, can be increased to 4Gi or 8Gi)
- **CPU**: 2 (can be increased for faster processing)
- **Timeout**: 3600 seconds (1 hour) for long-running ML tasks
- **Max Instances**: 10 (adjust based on traffic)

### Environment Variables

Set environment variables for your Cloud Run service:

```bash
gcloud run services update agentic-ml-backend \
  --region us-central1 \
  --set-env-vars CORS_ORIGINS=https://your-frontend-domain.com
```

### CORS Configuration

Update CORS origins to allow your frontend:

```bash
gcloud run services update agentic-ml-backend \
  --region us-central1 \
  --set-env-vars CORS_ORIGINS=https://your-frontend-domain.com,https://localhost:3000
```

## Getting the Service URL

After deployment, get your service URL:

```bash
gcloud run services describe agentic-ml-backend \
  --region us-central1 \
  --format 'value(status.url)'
```

The URL will be in the format: `https://agentic-ml-backend-xxxxx-uc.a.run.app`

## Testing the Deployment

1. **Health Check**:
   ```bash
   curl https://YOUR_SERVICE_URL/api/health
   ```

2. **Test Pipeline** (using curl with form data):
   ```bash
   curl -X POST https://YOUR_SERVICE_URL/api/run-pipeline \
     -F "file=@test.csv" \
     -F "api_key=YOUR_API_KEY" \
     -F "target_variable=target" \
     -F "model_name=RandomForestClassifier"
   ```

## Updating the Deployment

To update the service after making code changes:

1. **Rebuild the Docker image**:
   ```bash
   cd backend
   docker build -t gcr.io/YOUR_PROJECT_ID/agentic-ml-backend .
   ```

2. **Push the updated image**:
   ```bash
   docker push gcr.io/YOUR_PROJECT_ID/agentic-ml-backend
   ```

3. **Redeploy to Cloud Run**:
   ```bash
   gcloud run deploy agentic-ml-backend \
     --image gcr.io/YOUR_PROJECT_ID/agentic-ml-backend \
     --region us-central1
   ```

Or use the direct deploy method which rebuilds automatically:
```bash
cd backend
gcloud run deploy agentic-ml-backend \
  --source . \
  --region us-central1
```

## Monitoring

View logs:
```bash
gcloud run services logs read agentic-ml-backend --region us-central1
```

View metrics in Cloud Console:
- Go to Cloud Run → agentic-ml-backend → Metrics

## Cost Optimization

- **Min Instances**: Set to 0 to scale to zero when not in use
- **Max Instances**: Limit based on expected traffic
- **CPU Allocation**: Use "CPU is only allocated during request processing" for cost savings
- **Memory**: Right-size based on actual usage

## Troubleshooting

### Common Issues

1. **Timeout Errors**:
   - Increase timeout: `--timeout 3600`
   - Optimize ML pipeline code

2. **Memory Errors**:
   - Increase memory: `--memory 4Gi`

3. **CORS Errors**:
   - Update CORS_ORIGINS environment variable
   - Ensure frontend URL is in allowed origins

4. **API Key Issues**:
   - API keys are provided by users via the UI
   - No need to set GOOGLE_API_KEY in Cloud Run

## Security Best Practices

1. **Authentication**: Consider requiring authentication for production:
   ```bash
   gcloud run services update agentic-ml-backend \
     --region us-central1 \
     --no-allow-unauthenticated
   ```

2. **IAM**: Use IAM to control who can invoke the service

3. **VPC**: Connect to VPC for private resources if needed

4. **Secrets**: Use Secret Manager for sensitive data (if needed in future)

## Next Steps

1. Deploy frontend separately (or use Cloud Run for frontend too)
2. Set up custom domain
3. Configure monitoring and alerting
4. Set up CI/CD pipeline

