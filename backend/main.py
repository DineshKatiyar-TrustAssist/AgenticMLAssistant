"""
FastAPI backend for Agentic ML Assistant
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import asyncio
from typing import Optional
from pydantic import BaseModel
from ml_ai_agents import run_pipeline

app = FastAPI(title="Agentic ML Assistant API")

# CORS middleware
# Allow origins from environment variable or default to localhost
allowed_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineRequest(BaseModel):
    api_key: str
    target_variable: str
    model_name: str


@app.post("/api/run-pipeline")
async def run_ml_pipeline(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    target_variable: str = Form(...),
    model_name: str = Form(...)
):
    """Run the multi-agent ML pipeline"""
    try:
        # Set API key
        os.environ["GOOGLE_API_KEY"] = api_key
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(await file.read())
            csv_path = tmp_file.name
        
        # Create task
        task = f"""
        Load the CSV file from {csv_path}, perform Exploratory Data Analysis (EDA),
        clean the data, preprocess features, use '{target_variable}' as the target variable,
        train a {model_name}, evaluate the model using accuracy and other metrics, and output results.
        """
        
        # Run pipeline
        result = await run_pipeline(task)
        
        # Clean up temp file
        if os.path.exists(csv_path):
            os.unlink(csv_path)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable for Cloud Run, default to 8000 for local
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

