"""
FastAPI application for Handwritten Equation Solver.

Provides REST API endpoints for:
- Uploading equation images
- Getting predictions and results
- Health checks
"""

import os
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from .preprocessing.image_utils import (
    load_image_from_base64,
    preprocess_image,
    normalize_for_model
)
from .preprocessing.segmentation import CharacterSegmenter
from .model.predictor import EquationPredictor
from .utils.equation_parser import EquationParser


# Configuration
MODEL_PATH = Path(__file__).parent.parent / "models" / "equation_solver_model.h5"


# Pydantic models for API
class ImageInput(BaseModel):
    """Request body for image input."""
    image: str  # Base64 encoded image


class PredictionResult(BaseModel):
    """Response model for predictions."""
    recognized_equation: str
    result: Optional[float]
    formatted_result: str
    display_equation: str
    confidence_scores: list
    error: Optional[str]
    success: bool


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    version: str


# Global instances
predictor: Optional[EquationPredictor] = None
segmenter: Optional[CharacterSegmenter] = None
parser: Optional[EquationParser] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global predictor, segmenter, parser
    
    # Startup
    print("Starting Handwritten Equation Solver API...")
    
    # Initialize components
    predictor = EquationPredictor()
    segmenter = CharacterSegmenter()
    parser = EquationParser()
    
    # Load model if exists
    if MODEL_PATH.exists():
        predictor.load_model(str(MODEL_PATH))
        print(f"Model loaded from: {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")
        print("Please train the model first using the training script.")
    
    yield
    
    # Shutdown
    print("Shutting down Handwritten Equation Solver API...")


# Create FastAPI app
app = FastAPI(
    title="Handwritten Equation Solver API",
    description="API for recognizing and solving handwritten mathematical equations using CNN",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health status."""
    return HealthResponse(
        status="online",
        model_loaded=predictor.is_loaded() if predictor else False,
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded() if predictor else False,
        version="1.0.0"
    )


@app.post("/solve", response_model=PredictionResult)
async def solve_equation(input_data: ImageInput):
    """
    Solve a handwritten equation from an image.
    
    Accepts a base64-encoded image and returns:
    - The recognized equation
    - The computed result
    - Confidence scores for each character
    """
    global predictor, segmenter, parser
    
    if not predictor or not predictor.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Load and preprocess image
        image = load_image_from_base64(input_data.image)
        
        # Segment characters
        segments = segmenter.segment(image)
        
        if not segments:
            return PredictionResult(
                recognized_equation="",
                result=None,
                formatted_result="",
                display_equation="",
                confidence_scores=[],
                error="No characters detected in the image",
                success=False
            )
        
        # Preprocess each character and predict
        char_images = []
        for seg in segments:
            processed = preprocess_image(seg.image, target_size=(28, 28))
            char_images.append(processed)
        
        # Get predictions
        equation_str, predictions = predictor.predict_equation(char_images)
        
        # Parse and evaluate
        parse_result = parser.parse_and_evaluate(equation_str)
        
        # Format results
        confidence_scores = [
            {"char": char, "confidence": round(conf, 4)}
            for char, conf in predictions
        ]
        
        formatted_result = ""
        if parse_result.is_valid:
            formatted_result = parser.format_result(parse_result.result)
        
        display_equation = parser.format_equation_for_display(equation_str)
        
        return PredictionResult(
            recognized_equation=equation_str,
            result=parse_result.result,
            formatted_result=formatted_result,
            display_equation=display_equation,
            confidence_scores=confidence_scores,
            error=parse_result.error,
            success=parse_result.is_valid
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/upload", response_model=PredictionResult)
async def upload_and_solve(file: UploadFile = File(...)):
    """
    Upload an image file and solve the equation.
    
    Alternative to the /solve endpoint for file uploads.
    """
    import base64
    
    if not predictor or not predictor.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Read file
        contents = await file.read()
        
        # Convert to base64
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # Use the solve endpoint logic
        input_data = ImageInput(image=base64_image)
        return await solve_equation(input_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing uploaded file: {str(e)}"
        )


@app.get("/model/status")
async def model_status():
    """Get detailed model status."""
    return {
        "loaded": predictor.is_loaded() if predictor else False,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "classes": predictor.class_labels if predictor else []
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

