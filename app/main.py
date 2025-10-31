from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from .predictor import predictor # Import the initialized predictor instance
from .schemas import PredictionRequest, PredictionResponse

# Define the application
app = FastAPI(
    title="Water Quality Monitoring API",
    description="An API to predict water quality using a Symbolic Graph Attention Network (SymGAT) model.",
    version="1.0.0"
)

@app.get("/", tags=["Health Check"])
def read_root():
    """Health check endpoint to confirm the API is running."""
    return {"status": "Water Quality API is running."}

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict_water_quality(request: PredictionRequest):
    try:
        input_data = np.array([
            [
                feature.ammonia,
                feature.bod,
                feature.dissolved_oxygen,
                feature.orthophosphate,
                feature.ph,
                feature.temperature,
                feature.nitrogen,
                feature.nitrate
            ]
            for feature in request.features
        ])
        result = predictor.predict(input_data)
        explanation = predictor.explain(input_data)
        return {
            "predictions": result["predictions"],
            "pollutant_probabilities": result["pollutant_probabilities"],
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# To run the app locally:
# uvicorn app.main:app --reload
