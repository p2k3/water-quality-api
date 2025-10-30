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
    """
    Predicts water quality score and pollutant probabilities from input features.
    
    - **ammonia**: Ammonia concentration (mg/l)
    - **bod**: Biochemical Oxygen Demand (mg/l)
    - **dissolved_oxygen**: Dissolved Oxygen (mg/l)
    - **orthophosphate**: Orthophosphate concentration (mg/l)
    - **ph**: pH value (ph units)
    - **temperature**: Water temperature (cel)
    - **nitrogen**: Total Nitrogen (mg/l)
    - **nitrate**: Nitrate concentration (mg/l)
    """
    try:
        # Convert Pydantic models to a NumPy array
        # The order must match the model's training feature order
        input_data = np.array([
            [
                item.ammonia, item.bod, item.dissolved_oxygen, item.orthophosphate,
                item.ph, item.temperature, item.nitrogen, item.nitrate
            ]
            for item in request.inputs
        ])
        
        # Use the global predictor instance to make a prediction
        result = predictor.predict(input_data)
        
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the app locally:
# uvicorn app.main:app --reload
