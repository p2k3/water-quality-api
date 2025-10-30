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
    
    Features required:
    - **ph**: pH level (0-14)
    - **hardness**: Water hardness (mg/L)
    - **solids**: Total dissolved solids (ppm)
    - **chloramines**: Chloramines level (ppm)
    - **sulfate**: Sulfates (mg/L)
    - **conductivity**: Electrical conductivity (μS/cm)
    - **organic_carbon**: Organic carbon (ppm)
    - **trihalomethanes**: Trihalomethanes (μg/L)
    - **turbidity**: Turbidity (NTU)
    """
    try:
        # Convert Pydantic models to a NumPy array
        # The order must match the model's training feature order
        input_data = np.array([
            [
                feature.ph,
                feature.hardness,
                feature.solids,
                feature.chloramines,
                feature.sulfate,
                feature.conductivity,
                feature.organic_carbon,
                feature.trihalomethanes,
                feature.turbidity
            ]
            for feature in request.features
        ])
        
        # Use the global predictor instance to make a prediction
        result = predictor.predict(input_data)
        
        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the app locally:
# uvicorn app.main:app --reload
