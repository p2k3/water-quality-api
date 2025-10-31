from pydantic import BaseModel
from typing import List

from pydantic import Field

class WaterFeatures(BaseModel):
    ammonia: float = Field(..., description="Ammonia (mg/L). Acceptable range: 0.001–0.5", example=0.2)
    bod: float = Field(..., description="Biochemical Oxygen Demand (mg/L). Acceptable range: 0.001–5.0", example=3.5)
    dissolved_oxygen: float = Field(..., description="Dissolved Oxygen (mg/L). Acceptable range: 6.0+", example=7.0)
    orthophosphate: float = Field(..., description="Orthophosphate (mg/L). Acceptable range: 0.001–0.1", example=0.05)
    ph: float = Field(..., description="pH. Acceptable range: 6.5–8.5", example=7.2)
    temperature: float = Field(..., description="Temperature (°C). Acceptable range: 0.001–30", example=22.0)
    nitrogen: float = Field(..., description="Nitrogen (mg/L). Acceptable range: 0.001–10.0", example=2.5)
    nitrate: float = Field(..., description="Nitrate (mg/L). Acceptable range: 0.001–10.0", example=5.0)

from pydantic import Field

class PredictionRequest(BaseModel):
    """
    ## Prediction Request
    Submit a list of water sample features to get a prediction. Each parameter includes its acceptable range for realistic results.
    """
    features: List[WaterFeatures]

    class Config:
        schema_extra = {
            "example": {
                "features": [{
                    "ammonia": 0.2, # Acceptable range: 0.001–0.5 mg/L
                    "bod": 3.5, # Acceptable range: 0.001–5.0 mg/L
                    "dissolved_oxygen": 7.0, # Acceptable range: 6.0+ mg/L
                    "orthophosphate": 0.05, # Acceptable range: 0.001–0.1 mg/L
                    "ph": 7.2, # Acceptable range: 6.5–8.5
                    "temperature": 22.0, # Acceptable range: 0.001–30°C
                    "nitrogen": 2.5, # Acceptable range: 0.001–10.0 mg/L
                    "nitrate": 5.0 # Acceptable range: 0.001–10.0 mg/L
                }]
            }
        }

class PredictionResponse(BaseModel):
    """
    ## Prediction Response
    Returns:
    - **predictions**: Water quality score (0=unsafe, 1=safe)
    - **pollutant_probabilities**: Probabilities for each pollutant type
    - **explanation**: SHAP-based feature importances, water quality status, main contributor, and plain-language advice
    """
    predictions: List[float]
    pollutant_probabilities: List[List[float]]
    explanation: dict    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [0.95],
                "pollutant_probabilities": [[0.1, 0.2, 0.3, 0.4]],
                "explanation": {
                    "feature_importance": {
                        "ph": 0.2,
                        "ammonia": 0.1,
                        "bod": 0.3,
                        "nitrate": 0.4
                    },
                    "status": "Safe",
                    "main_contributor": "nitrate",
                    "plain_explanation": "Water quality status: Safe. Main contributor: nitrate (importance: 0.40). Limit agricultural runoff and fertilizer use."
                }
            }
        }