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

class ParameterBreakdown(BaseModel):
    parameter: str
    value: float
    limit: str
    reason: str

class Explainability(BaseModel):
    feature_scores: dict
    attention_map: list = Field(default_factory=list, description="Attention weights (if available)")
    integrated_gradients: dict = Field(default_factory=dict, description="Integrated Gradients attribution (if available)")
    plain_explanation: str

class AuditInfo(BaseModel):
    timestamp: str
    model_version: str

class PredictionResponse(BaseModel):
    """
    ## Prediction Response
    Returns:
    - **classification**: Compliant / Non-Compliant with DEAS12:2018
    - **risk_score**: Probability (0–1) of non-compliance
    - **parameter_breakdown**: List of parameters causing non-compliance
    - **forecast**: Predicted water quality status
    - **predictions**: Water quality score (0=unsafe, 1=safe)
    - **pollutant_probabilities**: Probabilities for each pollutant type
    - **explainability**: SHAP/IG/attention/plain explanation
    - **audit**: Timestamp, model version
    """
    classification: str
    risk_score: float
    parameter_breakdown: List[ParameterBreakdown]
    forecast: str
    predictions: List[float]
    pollutant_probabilities: List[List[float]]
    explainability: Explainability
    audit: AuditInfo
    class Config:
        schema_extra = {
            "example": {
                "classification": "Non-Compliant",
                "risk_score": 0.87,
                "parameter_breakdown": [
                    {"parameter": "ph", "value": 5.2, "limit": ">=6.5", "reason": "Below DEAS12:2018 limit"}
                ],
                "forecast": "Unsafe",
                "predictions": [0.95],
                "pollutant_probabilities": [[0.1, 0.2, 0.3, 0.4]],
                "explainability": {
                    "feature_scores": {"ph": 0.45, "nitrate": 0.32},
                    "attention_map": [0.1, 0.2, 0.3, 0.4],
                    "integrated_gradients": {"ph": 0.12, "nitrate": 0.08},
                    "plain_explanation": "Station X flagged as non-compliant because pH = 5.2, below the DEAS limit of 6.5"
                },
                "audit": {
                    "timestamp": "2025-10-31T12:34:56Z",
                    "model_version": "1.0.0"
                }
            }
        }