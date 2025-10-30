from pydantic import BaseModel
from typing import List

class WaterFeatures(BaseModel):
    ammonia: float
    bod: float
    dissolved_oxygen: float
    orthophosphate: float
    ph: float
    temperature: float
    nitrogen: float
    nitrate: float

class PredictionRequest(BaseModel):
    features: List[WaterFeatures]

    class Config:
        schema_extra = {
            "example": {
                "features": [{
                    "ammonia": 0.2,
                    "bod": 3.5,
                    "dissolved_oxygen": 7.0,
                    "orthophosphate": 0.05,
                    "ph": 7.2,
                    "temperature": 22.0,
                    "nitrogen": 2.5,
                    "nitrate": 5.0
                }]
            }
        }

class PredictionResponse(BaseModel):
    predictions: List[float]
    pollutant_probabilities: List[List[float]]
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [0.95],
                "pollutant_probabilities": [[0.1, 0.2, 0.3, 0.4]]
            }
        }