from pydantic import BaseModel
from typing import List

# Defines the structure for a single set of water quality features
class WaterFeatures(BaseModel):
    ph: float
    hardness: float
    solids: float
    chloramines: float
    sulfate: float
    conductivity: float
    organic_carbon: float
    trihalomethanes: float
    turbidity: float

# Defines the structure of the incoming request body
class PredictionRequest(BaseModel):
    features: List[WaterFeatures]

# Defines the structure of the prediction response
class PredictionResponse(BaseModel):
    predictions: List[float]
    pollutant_probabilities: List[List[float]]