from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from .predictor import predictor # Import the initialized predictor instance
from .schemas import PredictionRequest, PredictionResponse


# Define the application
app = FastAPI(
    title="Water Quality Monitoring API",
    description="""
    # Water Quality Monitoring API
    ![Logo](/static/logo.png)

    Welcome to the interactive water quality prediction system. Enter your water sample parameters and get:
    - Water quality status (Safe, Moderate, Unsafe)
    - Main contributing pollutant and actionable advice
    - SHAP-based feature importances for transparency

    **How to use:**
    1. Click 'Try it out' on the /predict endpoint.
    2. Enter realistic values (see parameter ranges).
    3. View the prediction, explanation, and recommendations.

    _Powered by Symbolic Graph Attention Network (SymGAT) and SHAP explainability._
    """,
    version="1.0.0",
    swagger_ui_parameters={
        "faviconUrl": "/static/favicon.ico",
        "theme": "dark"
    }
)

# Mount static directory for branding assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS for all origins (for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for branding assets
app.mount("/static", StaticFiles(directory="static"), name="static")


# Health check and info endpoint
@app.get("/", tags=["Health & Info"], summary="API Health Check & Info")
def read_root():
    """
    ## API Health & Info
    Returns API status, model version, and contact info.
    """
    return {
        "status": "Water Quality API is running.",
        "model_version": "1.0.0",
        "last_update": "2025-10-31",
        "contact": "support@waterqualityapi.com"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict_water_quality(request: PredictionRequest):
    try:
        # Acceptable ranges for each feature
        ranges = {
            "ammonia": (0.001, 0.5),
            "bod": (0.001, 5.0),
            "dissolved_oxygen": (6.0, float('inf')),
            "orthophosphate": (0.001, 0.1),
            "ph": (6.5, 8.5),
            "temperature": (0.001, 30.0),
            "nitrogen": (0.001, 10.0),
            "nitrate": (0.001, 10.0)
        }
        # Validate each feature in the request
        for idx, feature in enumerate(request.features):
            for key, (min_val, max_val) in ranges.items():
                value = getattr(feature, key)
                if value < min_val or (max_val != float('inf') and value > max_val):
                    raise HTTPException(
                        status_code=422,
                        detail=f"Feature '{key}' at index {idx} is out of acceptable range ({min_val}–{max_val}). Value: {value}"
                    )
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

        # Map feature importances to names
        feature_names = ["ammonia", "bod", "dissolved_oxygen", "orthophosphate", "ph", "temperature", "nitrogen", "nitrate"]
        importances = explanation.get("feature_importances", [])
        feature_importance_dict = dict(zip(feature_names, importances))

        # Determine water quality status (simple logic: threshold on prediction)
        pred_value = result["predictions"][0]
        if pred_value >= 0.8:
            status = "Safe"
        elif pred_value >= 0.5:
            status = "Moderate"
        else:
            status = "Unsafe"

        # Find main contributing feature
        main_feature = max(feature_importance_dict, key=feature_importance_dict.get)
        main_value = feature_importance_dict[main_feature]

        # Generate plain-language explanation
        advice_map = {
            "ammonia": "Reduce ammonia sources (e.g., sewage, fertilizer runoff).",
            "bod": "Lower organic pollution to reduce BOD.",
            "dissolved_oxygen": "Increase aeration or reduce organic load.",
            "orthophosphate": "Limit phosphate detergents and agricultural runoff.",
            "ph": "Monitor pH and avoid chemical spills.",
            "temperature": "Control thermal pollution from industrial sources.",
            "nitrogen": "Reduce nitrogen fertilizers and waste.",
            "nitrate": "Limit agricultural runoff and fertilizer use."
        }
        advice = advice_map.get(main_feature, "Monitor water quality closely.")
        plain_explanation = (
            f"Water quality status: {status}. "
            f"Main contributor: {main_feature} (importance: {main_value:.2f}). "
            f"{advice}"
        )

        return {
            "predictions": result["predictions"],
            "pollutant_probabilities": result["pollutant_probabilities"],
            "explanation": {
                "feature_importance": feature_importance_dict,
                "status": status,
                "main_contributor": main_feature,
                "plain_explanation": plain_explanation
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# To run the app locally:
# uvicorn app.main:app --reload
