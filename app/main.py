from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import os
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

# Mount static directory for branding assets (only if present)
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
else:
    # In some deployment modes (Docker or CI) the static folder might be missing.
    # We avoid failing startup and log a warning instead.
    import logging
    logging.getLogger("uvicorn.error").warning("Static directory 'static' not found; skipping StaticFiles mount.")

# Configure CORS. Prefer explicit FRONTEND_ORIGINS in production (comma-separated),
# otherwise fall back to allow-all for development convenience.
frontend_origins_env = os.getenv("FRONTEND_ORIGINS") or os.getenv("VITE_API_URL")
if frontend_origins_env:
    # If user passed a single VITE_API_URL, allow that origin. If comma-separated list, split.
    if "," in frontend_origins_env:
        allowed_origins = [o.strip() for o in frontend_origins_env.split(',') if o.strip()]
    else:
        # If a full URL was provided (e.g. https://my-app.vercel.app), use it as-is.
        allowed_origins = [frontend_origins_env.strip()]
else:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Note: StaticFiles mount already handled above only if the directory exists.


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

        # Compliance logic (DEAS12:2018)
        parameter_breakdown = []
        compliant = True
        for key, (min_val, max_val) in ranges.items():
            value = getattr(request.features[0], key)
            if value < min_val or (max_val != float('inf') and value > max_val):
                compliant = False
                reason = f"{'Below' if value < min_val else 'Above'} DEAS12:2018 limit"
                limit_str = f">={min_val}" if value < min_val else f"<={max_val}"
                parameter_breakdown.append({
                    "parameter": key,
                    "value": value,
                    "limit": limit_str,
                    "reason": reason
                })

        classification = "Compliant" if compliant else "Non-Compliant"
        risk_score = 1.0 - result["predictions"][0] if not compliant else 0.0
        forecast = "Safe" if result["predictions"][0] >= 0.8 else ("Moderate" if result["predictions"][0] >= 0.5 else "Unsafe")

        # Find main contributing feature (from SHAP importances)
        # If importances are all tiny/zero, fallback to pollutant probabilities for the 'top' signal
        main_feature = None
        main_value = 0.0
        if importances and any([abs(v) > 1e-6 for v in importances]):
            main_feature = max(feature_importance_dict, key=feature_importance_dict.get)
            main_value = feature_importance_dict[main_feature]
        else:
            # fallback: choose pollutant with highest probability
            if result.get('pollutant_probabilities') and len(result['pollutant_probabilities'])>0:
                probs = result['pollutant_probabilities'][0]
                idx = int(np.argmax(probs))
                pf_map = ['bacterial','chemical','organic','agricultural']
                main_feature = pf_map[idx]
                main_value = probs[idx]

        # Generate plain-language explanation — more informative
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
        # Build explanation text
        if not compliant:
            reasons = "; ".join([f"{p['parameter']} {p['reason']}" for p in parameter_breakdown])
            advice = advice_map.get(main_feature, "Monitor water quality closely.")
            plain_explanation = (
                f"{classification}: Station flagged as {classification.lower()} because {reasons}. Top contributing factor appears to be '{main_feature}' (importance {main_value:.3f}). {advice}"
            )
        else:
            # compliant: provide auditable model context (raw and sigmoid scores, top feature/pollutant)
            raw_score = result.get('raw_predictions', [None])[0]
            sigmoid_score = result.get('predictions', [None])[0]
            top_pollutant_idx = int(np.argmax(result.get('pollutant_probabilities',[[]])[0])) if result.get('pollutant_probabilities') else None
            pollutant_labels = ['Bacterial','Chemical','Organic','Agricultural']
            top_pollutant = pollutant_labels[top_pollutant_idx] if top_pollutant_idx is not None else 'N/A'
            advice = advice_map.get(main_feature, "Continue routine monitoring and maintain current controls.")
            plain_explanation = (
                f"{classification}: All parameters are within limits. Model raw score={raw_score:.3f}, scaled score={sigmoid_score:.3f}. Top signal: {main_feature} (importance {main_value:.3f}); likely pollutant: {top_pollutant}. {advice}"
            )

        # Explainability (SHAP, IG, attention map placeholders)
        explainability = {
            "feature_scores": feature_importance_dict,
            "attention_map": explanation.get("attention_map", []),
            "integrated_gradients": explanation.get("integrated_gradients", {}),
            "plain_explanation": plain_explanation,
            "main_feature": main_feature,
            "main_feature_value": main_value
        }

        # Audit info
        from datetime import datetime
        audit = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_version": "1.0.0"
        }

        return {
            "classification": classification,
            "risk_score": risk_score,
            "parameter_breakdown": parameter_breakdown,
            "forecast": forecast,
            "predictions": result["predictions"],
            "raw_predictions": result.get("raw_predictions", []),
            "pollutant_probabilities": result["pollutant_probabilities"],
            "explainability": explainability,
            "audit": audit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", tags=["Health & Info"], summary="Service health check")
def health_check():
    """Returns service readiness and model status in a user-friendly format."""
    try:
        model_ready = getattr(predictor, 'ready', False)
        if model_ready:
            return {
                "status": "ok",
                "message": "Backend is ready and the model is loaded.",
                "model_version": "1.0.0",
                "ready": True
            }
        else:
            return {
                "status": "starting",
                "message": "Backend is running but the model is still loading. Try again in a moment.",
                "model_version": "1.0.0",
                "ready": False
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "ready": False
        }
# To run the app locally:
# uvicorn app.main:app --reload
