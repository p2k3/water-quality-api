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
        # DEAS21:2018 compliance ranges (for reporting, not blocking)
        compliance_ranges = {
            "ammonia": (0.001, 0.5),
            "bod": (0.001, 5.0),
            "dissolved_oxygen": (6.0, float('inf')),
            "orthophosphate": (0.001, 0.1),
            "ph": (6.5, 8.5),
            "temperature": (0.001, 30.0),
            "nitrogen": (0.001, 10.0),
            "nitrate": (0.001, 10.0)
        }
        
        # Physical/technical validation limits (reject only impossible values)
        # These are much wider to allow testing of contaminated/unsafe samples
        technical_limits = {
            "ammonia": (0.0, 50.0),        # Up to 50 mg/L (severe pollution)
            "bod": (0.0, 50.0),             # Up to 50 mg/L (raw sewage levels)
            "dissolved_oxygen": (0.0, 20.0), # 0-20 mg/L (physically possible)
            "orthophosphate": (0.0, 10.0),  # Up to 10 mg/L (extreme eutrophication)
            "ph": (0.0, 14.0),              # 0-14 (full pH scale)
            "temperature": (0.0, 50.0),     # 0-50°C (physically realistic for water bodies)
            "nitrogen": (0.0, 100.0),       # Up to 100 mg/L (industrial discharge)
            "nitrate": (0.0, 200.0)         # Up to 200 mg/L (agricultural pollution)
        }
        
        # Technical validation only (reject physically impossible values)
        for idx, feature in enumerate(request.features):
            for key, (min_val, max_val) in technical_limits.items():
                value = getattr(feature, key)
                if value < min_val or value > max_val:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Feature '{key}' at index {idx} is physically impossible. Value: {value}. Valid range: {min_val}–{max_val}"
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

        # Compliance assessment (DEAS21:2018) - for reporting only
        parameter_breakdown = []
        compliant = True
        critical_violations = []
        
        for key, (min_val, max_val) in compliance_ranges.items():
            value = getattr(request.features[0], key)
            if value < min_val or (max_val != float('inf') and value > max_val):
                compliant = False
                reason = f"{'Below' if value < min_val else 'Above'} DEAS21:2018 limit"
                limit_str = f">={min_val}" if value < min_val else f"<={max_val}"
                
                # Calculate deviation percentage
                if value < min_val:
                    deviation_percent = ((min_val - value) / min_val) * 100
                else:
                    deviation_percent = ((value - max_val) / max_val) * 100 if max_val != float('inf') else 0
                
                # Determine severity
                severity = "CRITICAL" if deviation_percent > 20 else "MODERATE"
                if severity == "CRITICAL":
                    critical_violations.append(key)
                
                parameter_breakdown.append({
                    "parameter": key,
                    "value": value,
                    "limit": limit_str,
                    "reason": reason,
                    "severity": severity,
                    "deviation_percent": deviation_percent
                })

        # ========== UNIFIED CLASSIFICATION LOGIC ==========
        # Combines DEAS compliance with model prediction for final forecast
        # Priority: DEAS Non-Compliance overrides model prediction
        
        prediction_score = result["predictions"][0]
        base_risk = 1.0 - prediction_score
        
        # Calculate risk score with violation penalty
        if not compliant:
            violation_penalty = len(critical_violations) * 0.15
            risk_score = min(1.0, base_risk + violation_penalty)
        else:
            risk_score = base_risk
        
        # ========== UNIFIED FORECAST LOGIC ==========
        # Non-compliant water CANNOT be "Safe"
        if not compliant:
            # Non-compliant with critical violations = Unsafe
            if len(critical_violations) > 0 or risk_score > 0.7:
                forecast = "Unsafe"
                confidence = "High"
                classification = "Non-Compliant"
                card_color = "danger"  # Red/Pink for frontend
            else:
                # Non-compliant but moderate violations = Moderate
                forecast = "Moderate"
                confidence = "Medium"
                classification = "Non-Compliant"
                card_color = "warning"  # Orange/Yellow
        else:
            # Compliant - use model prediction
            if prediction_score >= 0.7:
                forecast = "Safe"
                confidence = "High"
                classification = "Compliant"
                card_color = "success"  # Green
            elif prediction_score >= 0.4:
                forecast = "Moderate"
                confidence = "Medium"
                classification = "Compliant"
                card_color = "warning"  # Orange/Yellow
            else:
                # Compliant but model detects hidden risk
                forecast = "Unsafe"
                confidence = "High"
                classification = "Compliant (Model Detects Risk)"
                card_color = "danger"  # Red/Pink

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
            severity_note = f" ({len(critical_violations)} critical violation(s))" if critical_violations else ""
            plain_explanation = (
                f"{classification}{severity_note}: Water sample violates DEAS 21:2018 standards - {reasons}. "
                f"Forecast: {forecast}. Top contributing factor: '{main_feature}' (SHAP importance: {main_value:.3f}). "
                f"Risk Score: {risk_score:.1%}. Recommendation: {advice}"
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
                f"{classification}: All parameters within DEAS 21:2018 limits. "
                f"Model quality score: {sigmoid_score:.3f} (raw: {raw_score:.3f}). "
                f"Forecast: {forecast}. Top signal: {main_feature} (SHAP: {main_value:.3f}); "
                f"Likely pollutant type: {top_pollutant}. Recommendation: {advice}"
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
            "forecast": forecast,
            "card_color": card_color,
            "confidence": confidence,
            "risk_score": risk_score,
            "parameter_breakdown": parameter_breakdown,
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
