from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Water Quality Prediction API",
    description="""
    # 💧 Water Quality Prediction API
    
    This professional API uses a **Symbolic Graph Attention Network (SymGAT)** to predict water quality parameters.
    
    ## 📊 Features Required
    | Parameter | Description | Unit |
    |-----------|------------|------|
    | pH | Acidity/basicity level | 0-14 scale |
    | Hardness | Mineral content | mg/L |
    | Solids | Total dissolved solids | ppm |
    | Chloramines | Disinfectant level | ppm |
    | Sulfate | Sulfate concentration | mg/L |
    | Conductivity | Electrical conductivity | μS/cm |
    | Organic Carbon | Organic matter content | ppm |
    | Trihalomethanes | Disinfection byproducts | μg/L |
    | Turbidity | Water clarity | NTU |
    
    ## 🚀 Example Request
    ```json
    {
      "features": [
        {
          "ph": 7.0,
          "hardness": 150.0,
          "solids": 20000.0,
          "chloramines": 7.0,
          "sulfate": 350.0,
          "conductivity": 400.0,
          "organic_carbon": 15.0,
          "trihalomethanes": 80.0,
          "turbidity": 4.0
        }
      ]
    }
    ```
    
    ## 📈 Response Format
    The API returns detailed predictions and pollutant probabilities for each sample.
    """,
    version="1.0.0",
    contact={
        "name": "Water Quality Team",
        "url": "https://github.com/p2k3/water-quality-api"
    },
    docs_url=None,  # Disable default docs
    redoc_url="/redoc"
)

# Custom CSS for Swagger UI
swagger_ui_css = """
<style>
    body {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .swagger-ui {
        background-color: white;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        border-radius: 8px;
        padding: 20px;
    }
    .swagger-ui .topbar {
        background-color: #2c3e50;
        padding: 15px;
    }
    .swagger-ui .info .title {
        color: #2c3e50;
        font-size: 36px;
    }
    .swagger-ui .opblock.opblock-post {
        background: rgba(97,175,254,.1);
        border-color: #61affe;
    }
    .swagger-ui .btn.execute {
        background-color: #2c3e50;
        border-color: #2c3e50;
    }
    .swagger-ui .btn.execute:hover {
        background-color: #34495e;
    }
    .swagger-ui table tbody tr td {
        padding: 10px;
    }
    .swagger-ui .markdown p, .swagger-ui .markdown li {
        font-size: 16px;
    }
    .swagger-ui .scheme-container {
        background-color: white;
        box-shadow: none;
    }
</style>
"""

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        extra_html=swagger_ui_css,
    )