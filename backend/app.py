import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from backend.model_handler import ModelHandler
from backend.explanation_engine import explain_prediction


app = FastAPI(
    title="Car Acceptability Predictor API",
    description="API to predict acceptability of a car based on physical features",
    version="1.0.0"
)

# Instantiate the model handler
try:
    model_handler = ModelHandler()
    print("SUCCESS: Machine learning model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load machine learning model: {e}")
    model_handler = None

# Request validation model
class PredictRequest(BaseModel):
    buying: str = Field(..., description="Buying price level (vhigh, high, med, low)")
    maint: str = Field(..., description="Maintenance cost level (vhigh, high, med, low)")
    doors: str = Field(..., description="Number of doors (2, 3, 4, 5more)")
    persons: str = Field(..., description="Capacity in terms of persons (2, 4, more)")
    lug_boot: str = Field(..., description="Size of luggage boot (small, med, big)")
    safety: str = Field(..., description="Estimated safety level (low, med, high)")

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": str(exc)}
    )

@app.post("/api/predict")
async def predict_car_acceptability(request: PredictRequest):
    if not model_handler:
        raise HTTPException(
            status_code=503,
            detail="Model service is currently unavailable. Check server logs."
        )
    
    # Request data as dict
    input_data = request.model_dump()
    
    # Run prediction
    result = model_handler.predict(input_data)
    
    # Generate explanation
    explanation = explain_prediction(
        inputs=input_data,
        prediction_class=result["prediction"],
        contributions=result["contributions"],
        confidence=result["confidence"],
        second_confidence=result["second_confidence"]
    )
    
    return {
        "status": "success",
        "prediction": result["prediction"],
        "prediction_label": result["prediction_label"],
        "confidence": result["confidence"],
        "explanation": explanation,
        "inputs": input_data
    }

@app.get("/api/metadata")
async def get_model_metadata():
    if not model_handler:
        raise HTTPException(
            status_code=503,
            detail="Model service is currently unavailable. Check server logs."
        )
    metadata = model_handler.get_metadata()
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Model metadata not found."
        )
    return {
        "status": "success",
        "metadata": metadata
    }

# Check and mount frontend static directory
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    @app.get("/")
    async def root_fallback():
        return {"message": "API is running. Static frontend directory not found. Please create 'frontend' folder."}
