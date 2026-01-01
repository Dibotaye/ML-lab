from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="IRIS Classification API",
    description="ML API for IRIS flower species classification using Decision Tree and Logistic Regression",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
models = {}
scaler = None

@app.on_event("startup")
async def load_models():
    global models, scaler
    try:
        # Load models from the models directory
        models_dir = "models"  # Models are in the root directory
        
        models["decision_tree"] = joblib.load(os.path.join(models_dir, "decision_tree.joblib"))
        models["logistic"] = joblib.load(os.path.join(models_dir, "logistic_regression.joblib"))
        scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        
        print("✓ Models loaded successfully!")
        print(f"Available models: {list(models.keys())}")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure to run the training script first to generate models!")

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    features: List[float]
    model_type: str = "logistic"  # "logistic" or "decision_tree"

class PredictionResponse(BaseModel):
    prediction: int
    probabilities: List[float]
    predicted_species: str
    model_used: str
    confidence: float

# Species mapping
SPECIES_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}

@app.get("/")
async def root():
    return {
        "message": "IRIS Classification API",
        "version": "1.0.0",
        "available_endpoints": ["/predict", "/health"],
        "models_loaded": len(models) > 0
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "available_models": list(models.keys()) if models else []
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Validate input
        if len(request.features) != 4:
            raise HTTPException(
                status_code=400, 
                detail="Expected 4 features: sepal_length, sepal_width, petal_length, petal_width"
            )
        
        # Validate model type
        if request.model_type not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_type}' not available. Choose from: {list(models.keys())}"
            )
        
        # Prepare features
        features = np.array([request.features])
        
        # Get the model
        model = models[request.model_type]
        
        # Apply scaling for logistic regression
        if request.model_type == "logistic":
            if scaler is None:
                raise HTTPException(status_code=500, detail="Scaler not loaded")
            features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0].tolist()
        
        # Get species name and confidence
        predicted_species = SPECIES_NAMES[prediction]
        confidence = max(probabilities)
        
        return PredictionResponse(
            prediction=int(prediction),
            probabilities=probabilities,
            predicted_species=predicted_species,
            model_used=request.model_type,
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)