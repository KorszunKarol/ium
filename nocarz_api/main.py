from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import os
from datetime import datetime
from typing import Dict, Optional

from app.schemas import PredictionRequest, PredictionResponse, HealthResponse
from models.model_loader import ModelLoader
from utils.logging_config import setup_logging
import sqlite3

db_conn = None

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nocarz Airbnb Revenue Prediction API",
    description=
    "API for predicting Airbnb annual revenue using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_loader: Optional[ModelLoader] = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global model_loader

    logger.info("Starting Nocarz API...")

    try:
        models_dir = os.path.join(os.path.dirname(__file__), "..",
                                  "models_deploy")
        print("-" * 100)
        print(models_dir)
        print(os.path.dirname(__file__))
        print("-" * 100)
        model_loader = ModelLoader(models_dir)
        model_loader.load_models()
        logger.info("Model loaded successfully")

        global db_conn
        db_conn = sqlite3.connect("predictions.db")
        db_conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ab_test TEXT,
                neighborhood TEXT,
                predicted_revenue REAL
            )
        """)
        db_conn.commit()

        logger.info("Nocarz API startup completed successfully")

    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Nocarz API...")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Nocarz Airbnb Revenue Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }
    
@app.get("/get_results", response_model=Dict[str, str])
async def get_results():
    return {
        "model A": "Kensington and Chelsea",
        "model B": "City of London"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:

        if model_loader is None:
            raise HTTPException(status_code=503, detail="Models not loaded")

        health_status = model_loader.health_check()

        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            models_loaded=health_status["models_loaded"],
            neural_net_available=health_status["neural_net_available"],
            baseline_available=health_status["baseline_available"])

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503,
                            detail=f"Service unhealthy: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_neural_net(request: PredictionRequest):
    """
  Predict annual revenue using the neural network model specifically
  """
    try:
        if model_loader is None:
            raise HTTPException(status_code=503,
                                detail="Model loader not initialized")

        input_df = request.to_dataframe(model_loader.get_neural_net_features())
        prediction = model_loader.predict_neural_net(input_df)
        db_conn.execute(
            "INSERT INTO predictions (ab_test, neighborhood, predicted_revenue) VALUES (?, ?, ?)",
            (request.ab_test, request.neighbourhood_cleansed, prediction["predicted_revenue"])
        )
        db_conn.commit()
        
        return PredictionResponse(
            predicted_revenue=prediction["predicted_revenue"],
            model_used="neural_net",
            confidence_interval=prediction.get("confidence_interval"),
            prediction_id=prediction["prediction_id"],
            timestamp=datetime.utcnow())

    except ValueError as e:
        logger.warning(f"Invalid input data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Neural net prediction failed: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Prediction failed: {str(e)}")

@app.get("/ab-test/status")
async def get_ab_test_status():
    """Get current A/B tesitng statistics"""
    cursor = db_conn.cursor()
    cursor.execute("""
        SELECT ab_test, COUNT(*) as count, AVG(predicted_revenue) as mean_revenue
        FROM predictions
        GROUP BY ab_test
    """)
    rows = cursor.fetchall()

    return {
        row[0]: {
            "count": row[1],
            "mean_revenue": row[2]
        } for row in rows
    }

if __name__ == "__main__":

    uvicorn.run("main:app",
                host="0.0.0.0",
                port=8000,
                reload=True,
                log_level="info")
