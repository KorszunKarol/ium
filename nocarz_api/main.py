"""
FastAPI application for Airbnb Annual Revenue Prediction

This API provides endpoints for predicting annual revenue using both:
- Baseline model (neighborhood median)
- Advanced neural network model

With A/B testing capabilities for comparing model performance.

Author: Deployment Team
Created: 2025-05-31
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import random

from app.schemas import PredictionRequest, PredictionResponse, HealthResponse, ABTestResponse
from models.model_loader import ModelLoader
from utils.ab_testing import ABTestManager
from utils.logging_config import setup_logging

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
ab_test_manager: Optional[ABTestManager] = None


@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global model_loader, ab_test_manager

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
        logger.info("Models loaded successfully")

        ab_test_manager = ABTestManager()
        logger.info("A/B testing manager initialized")

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
async def predict_revenue(request: PredictionRequest):
    """
  Predict annual revenue for an Airbnb listing  This endpoint uses A/B testing to serve predictions from either:
  - Baseline model (neighborhood median)
  - Neural network model  The model selection is randomized based on the configured split ratio.
  """
    try:
        if model_loader is None or ab_test_manager is None:
            raise HTTPException(status_code=503,
                                detail="Services not initialized")

        selected_model = ab_test_manager.select_model()

        if selected_model == "neural_net":
            # Use neural net model columns
            input_df = request.to_dataframe(
                model_loader.get_neural_net_features())
            prediction = model_loader.predict_neural_net(input_df)
        else:
            # Use baseline model columns
            input_df = request.to_dataframe(
                model_loader.get_baseline_features())
            prediction = model_loader.predict_baseline(input_df)

        ab_test_manager.log_prediction(
            model_used=selected_model,
            input_data=input_df.to_dict(orient="records")[0],
            prediction=prediction)

        return PredictionResponse(
            predicted_revenue=prediction["predicted_revenue"],
            model_used=selected_model,
            confidence_interval=prediction.get("confidence_interval"),
            prediction_id=prediction["prediction_id"],
            timestamp=datetime.utcnow())

    except ValueError as e:
        logger.warning(f"Invalid input data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Prediction failed: {str(e)}")


@app.post("/predict/neural_net", response_model=PredictionResponse)
async def predict_neural_net(request: PredictionRequest):
    """
  Predict annual revenue using the neural network model specifically  This endpoint bypasses A/B testing and uses only the neural network model.
  """
    try:
        if model_loader is None:
            raise HTTPException(status_code=503,
                                detail="Model loader not initialized")

        input_df = request.to_dataframe(model_loader.get_neural_net_features())
        prediction = model_loader.predict_neural_net(input_df)

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


@app.post("/predict/baseline", response_model=PredictionResponse)
async def predict_baseline(request: PredictionRequest):
    """
  Predict annual revenue using the baseline model specifically  This endpoint bypasses A/B testing and uses only the baseline model.
  """
    try:
        if model_loader is None:
            raise HTTPException(status_code=503,
                                detail="Model loader not initialized")

        input_df = request.to_dataframe(model_loader.get_baseline_features())
        prediction = model_loader.predict_baseline(input_df)

        return PredictionResponse(
            predicted_revenue=prediction["predicted_revenue"],
            model_used="baseline",
            confidence_interval=prediction.get("confidence_interval"),
            prediction_id=prediction["prediction_id"],
            timestamp=datetime.utcnow())

    except ValueError as e:
        logger.warning(f"Invalid input data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Baseline prediction failed: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Prediction failed: {str(e)}")


@app.get("/ab-test/status", response_model=ABTestResponse)
async def get_ab_test_status():
    """Get current A/B testing status and statistics"""
    try:
        if ab_test_manager is None:
            raise HTTPException(status_code=503,
                                detail="A/B test manager not initialized")

        stats = ab_test_manager.get_statistics()

        return ABTestResponse(
            current_split=ab_test_manager.get_current_split(),
            total_predictions=stats["total_predictions"],
            neural_net_count=stats["neural_net_count"],
            baseline_count=stats["baseline_count"],
            neural_net_percentage=stats["neural_net_percentage"],
            baseline_percentage=stats["baseline_percentage"])

    except Exception as e:
        logger.error(f"Failed to get A/B test status: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Failed to get status: {str(e)}")


@app.post("/ab-test/split")
async def update_ab_test_split(neural_net_ratio: float):
    """
  Update the A/B testing split ratio  Args:
      neural_net_ratio: Fraction of requests to route to neural net (0.0 to 1.0)
  """
    try:
        if ab_test_manager is None:
            raise HTTPException(status_code=503,
                                detail="A/B test manager not initialized")

        if not 0.0 <= neural_net_ratio <= 1.0:
            raise HTTPException(status_code=400,
                                detail="Ratio must be between 0.0 and 1.0")

        ab_test_manager.update_split(neural_net_ratio)

        return {
            "message": "A/B test split updated successfully",
            "new_neural_net_ratio": neural_net_ratio,
            "new_baseline_ratio": 1.0 - neural_net_ratio
        }

    except Exception as e:
        logger.error(f"Failed to update A/B test split: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Failed to update split: {str(e)}")


if __name__ == "__main__":

    uvicorn.run("main:app",
                host="0.0.0.0",
                port=8000,
                reload=True,
                log_level="info")
