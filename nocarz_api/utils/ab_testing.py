"""
A/B Testing Manager for model selection

This module handles A/B testing logic for routing requests between
the neural network and baseline models.

Author: Deployment Team
Created: 2025-05-31
"""

import random
import logging
import threading
from typing import Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ABTestManager:
    """Manages A/B testing for model selection"""

    def __init__(self, neural_net_ratio: float = 0.5):
        """
     Initialize A/B testing manager
     Args:
         neural_net_ratio: Fraction of requests to route to neural net (0.0 to 1.0)
     """
        self.neural_net_ratio = neural_net_ratio
        self.baseline_ratio = 1.0 - neural_net_ratio

        self._lock = threading.Lock()
        self._stats = {
            "total_predictions": 0,
            "neural_net_count": 0,
            "baseline_count": 0,
            "start_time": datetime.utcnow()
        }

        logger.info(
            f"A/B testing initialized: {neural_net_ratio:.1%} neural net, {self.baseline_ratio:.1%} baseline"
        )

    def select_model(self) -> str:
        """
     Select which model to use based on A/B testing ratio
     Returns:
         "neural_net" or "baseline"
     """

        if random.random() < self.neural_net_ratio:
            selected = "neural_net"
        else:
            selected = "baseline"

        with self._lock:
            self._stats["total_predictions"] += 1
            if selected == "neural_net":
                self._stats["neural_net_count"] += 1
            else:
                self._stats["baseline_count"] += 1

        logger.debug(f"Selected model: {selected}")
        return selected

    def log_prediction(self, model_used: str, input_data: Dict[str, Any],
                       prediction: Dict[str, Any]):
        """
     Log prediction for monitoring and analysis in a structured JSON format.
     Args:
         model_used: Which model was used ("neural_net" or "baseline")
         input_data: Input data used for prediction
         prediction: Prediction result
     """
        log_entry = {
            "event": "prediction",
            "model_used": model_used,
            "prediction_id": prediction["prediction_id"],
            "predicted_revenue": prediction['predicted_revenue'],
            "input_features": input_data
        }

        logger.info(json.dumps(log_entry))

    def get_statistics(self) -> Dict[str, Any]:
        """Get current A/B testing statistics"""
        with self._lock:
            stats = self._stats.copy()

        total = stats["total_predictions"]
        if total > 0:
            neural_net_pct = (stats["neural_net_count"] / total) * 100
            baseline_pct = (stats["baseline_count"] / total) * 100
        else:
            neural_net_pct = 0.0
            baseline_pct = 0.0

        return {
            "total_predictions":
            total,
            "neural_net_count":
            stats["neural_net_count"],
            "baseline_count":
            stats["baseline_count"],
            "neural_net_percentage":
            neural_net_pct,
            "baseline_percentage":
            baseline_pct,
            "uptime_hours":
            (datetime.utcnow() - stats["start_time"]).total_seconds() / 3600
        }

    def get_current_split(self) -> Dict[str, float]:
        """Get current split configuration"""
        return {
            "neural_net": self.neural_net_ratio,
            "baseline": self.baseline_ratio
        }

    def update_split(self, neural_net_ratio: float):
        """
     Update the A/B testing split ratio
     Args:
         neural_net_ratio: New fraction for neural net (0.0 to 1.0)
     """
        if not 0.0 <= neural_net_ratio <= 1.0:
            raise ValueError("Neural net ratio must be between 0.0 and 1.0")

        old_ratio = self.neural_net_ratio
        self.neural_net_ratio = neural_net_ratio
        self.baseline_ratio = 1.0 - neural_net_ratio

        logger.info(
            f"A/B split updated: {old_ratio:.1%} → {neural_net_ratio:.1%} neural net, "
            f"{1.0 - old_ratio:.1%} → {self.baseline_ratio:.1%} baseline")

    def reset_statistics(self):
        """Reset prediction statistics"""
        with self._lock:
            self._stats = {
                "total_predictions": 0,
                "neural_net_count": 0,
                "baseline_count": 0,
                "start_time": datetime.utcnow()
            }

        logger.info("A/B testing statistics reset")

    def get_model_performance_comparison(self) -> Dict[str, Any]:
        """
     Get performance comparison between models
     Note: This is a placeholder for future implementation where we could
     track prediction accuracy, response times, etc.
     """
        return {
            "neural_net": {
                "predictions": self._stats["neural_net_count"],
                "avg_response_time_ms": None,
                "accuracy_score": None
            },
            "baseline": {
                "predictions": self._stats["baseline_count"],
                "avg_response_time_ms": None,
                "accuracy_score": None
            }
        }
