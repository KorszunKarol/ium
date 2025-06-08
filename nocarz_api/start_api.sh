#!/bin/bash
cd "$(dirname "$0")" || exit

echo "Starting Nocarz Revenue Prediction API..."

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
