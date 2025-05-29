#!/bin/bash
# Runner script for generating professional visualizations

echo "=== Airbnb Data Professional Visualization Generator ==="
echo "This script will generate all visualizations and save them as high-quality images."
echo ""

# Get the script directory and change to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if we're in the right directory
if [ ! -d "data/processed/etap2" ]; then
    echo "Error: data/processed/etap2 directory not found."
    echo "Current directory: $(pwd)"
    echo "Please ensure you're running this script from the project root directory."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p reports/figures/etap2

# Run the Python script
echo "Starting visualization generation..."
python scripts/generate_professional_visualizations.py

echo ""
echo "=== Visualization Generation Complete ==="
echo "Check the reports/figures/etap2/ directory for generated plots."
echo ""
echo "Generated files include:"
echo "- PNG files for static plots (high resolution)"
echo "- HTML files for interactive plots (Plotly)"
echo ""
