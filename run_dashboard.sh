#!/bin/bash

# FINN Growth Dashboard Launch Script

echo "🚗 Starting FINN Growth Dashboard..."
echo "====================================="

# Check if we're in the right directory
if [ ! -f "dashboard.py" ]; then
    echo "Error: dashboard.py not found. Please run this script from the growth-dashboard directory."
    exit 1
fi

# Check if required packages are installed
python -c "import streamlit, pandas, numpy, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

echo "Launching dashboard at http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"
echo "====================================="

# Launch Streamlit
streamlit run dashboard.py