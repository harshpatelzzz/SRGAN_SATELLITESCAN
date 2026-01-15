#!/bin/bash
# Startup script for SRGAN system

echo "=========================================="
echo "SRGAN Satellite Imagery Super-Resolution"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install -r api/requirements.txt

# Check for model
if [ ! -f "checkpoints/generator_final.pth" ] && [ ! -f "checkpoints/generator_pretrained_final.pth" ]; then
    echo ""
    echo "⚠️  WARNING: No trained model found!"
    echo "Run training first: python main.py pretrain"
    echo ""
fi

echo ""
echo "Starting services..."
echo ""

# Start API in background
echo "Starting FastAPI backend..."
cd api
python main.py &
API_PID=$!
cd ..

# Wait a moment
sleep 3

# Start frontend
echo "Starting Next.js frontend..."
cd frontend-next
npm install 2>/dev/null
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "=========================================="
echo "Services started!"
echo "=========================================="
echo "API: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for interrupt
trap "kill $API_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
