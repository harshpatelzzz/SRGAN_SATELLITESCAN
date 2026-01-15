@echo off
REM Startup script for SRGAN system (Windows)

echo ==========================================
echo SRGAN Satellite Imagery Super-Resolution
echo ==========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
pip install -r api\requirements.txt

REM Check for model
if not exist "checkpoints\generator_final.pth" if not exist "checkpoints\generator_pretrained_final.pth" (
    echo.
    echo WARNING: No trained model found!
    echo Run training first: python main.py pretrain
    echo.
)

echo.
echo Starting services...
echo.

REM Start API in background
echo Starting FastAPI backend...
start "SRGAN API" cmd /k "cd api && python main.py"

REM Wait a moment
timeout /t 3 /nobreak >nul

REM Start frontend
echo Starting Next.js frontend...
cd frontend-next
call npm install 2>nul
start "SRGAN Frontend" cmd /k "npm run dev"
cd ..

echo.
echo ==========================================
echo Services started!
echo ==========================================
echo API: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Close the windows to stop services
echo.

pause
