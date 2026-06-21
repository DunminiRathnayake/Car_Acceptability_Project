@echo off
echo ===================================================
echo Starting Car Acceptability Predictor Web Server...
echo ===================================================
echo.
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Failed to start uvicorn. Please ensure Python dependencies are installed.
    echo Run: pip install -r backend/requirements.txt
    echo.
    pause
)
