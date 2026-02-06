@echo off
REM Quick Start Script for Early Fusion Model

echo.
echo ================================================
echo  Early Fusion SPEI6 Prediction System
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
pip install -q -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed

echo.
echo [2/3] Training the model...
echo This may take a few minutes...
python train_early_fusion.py
if %errorlevel% neq 0 (
    echo Error: Model training failed
    pause
    exit /b 1
)
echo ✓ Model trained successfully

echo.
echo [3/3] Launching prediction UI...
echo.
echo Opening Streamlit app at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
timeout /t 2 /nobreak
streamlit run app_predict.py

pause
