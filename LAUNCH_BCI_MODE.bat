@echo off
REM ========================================
REM Launch Consciousness App with FREE BCI Data
REM ========================================

echo.
echo ========================================
echo   Consciousness App - BCI Mode
echo   FREE Emotiv Brain Data!
echo ========================================
echo.
echo Using FREE BCI Data:
echo   - Performance Metrics (focus, stress, etc.)
echo   - Mental Commands (if trained)
echo   - Facial Expressions
echo.
echo No license required!
echo.

REM Check if Emotiv software is running
echo Checking for Emotiv Cortex service...
powershell -Command "Test-NetConnection -ComputerName localhost -Port 6868 -InformationLevel Quiet" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Emotiv Cortex not detected on port 6868
    echo Please start EmotivPRO, Emotiv Launcher, or EmotivBCI first!
    echo.
    pause
    exit /b 1
)

echo [OK] Emotiv Cortex service detected
echo.

REM Launch with BCI
echo Starting Consciousness App with BCI data...
echo.

python run.py --eeg-source bci --mode generate --test-rng --debug

echo.
echo ========================================
echo Session ended
echo ========================================
pause
