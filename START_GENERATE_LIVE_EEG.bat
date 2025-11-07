@echo off
REM Launch Consciousness App - Generate Mode with Live EEG
echo.
echo ========================================
echo  CONSCIOUSNESS APP - GENERATE MODE
echo  LIVE EEG ENABLED
echo ========================================
echo.
echo Starting data generation with live Emotiv EEG...
echo Make sure your headset is connected!
echo.

python launch_with_live_eeg.py --mode generate --test-rng

pause
