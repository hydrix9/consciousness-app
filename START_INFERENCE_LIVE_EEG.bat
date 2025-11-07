@echo off
REM Launch Consciousness App - Inference Mode with Live EEG
echo.
echo ========================================
echo  CONSCIOUSNESS APP - INFERENCE MODE
echo  LIVE EEG ENABLED
echo ========================================
echo.
echo Starting AI inference with live Emotiv EEG...
echo Make sure your headset is connected!
echo.

python launch_with_live_eeg.py --mode inference --test-rng

pause
