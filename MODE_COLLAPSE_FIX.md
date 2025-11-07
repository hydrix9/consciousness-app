# Fix for Mode Collapse: Collect More Training Data

## The Problem

Your LSTM model has **mode collapse** - it outputs the same value (0.000011) for every input!

**Root Cause:** Only 8 training samples - WAY too few for a neural network.

**Evidence:**
```
Test 1: Prediction: 0.000011
Test 2: Prediction: 0.000011
Test 3: Prediction: 0.000011
...
Test 10: Prediction: 0.000011

Std Dev: 0.000000 (no variance at all!)
```

## The Solution: Collect More Data

### Step 1: Delete the Bad Model

```powershell
cd "d:\MEGA\Projects\Consciousness\consciousness-app"

# Delete the collapsed model
Remove-Item -Recurse -Force models\eeg_lstm_basic_20251101_212012

# Edit model_registry.json - remove the eeg_lstm_basic entry
# (or just delete the whole file, it will be recreated)
Remove-Item models\model_registry.json
```

### Step 2: Collect Training Data (100+ samples minimum)

```powershell
# Start the app with test hardware
python run.py --test-rng --test-eeg-mode stable --debug

# In the GUI:
# 1. Click "Start Session"
# 2. Draw varied patterns for 5-10 minutes:
#    - Different colors
#    - Different positions
#    - Different brush sizes
#    - Change layers (1, 2, 3)
#    - Navigate pocket dimensions
#    - Draw curves, lines, circles, etc.
# 3. Click "Stop Session"
# 4. Note the session file created in data/

# Repeat 2-3 more times to get multiple sessions
```

### Step 3: Train with More Data

```powershell
# Option A: Use the GUI Training tab
python run.py
# Go to Training tab
# Select your data sessions
# Click Train
# Wait for completion (should take longer with more data!)

# Option B: Command line training
python -m src.ml.training_pipeline --data-dir data --architecture lstm
```

### Step 4: Verify the New Model

```powershell
# Test the new model
python test_model_predictions.py

# Expected output:
# ✅ GOOD VARIANCE
#    Predictions are varied
#    Std Dev: > 0.1  (shows variety!)
```

### Step 5: Test in GUI

```powershell
python run.py

# In GUI:
# 1. Go to "🧠 Inference Mode" tab
# 2. Select: eeg_lstm_basic (new model)
# 3. Click: "Load Model & Start Inference"
# 4. Watch for VARIED diamond markers across the canvas
# 5. Colors should change
# 6. Positions should vary
# 7. NO repetitive "G" patterns!
```

## Why 8 Samples Isn't Enough

Neural networks need diversity to learn patterns:

- **8 samples:** Model can't learn → outputs same value for everything (mode collapse)
- **100+ samples:** Model learns variety → outputs different values
- **1000+ samples:** Even better generalization

## What You Should See After Retraining

**Before (8 samples, mode collapse):**
```
Prediction 1: 0.000011
Prediction 2: 0.000011
Prediction 3: 0.000011
→ Draws same "G" pattern repeatedly
```

**After (100+ samples, good training):**
```
Prediction 1: 0.234
Prediction 2: 0.678
Prediction 3: 0.123
→ Draws varied positions/colors across canvas
```

## Key Points

1. ✅ Architecture is correct (Sigmoid is there)
2. ✅ Code can load LSTM models now
3. ✅ Visualization handles single-output models
4. ❌ **Current model has mode collapse from too little data**
5. ✅ **Solution: Collect 100+ training samples and retrain**

---

**Bottom line:** The "G" patterns are from mode collapse due to insufficient training data. Collect more data (5-10 minutes of varied drawing) and retrain!
