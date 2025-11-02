# ❌ TRAINING DATA MISSING - SOLUTION

## Problem

You're seeing these warnings:
```
⚠️  EEG inputs requested but not available
⚠️  No input sequences available, falling back to synthetic data
```

**Root Cause**: No session files found in `data/` directory!

## Diagnosis

Run this to check your data:
```bash
cd consciousness-app
python check_training_data.py
```

Current status: **0 session files** found.

## Solution: Generate Training Data First

You need to run the app and interact with it to generate data BEFORE training.

### Step 1: Generate Data (Choose One)

#### Option A: RNG-Only Data (Fastest)
```bash
cd consciousness-app
python run.py --test-rng --no-eeg --debug
```

**What to do:**
1. App will open with a canvas
2. Draw something on the canvas (circles, lines, scribbles)
3. Let it run for 2-3 minutes while drawing
4. Close the app (it auto-saves to `data/session_TIMESTAMP.json`)

**Result**: Creates RNG data for RNG-only models

---

#### Option B: RNG + Mock EEG Data (Recommended)
```bash
cd consciousness-app
python run.py --test-rng --test-eeg-mode stable --debug
```

**What to do:**
1. App opens with canvas
2. Draw on the canvas
3. System generates both RNG and simulated EEG data
4. Run for 3-5 minutes
5. Close app

**Result**: Creates both RNG and EEG data for ALL 8 models

---

#### Option C: Real Hardware (Advanced)
```bash
cd consciousness-app
python run.py --test-rng --debug
```

**What to do:**
1. Connect Emotiv EEG headset
2. Draw on canvas
3. Real EEG data is recorded
4. Run for 5+ minutes for good dataset
5. Close app

**Result**: Real RNG + EEG data for production models

---

### Step 2: Verify Data Was Created

```bash
cd consciousness-app
python check_training_data.py
```

You should see:
```
📁 Found 1 session files in data/

📊 Data Summary:
1. session_20251101_XXXXXX.json
   Drawing actions: 1,234
   EEG samples: 5,678  (or 0 if using --no-eeg)
   RNG samples: 9,012
```

---

### Step 3: Train Models

Once you have data files, run training:

```bash
cd consciousness-app
python run.py --mode train --data-dir data
```

**Expected output:**
```
🎮 CUDA GPU detected: NVIDIA GeForce RTX 4090
   Automatically enabling PyTorch GPU training

🚀 Starting Multi-Model Consciousness Training
🌟 Training all default variants

📊 Found 1 session files
📊 Total samples: 1,234 drawing actions, 5,678 EEG, 9,012 RNG

Training variant 1/8: rng_lstm_basic
🎮 GPU ACCELERATION ENABLED!
...
```

---

## Why This Happens

The training pipeline:
1. Looks for `data/session*.json` files ✅
2. If found → Uses real data ✅
3. If NOT found → Falls back to synthetic data ⚠️

You're at step 3 because no real data exists yet.

---

## Quick Start Workflow

```bash
# 1. Generate data (choose one):
cd consciousness-app

# Option A: Quick test (2 min)
python run.py --test-rng --no-eeg --debug
# Draw on canvas for 2 minutes, then close

# Option B: Better test (3-5 min) - RECOMMENDED
python run.py --test-rng --test-eeg-mode stable --debug
# Draw on canvas for 3-5 minutes, then close

# 2. Check data was created:
python check_training_data.py

# 3. Train models:
python run.py --mode train --data-dir data
# OR
python -m src.main --mode train --data-dir data

# 4. Models will be saved to models/ directory
```

---

## Expected Timeline

| Action | Time | Result |
|--------|------|--------|
| Generate data | 2-5 min | 1 session file with ~1000+ samples |
| Check data | 5 sec | Verify data exists |
| Train 8 models | 10-20 min | 8 trained models in `models/` |

With GPU (RTX 4090): **~3 min per model** (24 min total for 8 models)  
Without GPU: **~90 min per model** (12 hours total)

---

## Troubleshooting

### "Still seeing synthetic data warning"
- Check `data/` folder contains `.json` files
- Run `python check_training_data.py` to diagnose
- Make sure you drew on canvas (not just opened and closed app)

### "Not enough samples"
- Need at least 100 drawing actions for training
- Run app longer (5+ minutes)
- Draw more actively on canvas

### "EEG inputs requested but not available"
- Your session files have NO EEG data (ran with `--no-eeg`)
- Solution 1: Train RNG-only models (works fine)
- Solution 2: Generate new data with `--test-eeg-mode stable`

---

## Files Structure

After generating data:
```
consciousness-app/
├── data/
│   ├── session_20251101_120000.json  ← Your training data
│   ├── session_20251101_121500.json  ← More data
│   └── session_20251101_123000.json  ← Even more
├── models/
│   ├── rng_lstm_basic_20251101.pth   ← Trained models (after training)
│   ├── rng_gru_basic_20251101.pth
│   └── ... (8 models total)
└── check_training_data.py            ← Diagnostic tool
```

---

## Summary

✅ **Current State**: No training data exists  
✅ **Solution**: Generate data by running app and drawing  
✅ **Next Steps**:
1. Run app with `--test-rng --test-eeg-mode stable`
2. Draw for 3-5 minutes
3. Close app
4. Run training: `python run.py --mode train --data-dir data`
5. Wait ~24 minutes for 8 GPU-trained models

Then you'll have real models trained on your real data! 🎉
