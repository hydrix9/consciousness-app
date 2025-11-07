# Live EEG Mode - Quick Start Guide

## 🧠 Using Real Brain Data

The Consciousness App now supports **LIVE EEG data** from your Emotiv headset!

---

## Prerequisites

### Hardware Setup
1. **Emotiv EEG Headset** - Charged and ready
2. **TrueRNG V3 Device** (optional - can use `--test-rng` for testing)
3. **USB connections** for both devices

### Software Setup
1. **Emotiv Software** must be running (EmotivPRO or Cortex)
2. **API Credentials** configured in `config/app_config.yaml`:
   ```yaml
   hardware:
     emotiv:
       client_id: YOUR_CLIENT_ID
       client_secret: YOUR_CLIENT_SECRET
       license: YOUR_LICENSE_KEY (if using licensed features)
   ```

---

## Quick Launch Commands

### 1. **Generate Mode** (Data Collection with Real Brain Data)
```powershell
# Full live mode (real EEG + real RNG)
python run_live.py

# Real EEG + mock RNG (for testing)
python run_live.py --test-rng
```

**What it does:**
- ✅ Connects to Emotiv EEG headset
- ✅ Streams live brainwave data at 128 Hz
- ✅ Captures brain state during drawing
- ✅ Records consciousness-guided creativity
- ✅ Saves sessions for ML training

### 2. **Inference Mode** (Model Predictions Guided by Brain)
```powershell
# Full live inference mode
python run_live.py --mode inference

# Real EEG + mock RNG inference
python run_live.py --mode inference --test-rng
```

**What it does:**
- ✅ Loads trained ML models
- ✅ Uses LIVE brain data as input
- ✅ Generates brain-guided predictions
- ✅ Visualizes consciousness-driven art
- ✅ Real-time brain-to-canvas feedback

---

## Usage Workflow

### Data Collection Session (Generate Mode)

1. **Put on EEG headset** and ensure good contact
2. **Start Emotiv software** (EmotivPRO/Cortex)
3. **Launch app:**
   ```powershell
   cd "d:\MEGA\Projects\Consciousness\consciousness-app"
   python run_live.py
   ```
4. **Wait for connection** (you'll see "EEG Connected" in status)
5. **Click "Start Session"** in the GUI
6. **Draw naturally** for 5-10 minutes:
   - Vary your strokes
   - Change colors
   - Use different layers
   - Navigate pocket dimensions
   - Let your consciousness guide you!
7. **Click "Stop Session"** when done
8. **Session saved** with real brain data!

### Training with Real Brain Data

```powershell
# Train models on your live EEG sessions
python run.py --mode train --data-dir data
```

Models will learn correlations between:
- Your brainwave patterns
- Drawing actions
- Color choices
- Spatial movements
- Creative states

### Inference with Live Brain Control

1. **Train a model** on your real EEG data first
2. **Put on headset** and connect
3. **Launch inference mode:**
   ```powershell
   python run_live.py --mode inference
   ```
4. **Load your trained model** in the GUI
5. **Watch your brain control the canvas!**
   - Predictions use LIVE brain data
   - Different mental states → different patterns
   - Real-time consciousness visualization

---

## Comparison: Mock vs Live

| Feature | Mock Mode (`--no-eeg`) | Live Mode (`run_live.py`) |
|---------|------------------------|---------------------------|
| EEG Data | Simulated sine waves | Real brainwaves (128 Hz) |
| Brain Input | Static patterns | Dynamic mental states |
| Training Quality | Generic patterns | Personal brain signatures |
| Inference Accuracy | Random-ish | Brain-state driven |
| Connection Required | None | Emotiv headset + software |
| Best For | Testing/development | Real consciousness research |

---

## Troubleshooting

### "Failed to connect to EEG"

**Check:**
1. Is Emotiv software running?
2. Is headset powered on?
3. Are credentials in `config/app_config.yaml` correct?
4. Is headset paired with computer?

**Solution:**
- Start EmotivPRO/Cortex first
- Verify headset connection in Emotiv software
- Check `client_id` and `client_secret` in config

### "EEG data quality poor"

**Check:**
1. Electrode contact (wet sensors with saline)
2. Headset positioning (secure fit)
3. Battery level (charge headset)

**Solution:**
- Adjust headset for better contact
- Apply more saline solution
- Check signal quality in Emotiv software

### "No brain data in saved session"

**Check:**
1. Was EEG actually connected? (check console logs)
2. Did you use `--no-eeg` flag by accident?
3. Was session stopped prematurely?

**Solution:**
- Verify "EEG: Connected" in status bar
- Check console for "EEG bridge initialized"
- Use `run_live.py` to ensure EEG enabled

---

## Advanced Usage

### Custom EEG Sampling Rate
Edit `config/app_config.yaml`:
```yaml
timing:
  eeg_sampling_rate: 128  # Hz (128 or 256 typical)
```

### Force Specific EEG Backend
```powershell
# Force Cortex API (Emotiv cloud)
python run.py --mode generate --eeg-source cortex

# Force local simulation (for testing)
python run.py --mode generate --eeg-source simulated
```

### Combine with Other Options
```powershell
# Live EEG + mock RNG + debug logging
python run_live.py --test-rng --debug

# Inference with streaming server
python run_live.py --mode inference --enable-streaming --stream-port 8765
```

---

## What Makes Live EEG Special?

### During Data Collection
- **Brain states captured** during different creative actions
- **Mental focus** correlates with drawing precision
- **Emotional states** reflected in color choices
- **Consciousness levels** embedded in the data

### During Inference
- **Brain-guided predictions** instead of random
- **Mental state** influences output patterns
- **Real-time feedback loop** between mind and canvas
- **Consciousness visualized** through ML interpretations

### For Training
- **Personalized models** learn YOUR brain patterns
- **Mental signatures** create unique artistic styles
- **Brain-art correlations** enable consciousness modeling
- **Real science** meets creative expression!

---

## Next Steps

1. ✅ **Set up headset** and configure credentials
2. ✅ **Run live session**: `python run_live.py`
3. ✅ **Collect 10+ minutes** of varied drawing data
4. ✅ **Train model**: `python run.py --mode train --data-dir data`
5. ✅ **Test inference**: `python run_live.py --mode inference`
6. ✅ **Experience** brain-controlled art generation!

---

## Default vs Live Modes

| Command | EEG Source | Use Case |
|---------|-----------|----------|
| `python run.py` | Mock (no real EEG) | Testing, development |
| `python run.py --no-eeg` | Disabled | RNG-only experiments |
| `python run_live.py` | **Real Emotiv** | **Consciousness research** |
| `python run_live.py --mode inference` | **Real Emotiv** | **Brain-controlled art** |

---

**Ready to make it LIVE?** 🧠✨

```powershell
python run_live.py
```

Let your consciousness drive the canvas! 🎨🌌
