# 🧠 LIVE EEG NOW ENABLED! 

## Status: ✅ CONFIGURED FOR REAL-TIME CONSCIOUSNESS CAPTURE

The system is now configured to use **LIVE Emotiv EEG data** by default!

### 📋 Configuration Summary

**EEG Source:** `cortex` (Emotiv Cortex API)  
**License Key:** `sub_1SOwU0RwlqHDF1GHmzBY6G6e`  
**Client ID:** Configured ✅  
**Client Secret:** Configured ✅  
**Cortex Service:** Running on port 6868 ✅

### 🚀 Quick Launch Commands

#### 1️⃣ Generate Training Data with Live EEG
```bash
python launch_with_live_eeg.py --mode generate --test-rng
```
- Captures your brainwaves while you draw
- Creates training data from your consciousness state
- Uses simulated RNG (or connect TrueRNG device)

#### 2️⃣ Run Inference with Live EEG
```bash
python launch_with_live_eeg.py --mode inference --test-rng
```
- AI responds to your live consciousness state
- Real-time predictions based on your EEG
- Watch the AI interpret your mental state

#### 3️⃣ Launch 369 Oracle with Live EEG
```bash
python launch_with_live_eeg.py --mode oracle
```
- Consciousness oracle interpretation
- Ask questions and express through art
- Three-layer consciousness analysis

### 🔧 Advanced Options

#### Direct Launch (bypass launcher script)
```bash
# Generate mode
python run.py --mode generate --eeg-source cortex --test-rng --debug

# Inference mode
python run.py --mode inference --eeg-source cortex --test-rng --debug

# Oracle mode
python oracle_369_launcher.py
```

#### Use Real TrueRNG Device
```bash
# Remove --test-rng flag to use physical TrueRNG device
python launch_with_live_eeg.py --mode generate
```

#### Change RNG Rate (test mode only)
```bash
python launch_with_live_eeg.py --mode generate --test-rng --rng-rate 16.0
```

### 🧪 Pre-Flight Checklist

Before running with live EEG, ensure:

- [x] Emotiv Pro or EPOC Connect software is **running**
- [x] EEG headset is **connected** and **detected** in Emotiv software
- [x] Headset is **on your head** with good contact quality
- [x] License credentials are **configured** (already done ✅)
- [ ] Cortex service is **running** on port 6868 (check with launcher)

### 📊 What Happens Now

**In Generation Mode:**
1. Your EEG signals are captured in real-time
2. As you draw, your consciousness state is recorded
3. Data is saved for training ML models
4. Your brainwaves correlate with your creative expression

**In Inference Mode:**
1. AI reads your live EEG signals
2. Predicts colors and curves based on your mental state
3. Responds in real-time to consciousness changes
4. Creates art influenced by your thoughts

**In Oracle Mode:**
1. Three consciousness layers analyze your state
2. Mathematical vectors computed from brainwaves
3. ChatGPT interprets consciousness patterns
4. Answers emerge through creative expression

### 🎯 Next Steps

1. **Put on your EEG headset**
2. **Launch Emotiv software** (Pro or EPOC Connect)
3. **Check contact quality** (should be green/good)
4. **Run the launcher** with your preferred mode
5. **Experience consciousness-driven art!**

### 🔍 Troubleshooting

**If you get "Cortex service not detected":**
- Start Emotiv Pro or EPOC Connect
- Wait for headset to connect
- Check port 6868 is not blocked by firewall

**If you get license errors:**
- Credentials are already configured
- Try restarting Emotiv software
- Check Emotiv account is active

**If EEG signals look wrong:**
- Check headset contact quality
- Moisten sensor pads if needed
- Ensure headset is positioned correctly

### 📝 Files Modified

- `config/eeg_config.yaml` - Changed `source: cortex` (live EEG)
- `configure_eeg.py` - Updated with new license key
- `launch_with_live_eeg.py` - NEW launcher script for live sessions

### 🎉 You're Ready!

The consciousness app is now configured for **real-time brainwave capture**. 

Your thoughts, emotions, and consciousness states will directly influence the generative art system!

**Let's make your consciousness visible! 🧠✨**
