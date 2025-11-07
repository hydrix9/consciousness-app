# 🎉 BCI MODE - FREE EMOTIV BRAIN DATA

**The Consciousness App now uses FREE Emotiv BCI data instead of raw EEG!**

---

## 🆓 What This Means

- ✅ **NO LICENSE REQUIRED!** (was $99/month)
- ✅ **FREE brain data** from your Emotiv headset
- ✅ **23 virtual EEG channels** (vs 14 raw channels)
- ✅ **Works with all app modes** (generate, inference, oracle)
- ✅ **Better for consciousness app** (meaningful brain metrics)

---

## 🧠 Available Data

### Performance Metrics (FREE!)
- **Focus** - Mental focus/attention level
- **Stress** - Stress/tension
- **Engagement** - How engaged you are
- **Excitement** - Arousal/excitement level
- **Interest** - Interest/curiosity
- **Relaxation** - Calm/relaxed state

### Mental Commands (if trained)
- Push, Pull, Lift, Drop
- Left, Right, Rotate Left, Rotate Right

### Facial Expressions (automatic)
- Smile, Clench, Smirk, Blink, Wink, Surprise, Frown

---

## 🚀 Quick Start

### 1. Make sure Emotiv software is running
- EmotivPRO, Emotiv Launcher, or EmotivBCI
- Headset connected and working

### 2. Run the app (now uses BCI automatically!)

```powershell
python run.py --mode generate --test-rng
```

Or use the launcher:

```powershell
.\LAUNCH_BCI_MODE.bat
```

**That's it!** The app will use FREE BCI data! 🎉

---

## 📖 Documentation

- **[BCI_MODE_ENABLED.md](BCI_MODE_ENABLED.md)** - Complete BCI guide
- **[BCI_MODE_SWITCH_SUMMARY.md](BCI_MODE_SWITCH_SUMMARY.md)** - Technical summary

---

## ⚙️ Configuration

Already configured! But if you want to check:

`config/eeg_config.yaml`:
```yaml
eeg:
  source: bci  # FREE - no license required!
```

`config/app_config.yaml`:
```yaml
hardware:
  emotiv:
    client_id: <your_id>
    client_secret: <your_secret>
    license: ''  # Not needed for BCI mode!
```

---

## 🎮 How It Works

```
Your Brain
    ↓
Emotiv Headset
    ↓
Emotiv Software
    ↓
FREE BCI Data (Performance Metrics, Mental Commands, Facial Expressions)
    ↓
Virtual EEG Channels (PM_FOCUS, PM_STRESS, etc.)
    ↓
ML Models (LSTM/Transformer)
    ↓
Consciousness Predictions
```

---

## 🔍 Verify BCI Mode

Look for these messages when running with `--debug`:

```
✅ Emotiv BCI Source initialized (FREE - no license required)
✅ Authenticated with Cortex (BCI mode - no license)
✅ Subscribed to met stream (Performance Metrics)
✅ Subscribed to com stream (Mental Commands)
✅ Subscribed to fac stream (Facial Expressions)
✅ BCI streaming started
```

---

## 💡 Why BCI is Better for This App

| Feature | Raw EEG | BCI Metrics |
|---------|---------|------------|
| Cost | $99/month | **FREE** ✅ |
| Meaning | Electrical signals | Brain states ✅ |
| Noise | Higher | Lower ✅ |
| Setup | Complex | Easy ✅ |
| For Consciousness | Overkill | Perfect ✅ |

BCI metrics (focus, stress, engagement) are actually **more meaningful** for consciousness exploration than raw electrical signals!

---

## 🐛 Troubleshooting

### "Authentication failed"
- Check `client_id` and `client_secret` in `config/app_config.yaml`

### "No headset found"
- Connect Emotiv headset
- Start Emotiv software first
- Verify headset works in Emotiv software

### Falls back to Mock
- Check Emotiv software is running (port 6868)
- Look at debug logs: `python run.py --eeg-source bci --debug`

---

## ✨ Features

### Generate Mode
- Your **focus** affects drawing precision
- Your **stress** influences colors
- Your **engagement** modulates patterns
- Mental states create unique art!

### Inference Mode
- BCI metrics feed into neural networks
- Predictions based on your brain state
- Virtual channels work like raw EEG
- Same ML architecture, FREE data!

---

## 🎓 Resources

- Full documentation: [BCI_MODE_ENABLED.md](BCI_MODE_ENABLED.md)
- Technical summary: [BCI_MODE_SWITCH_SUMMARY.md](BCI_MODE_SWITCH_SUMMARY.md)
- Emotiv Cortex API: https://emotiv.gitbook.io/cortex-api/

---

**Your brain controls the art, and now it's FREE!** 🧠✨🎨
