# Issue: LSTM Model Outputs vs. Expected Inference Format

## The Problem

When you load an LSTM/GRU/Transformer model and run real inference, you see "G" patterns even though:
- ✅ Mock inference is disabled
- ✅ Model has Sigmoid activation
- ✅ Model predictions have good variance (not mode collapse)

## Root Cause

There are **TWO different model architectures** in the codebase:

### 1. PyTorchConsciousnessModel
**Outputs:** `{'colors': [r,g,b], 'positions': [x,y]}` (5 values total)

```python
# Output heads
self.output_heads = nn.ModuleDict({
    'colors': Linear(..., 3),      # RGB
    'positions': Linear(..., 2)    # XY
})
```

### 2. LSTM/GRU/Transformer/CNN-LSTM Models  
**Outputs:** Just **1 single value** (scalar from 0-1)

```python
# Only ONE output
self.fc = nn.Linear(hidden_size, 1)
self.sigmoid = nn.Sigmoid()  # Returns 1 value
```

## The Bug

In `painting_interface.py`, the `process_real_prediction()` function (line 1810) expects:

```python
colors_array = prediction.get('colors', np.array([[0.5, 0.5, 0.5]]))
positions_array = prediction.get('positions', np.array([[0.5, 0.5]]))
```

But LSTM models return:
```python
prediction = 0.274  # Just a single number!
```

The function tries to extract 5 values from 1 value → **Undefined behavior!**

## Secondary Issue: Synthetic Data Patterns

When there's not enough real data, the code generates synthetic EEG/RNG using:

```python
t = time.time()
modulation = np.sin(phase + np.arange(14) * 0.3) * 0.4 + 0.5
```

This creates **time-varying sinusoidal patterns** (similar to the mock inference issue!).

## Solutions

### Option A: Fix Inference Code (Recommended)

Update `process_real_prediction()` to detect model type and handle single-output models:

```python
def process_real_prediction(self, prediction):
    # Check if model has multi-output (PyTorchConsciousnessModel)
    if isinstance(prediction, dict) and 'colors' in prediction:
        # Multi-output model
        colors = prediction['colors']
        positions = prediction['positions']
    else:
        # Single-output model (LSTM/GRU/etc) - need to generate colors/positions
        # Use the single value creatively:
        value = prediction if isinstance(prediction, float) else prediction.item()
        
        # Generate varied colors based on:
        # - Current value
        # - Time (for variation)
        # - Random component (for unpredictability)
        t = time.time()
        r = int((value * 0.5 + np.sin(t * 0.7) * 0.3 + 0.2) * 255)
        g = int((value * 0.3 + np.cos(t * 0.5) * 0.4 + 0.3) * 255)
        b = int((value * 0.7 + np.sin(t * 1.1) * 0.2 + 0.1) * 255)
        
        # Generate position with some randomness
        x = int((value + np.random.rand() * 0.2) * self.canvas.width())
        y = int((1 - value + np.random.rand() * 0.2) * self.canvas.height())
```

### Option B: Retrain with PyTorchConsciousnessModel

Delete the LSTM model and retrain using the PyTorchConsciousnessModel architecture which has proper color/position outputs.

### Option C: Modify LSTM Models to Output 5 Values

Change LSTM architecture to:
```python
self.fc = nn.Linear(hidden_size, 5)  # Output 5 values instead of 1
self.sigmoid = nn.Sigmoid()
```

Then interpret as [r, g, b, x, y].

## Recommendation

**Use Option A** - it's the quickest fix and maintains backward compatibility with existing models.

The single-output models can still be useful - we just need to interpret their single prediction value creatively to generate visual output.

## Status

- ❌ Bug Identified: Mismatched model output formats
- ❌ Current code expects 5 values, LSTM provides 1
- ❌ Synthetic data uses sine/cosine (creates patterns)
- ⏳ Needs fix in `process_real_prediction()`
