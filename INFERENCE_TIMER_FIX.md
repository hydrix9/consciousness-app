# Inference Timer Conflict Fix

## Problem

1. **Inference predictions appearing even when not enabled**
2. **Still seeing repeated patterns (same predictions every time)**

## Root Cause

There were **TWO different inference systems** using the **SAME timer**, causing conflicts:

### 1. Real Inference Mode (Lines 1470-1630)
- Used for loading trained models from the Inference Mode tab
- Controlled by `self.inference_active` flag
- Button: "Load Model & Start Inference"
- Uses: `self.inference_timer`

### 2. Merged Mode Inference (Lines 2445-2550)
- Used for recursive recording with mock predictions
- Controlled by `self.inference_mode_enabled` flag  
- Checkbox: "Enable Inference Mode"
- **ALSO USED: `self.inference_timer`** ← CONFLICT!

### The Conflict

Both systems were trying to use `self.inference_timer`, so:
- When you enabled merged mode, it would start the timer
- When you disabled merged mode, real inference could still be using it
- Timers would overwrite each other's callbacks
- Predictions would run even when "disabled"

## The Fix

### Separated the Timers

**Real Inference Mode:**
```python
# Now uses self.real_inference_timer
if not hasattr(self, 'real_inference_timer'):
    self.real_inference_timer = QTimer()
    self.real_inference_timer.timeout.connect(self.run_real_inference_cycle)
```

**Merged Mode Inference:**
```python
# Now uses self.merged_mode_inference_timer
if not hasattr(self, 'merged_mode_inference_timer'):
    self.merged_mode_inference_timer = QTimer()
    self.merged_mode_inference_timer.timeout.connect(self.run_merged_mode_inference_cycle)
```

### Added Proper Enable/Disable Checks

**Merged Mode Cycle:**
```python
def run_merged_mode_inference_cycle(self):
    """Run a single inference cycle for merged mode"""
    # ONLY run if merged mode inference is explicitly enabled!
    if not (hasattr(self, 'inference_mode_enabled') and self.inference_mode_enabled):
        return  # ← Prevents predictions when disabled
    
    if hasattr(self, 'merged_mode_inference_engine') and self.merged_mode_inference_engine:
        # Generate predictions...
```

**Real Inference Cycle:**
```python
def run_real_inference_cycle(self):
    """Run a real inference cycle using the loaded model"""
    if not self.inference_active or not hasattr(self, 'inference_engine'):
        return  # ← Already had this check
    
    # Run inference...
```

### Renamed Engine Variables

- Real Inference: Still uses `self.inference_engine` (Advanced Inference Engine)
- Merged Mode: Now uses `self.merged_mode_inference_engine` (Mock Engine)

## Files Modified

### `src/gui/painting_interface.py`

**1. load_and_start_inference() - Line ~1526:**
```python
# OLD:
if not hasattr(self, 'inference_timer'):
    self.inference_timer = QTimer()
    self.inference_timer.timeout.connect(self.run_real_inference_cycle)
self.inference_timer.start(interval_ms)

# NEW:
if not hasattr(self, 'real_inference_timer'):
    self.real_inference_timer = QTimer()
    self.real_inference_timer.timeout.connect(self.run_real_inference_cycle)
self.real_inference_timer.start(interval_ms)
```

**2. stop_real_inference() - Line ~1570:**
```python
# OLD:
if hasattr(self, 'inference_timer'):
    self.inference_timer.stop()

# NEW:
if hasattr(self, 'real_inference_timer'):
    self.real_inference_timer.stop()
```

**3. setup_inference_components() - Line ~2467:**
```python
# OLD:
self.inference_engine = MockInferenceEngine()
if not hasattr(self, 'inference_timer'):
    self.inference_timer = QTimer()
    self.inference_timer.timeout.connect(self.run_inference_cycle)
self.inference_timer.start(interval_ms)

# NEW:
self.merged_mode_inference_engine = MockInferenceEngine()
if not hasattr(self, 'merged_mode_inference_timer'):
    self.merged_mode_inference_timer = QTimer()
    self.merged_mode_inference_timer.timeout.connect(self.run_merged_mode_inference_cycle)
self.merged_mode_inference_timer.start(interval_ms)
```

**4. stop_inference_components() - Line ~2496:**
```python
# OLD:
if hasattr(self, 'inference_timer'):
    self.inference_timer.stop()
if hasattr(self, 'inference_engine'):
    self.inference_engine.stop()

# NEW:
if hasattr(self, 'merged_mode_inference_timer'):
    self.merged_mode_inference_timer.stop()
if hasattr(self, 'merged_mode_inference_engine'):
    self.merged_mode_inference_engine.stop()
```

**5. update_inference_rate() - Line ~2510:**
```python
# OLD:
if hasattr(self, 'inference_timer') and ...:
    self.inference_timer.setInterval(interval_ms)

# NEW:
if hasattr(self, 'merged_mode_inference_timer') and ...:
    self.merged_mode_inference_timer.setInterval(interval_ms)
```

**6. run_inference_cycle() → run_merged_mode_inference_cycle() - Line ~2527:**
```python
# OLD:
def run_inference_cycle(self):
    if hasattr(self, 'inference_engine') and self.inference_engine:
        # Generate predictions...

# NEW:
def run_merged_mode_inference_cycle(self):
    # ONLY run if merged mode inference is explicitly enabled!
    if not (hasattr(self, 'inference_mode_enabled') and self.inference_mode_enabled):
        return
    
    if hasattr(self, 'merged_mode_inference_engine') and self.merged_mode_inference_engine:
        # Generate predictions...
```

## Expected Behavior After Fix

### Real Inference Mode (Inference Mode Tab)
1. Select a trained model from dropdown
2. Click "Load Model & Start Inference"
3. ✅ Diamond-shaped predictions appear on canvas
4. Click "Stop Inference"
5. ✅ Predictions stop immediately
6. ✅ No interference with other modes

### Merged Mode Inference
1. Check "Enable Inference Mode" checkbox
2. ✅ Glowing circle predictions appear on canvas
3. Uncheck "Enable Inference Mode"
4. ✅ Predictions stop immediately
5. ✅ No interference with real inference mode

### When BOTH Are Disabled
1. No checkboxes enabled
2. No "Start Inference" button pressed
3. ✅ **NO predictions should appear at all**
4. ✅ Canvas should only show user drawing

## Testing

Run the app and verify:
```bash
cd "d:\MEGA\Projects\Consciousness\consciousness-app"
python run.py --mode generate --test-rng --no-eeg --debug
```

**Test Cases:**

1. **Start app without enabling anything**
   - ✅ Should see NO inference predictions
   - ✅ Only user drawing appears

2. **Enable merged mode inference**
   - ✅ Glowing circles should appear
   - ✅ Disable checkbox → circles stop

3. **Go to Inference Mode tab, load model**
   - ✅ Diamond shapes should appear
   - ✅ Click stop → diamonds stop

4. **Try both at same time**
   - ✅ Both should work independently
   - ✅ No timer conflicts

## Summary

**Before:**
- ❌ Predictions running even when disabled
- ❌ Timer conflicts between modes
- ❌ Can't use both modes simultaneously

**After:**
- ✅ Predictions only when explicitly enabled
- ✅ Separate timers for each mode
- ✅ Both modes can coexist
- ✅ Clean enable/disable control
