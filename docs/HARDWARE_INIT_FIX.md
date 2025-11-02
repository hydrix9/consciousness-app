# Hardware Initialization Fix for Training Mode

## Issue
Training mode was incorrectly initializing hardware (TrueRNG v3 and Emotiv EEG via Cortex) even though it should only load data from saved HDF5 session files.

**User Report:**
> "for some reason its booting up cortex eeg stuff when training / this doesn't make sense, as the training data from --mode generate should have it"

## Root Cause
In `src/main.py`, the hardware initialization logic (lines 550-598) was executing for **all modes** (generate, train, inference) unless the `--no-hardware` or `--test-rng` flags were explicitly used.

This was illogical because:
- **Training mode** loads pre-recorded data from HDF5 session files - no live hardware needed
- **Generate mode** collects live data from hardware - hardware required
- **Inference mode** makes real-time predictions from live hardware - hardware required

## Solution
Wrapped the entire hardware initialization block in a mode check:

```python
if args.mode != 'train':
    # Initialize hardware for generate/inference modes
    if not args.no_hardware and not args.test_rng:
        app.initialize_hardware()
    else:
        # Mock hardware setup
        ...
    
    # Initialize data logger (requires hardware)
    app.initialize_data_logger()
else:
    # Training mode - no hardware needed
    logging.info("Training mode: skipping hardware initialization (loading from saved session files)")
    app.rng_device = None
    app.eeg_bridge = None

# Always initialize ML components (they don't require hardware)
app.initialize_ml_components()
```

## Changes Made

### `src/main.py` - Lines 550-610

**Before:**
- Hardware always initialized unless `--no-hardware` or `--test-rng` flags used
- Data logger always initialized
- No mode-specific logic

**After:**
- Hardware initialization **skipped** for training mode
- Data logger initialization **skipped** for training mode
- Hardware still initializes normally for generate/inference modes
- ML components always initialize (mode-independent)

### Key Modifications

1. **Added outer mode check** (line 551):
   ```python
   if args.mode != 'train':
   ```

2. **Added training mode branch** (lines 603-607):
   ```python
   else:
       # Training mode - no hardware needed
       logging.info("Training mode: skipping hardware initialization (loading from saved session files)")
       app.rng_device = None
       app.eeg_bridge = None
   ```

3. **Moved data logger initialization** inside generate/inference block (line 602)

## Verification

Created `test_training_mode_no_hardware.py` to verify the fix:

```
✅ ALL TESTS PASSED!

📝 Summary:
   • Training mode: No hardware initialization ✅
   • Generate mode: Hardware initialization works ✅
   • Fix successfully verified! ✅
```

## Mode-Specific Behavior

| Mode | Hardware Init | Data Logger Init | ML Components Init |
|------|--------------|------------------|-------------------|
| **generate** | ✅ Yes | ✅ Yes | ✅ Yes |
| **train** | ❌ No | ❌ No | ✅ Yes |
| **inference** | ✅ Yes | ✅ Yes | ✅ Yes |

## Benefits

1. **Faster training startup** - no hardware detection delays
2. **Offline training** - can train on machines without TrueRNG/EEG hardware
3. **Logical separation** - training uses historical data, generate/inference use live data
4. **Resource efficiency** - doesn't waste time initializing unused hardware

## Testing

To test the fix:

```bash
# Training mode - should NOT initialize hardware
python -m src.main --mode train --data-dir data

# Generate mode - should initialize hardware normally
python -m src.main --mode generate

# Inference mode - should initialize hardware normally
python -m src.main --mode inference
```

Expected output for training mode:
```
INFO - Training mode: skipping hardware initialization (loading from saved session files)
INFO - Found 2 training files in data
INFO - Starting model training...
```

**No Cortex EEG or TrueRNG initialization messages should appear in training mode!**

## Related Files

- `src/main.py` - Hardware initialization logic (MODIFIED)
- `src/ml/multi_model_trainer.py` - Real data loading (uses session files)
- `src/data/real_data_loader.py` - HDF5 session file loader
- `test_training_mode_no_hardware.py` - Verification test (NEW)

## Implementation Date
2025-01-XX

## Status
✅ **FIXED AND VERIFIED**
