#!/usr/bin/env python3
"""
Live EEG Mode Launcher
======================

This script launches the Consciousness App with REAL EEG hardware enabled.

It automatically:
- Enables real Emotiv EEG headset connection
- Uses real TrueRNG device (or mock if --test-rng is specified)
- Runs in generate or inference mode with live brain data
- Configures optimal settings for live sessions

Quick Start:

1. Generate Mode (Data Collection with Real EEG):
   python run_live.py

2. Inference Mode (Model Predictions with Real EEG):
   python run_live.py --mode inference

3. Generate Mode with Mock RNG (Real EEG, Test RNG):
   python run_live.py --test-rng

4. Inference Mode with Full Live Hardware:
   python run_live.py --mode inference

Hardware Requirements:
- Emotiv EEG headset (configured in config/app_config.yaml)
- TrueRNG V3 device (optional if using --test-rng)
- Valid Emotiv credentials in config file

The app will:
✓ Connect to real Emotiv EEG headset
✓ Stream live brainwave data (128 Hz)
✓ Use real TrueRNG (unless --test-rng specified)
✓ Enable full consciousness data generation
✓ Support model training with real brain data
✓ Run inference with live brain-guided predictions

For testing without EEG, use run.py with --no-eeg flag instead.
"""

import sys
import os
import subprocess

def main():
    """Launch with real EEG enabled."""
    
    # Build command
    cmd = [sys.executable, 'run.py']
    
    # Parse mode from arguments
    mode = 'generate'  # default
    test_rng = False
    extra_args = []
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--mode' and i + 1 < len(sys.argv) - 1:
            mode = sys.argv[i + 2]
        elif arg == '--test-rng':
            test_rng = True
        elif arg not in ['--mode', mode]:
            extra_args.append(arg)
    
    # Add mode
    cmd.extend(['--mode', mode])
    
    # Add test-rng if specified
    if test_rng:
        cmd.append('--test-rng')
    
    # Force real EEG by specifying eeg-source as auto (will try Cortex first)
    cmd.extend(['--eeg-source', 'auto'])
    
    # Enable debug logging for better visibility
    if '--debug' not in extra_args:
        cmd.append('--debug')
    
    # Add any extra arguments
    cmd.extend(extra_args)
    
    print("=" * 70)
    print("🧠 LAUNCHING LIVE EEG MODE")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"EEG Source: Real Emotiv Headset (auto-detect)")
    print(f"RNG Source: {'Mock/Test' if test_rng else 'Real TrueRNG V3'}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)
    print()
    print("Connecting to Emotiv EEG headset...")
    print("(Make sure headset is on and Emotiv software is running)")
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode
    except KeyboardInterrupt:
        print("\n\n⚠️  Session interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error launching live mode: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
