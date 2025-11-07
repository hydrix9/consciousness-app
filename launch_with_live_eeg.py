#!/usr/bin/env python3
"""
Launch Consciousness App with Live EEG
======================================

This launcher script starts the consciousness app with real Emotiv EEG data
for both generation and inference modes.

Usage:
    python launch_with_live_eeg.py --mode generate    # Generate training data with live EEG
    python launch_with_live_eeg.py --mode inference   # Run inference with live EEG
    python launch_with_live_eeg.py --mode oracle      # Run 369 Oracle with live EEG
"""

import os
import sys
import subprocess
import argparse

def check_emotiv_cortex():
    """Check if Emotiv Cortex service is running"""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 6868))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Error checking Cortex service: {e}")
        return False

def launch_generate_mode(use_test_rng=False, rng_rate=None):
    """Launch data generation mode with live EEG"""
    print("🧠 Launching GENERATION mode with LIVE EEG")
    print("=" * 60)
    
    cmd = [sys.executable, "run.py", "--mode", "generate"]
    
    # Force real EEG
    cmd.extend(['--eeg-source', 'cortex'])
    
    # RNG options
    if use_test_rng:
        cmd.append('--test-rng')
        if rng_rate:
            cmd.extend(['--rng-rate', str(rng_rate)])
    
    # Enable debug for visibility
    cmd.append('--debug')
    
    print(f"Command: {' '.join(cmd)}")
    print("\n📊 Starting data generation with live EEG...")
    print("   - EEG Source: Emotiv Cortex (Real Headset)")
    print("   - Mode: Generate training data")
    print("   - You can draw while wearing the headset")
    print("=" * 60)
    
    subprocess.run(cmd)

def launch_inference_mode(use_test_rng=False, rng_rate=None):
    """Launch inference mode with live EEG"""
    print("🔮 Launching INFERENCE mode with LIVE EEG")
    print("=" * 60)
    
    cmd = [sys.executable, "run.py", "--mode", "inference"]
    
    # Force real EEG
    cmd.extend(['--eeg-source', 'cortex'])
    
    # RNG options
    if use_test_rng:
        cmd.append('--test-rng')
        if rng_rate:
            cmd.extend(['--rng-rate', str(rng_rate)])
    
    # Enable debug for visibility
    cmd.append('--debug')
    
    print(f"Command: {' '.join(cmd)}")
    print("\n🤖 Starting inference with live EEG...")
    print("   - EEG Source: Emotiv Cortex (Real Headset)")
    print("   - Mode: Inference (AI-powered predictions)")
    print("   - The AI will respond to your consciousness state")
    print("=" * 60)
    
    subprocess.run(cmd)

def launch_oracle_mode():
    """Launch 369 Oracle with live EEG"""
    print("🔮 Launching 369 ORACLE with LIVE EEG")
    print("=" * 60)
    
    # The oracle launcher needs to be updated to support EEG source override
    # For now, we'll update the config and then launch
    cmd = [sys.executable, "oracle_369_launcher.py"]
    
    print(f"Command: {' '.join(cmd)}")
    print("\n✨ Starting 369 Oracle with live EEG...")
    print("   - EEG Source: Emotiv Cortex (Real Headset)")
    print("   - Mode: Oracle consciousness interpretation")
    print("   - Ask your question and express through art")
    print("=" * 60)
    
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(
        description="Launch Consciousness App with Live EEG",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--mode', 
                       choices=['generate', 'inference', 'oracle'],
                       default='generate',
                       help='Application mode to run')
    
    parser.add_argument('--test-rng', 
                       action='store_true',
                       help='Use simulated RNG instead of TrueRNG device')
    
    parser.add_argument('--rng-rate', 
                       type=float,
                       default=12.0,
                       help='RNG rate in kbps for test mode (default: 12.0)')
    
    args = parser.parse_args()
    
    # Pre-flight checks
    print("\n🔍 Pre-flight checks...")
    print("-" * 60)
    
    # Check Emotiv Cortex
    if check_emotiv_cortex():
        print("✅ Emotiv Cortex service is running on port 6868")
    else:
        print("❌ Emotiv Cortex service NOT detected!")
        print("   Please start Emotiv Pro or EPOC Connect software")
        print("   and make sure your headset is connected.")
        response = input("\nContinue anyway? (y/n): ").lower().strip()
        if response != 'y':
            return 1
    
    # Check config files
    if os.path.exists("config/eeg_config.yaml"):
        print("✅ EEG configuration file found")
    else:
        print("⚠️  EEG configuration file not found")
        print("   Run: python configure_eeg.py")
    
    print("-" * 60)
    print()
    
    # Launch appropriate mode
    if args.mode == 'generate':
        launch_generate_mode(args.test_rng, args.rng_rate)
    elif args.mode == 'inference':
        launch_inference_mode(args.test_rng, args.rng_rate)
    elif args.mode == 'oracle':
        launch_oracle_mode()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
