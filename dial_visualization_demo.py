"""
Quick Demo: Interlocking Dial Visualization

This creates a simple visual demo showing how the white dial overlay works.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

print("🎨 Interlocking Dial Visualization Demo")
print("=" * 60)

from src.gui.painting_interface import DIAL_SYSTEM_AVAILABLE

if DIAL_SYSTEM_AVAILABLE:
    print("✅ Dial system loaded successfully!")
    print()
    print("📋 FEATURE SUMMARY:")
    print("   • White brush stroke overlay for 3D dial geometry")
    print("   • Real-time conversion of curves to interlocking dials")
    print("   • Toggle on/off with checkbox in UI")
    print("   • Shows dial centers, boundaries, and curve paths")
    print()
    print("🎮 HOW TO USE:")
    print("   1. Start app:")
    print("      python run.py --mode generate --test-rng --no-eeg --debug")
    print()
    print("   2. In the Drawing Controls panel:")
    print("      ✅ Check 'Show Interlocking Dials' checkbox")
    print()
    print("   3. Draw curved strokes:")
    print("      • Draw circles, spirals, or any curved shapes")
    print("      • White overlays will appear showing 3D geometry")
    print("      • Each stroke creates dial geometry")
    print()
    print("   4. Visual elements you'll see:")
    print("      • Solid white curves (2px) - the 3D dial paths")
    print("      • Dashed white circles - dial boundaries")
    print("      • White dots - dial center points")
    print()
    print("✨ FEATURES:")
    print("   • Non-destructive overlay (doesn't affect your drawing)")
    print("   • Real-time geometry conversion")
    print("   • Multiple interlocking dials")
    print("   • Toggle on/off anytime")
    print()
    print("🎯 PERFECT FOR:")
    print("   • Understanding dial geometry generation")
    print("   • Visual feedback on 3D curve interpretation")
    print("   • Debugging dial system behavior")
    print("   • Creating interlocking geometric patterns")
    
else:
    print("❌ Dial system not available!")
    print("   Check that src/utils/curve_3d.py exists")
