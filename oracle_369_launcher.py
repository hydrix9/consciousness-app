"""
Main entry point for the 369 Oracle Consciousness System
Run this to launch the sophisticated consciousness oracle interface
"""

import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gui.oracle_369 import main

if __name__ == '__main__':
    print("🔮 Launching 369 Oracle - Consciousness Interpretation System")
    print("=" * 60)
    print("Welcome to the sophisticated consciousness oracle.")
    print("This system uses advanced mathematics and three consciousness")
    print("layers to interpret your questions through creative expression.")
    print("=" * 60)
    main()