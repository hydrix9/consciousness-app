"""
Hardware interface package initialization
"""

from .truerng_v3 import TrueRNGV3, RNGSample
from .emotiv_eeg import EmotivEEG, EEGSample

__all__ = ['TrueRNGV3', 'RNGSample', 'EmotivEEG', 'EEGSample']