# code_analyser/src/ml/__init__.py

"""
Initialises the machine learning subsystem for the Code Analyser.

This module provides:
- Supervised and reinforcement learning models for annotation prediction
- Inference utilities for AI-enhanced metric scoring
- Annotation generation using Together.ai
- Caching support for per-file annotation reuse
"""

from .config import AI_CACHE_DIR
from .inference import AnnotationEngine
from .model_tf import AnnotationClassifier
from .reward_functions import compute_reward
from .ai_signal_utils import label_to_score

__all__ = [
    "AI_CACHE_DIR",
    "AnnotationEngine",
    "AnnotationClassifier",
    "compute_reward",
    "label_to_score",
]
