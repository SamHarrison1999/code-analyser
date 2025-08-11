# code_analyser/src/ml/__init__.py
"""
Initialises the machine learning subsystem for the Code Analyser.

This module provides:
- Supervised and reinforcement learning models for annotation prediction
- Inference utilities for AI-enhanced metric scoring
- Annotation generation using Open AI
- Caching support for per-file annotation reuse
"""
# Keep initialisation lightweight to avoid circular imports and heavy start-up costs; we expose the same public API via lazy imports below.

# Public symbols re-exported by this package (unchanged interface for callers).
__all__ = [
    "AI_CACHE_DIR",
    "AnnotationEngine",
    "AnnotationClassifier",
    "compute_reward",
    "label_to_score",
]
# Provide a simple package version for diagnostics and logging.
__version__ = "0.1.1"

# TYPE_CHECKING allows editors and static tools to see types without importing heavy modules at runtime.
from typing import TYPE_CHECKING


# Lazily import attributes only when they are first accessed; this prevents import cycles (e.g., inference importing back into ml).
def __getattr__(name: str):
    # Route attribute access to the appropriate submodule on demand.
    if name == "AI_CACHE_DIR":
        # Config is lightweight; import locally to avoid global side effects.
        from .config import AI_CACHE_DIR as value

        return value
    if name == "AnnotationEngine":
        # Defer importing the engine to avoid triggering heavy model dependencies during package import.
        from .inference import AnnotationEngine as value

        return value
    if name == "AnnotationClassifier":
        # Import the TF/PyTorch classifier wrapper only when explicitly referenced.
        from .model_tf import AnnotationClassifier as value

        return value
    if name == "compute_reward":
        # Reward functions are only needed for RL-related workflows; import on demand.
        from .reward_functions import compute_reward as value

        return value
    if name == "label_to_score":
        # Utility to map labels to numeric scores for metrics/overlays; import lazily.
        from .ai_signal_utils import label_to_score as value

        return value
    # If the requested name is not one we export, raise the standard AttributeError.
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Ensure dir() lists our public API alongside normal globals for a nicer REPL/dev experience.
def __dir__():
    return sorted(list(globals().keys()) + __all__)


# During static type-checking, make symbols available without lazy indirection (no runtime cost).
if TYPE_CHECKING:
    from .config import AI_CACHE_DIR as AI_CACHE_DIR  # type: ignore
    from .inference import AnnotationEngine as AnnotationEngine  # type: ignore
    from .model_tf import AnnotationClassifier as AnnotationClassifier  # type: ignore
    from .reward_functions import compute_reward as compute_reward  # type: ignore
    from .ai_signal_utils import label_to_score as label_to_score  # type: ignore
