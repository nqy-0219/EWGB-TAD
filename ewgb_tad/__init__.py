"""
EWGB-TAD: Entropy-Weighted Multi-View Granular-Ball Trajectory Anomaly Detection.
"""

from .granular_ball import (
    GranularBall, build_granular_balls,
    compute_feature_entropy_weights, EWGBDetector,
)
from .detector import EWGBTAD, evaluate_detector

__version__ = "1.0.0"
__all__ = [
    "GranularBall", "build_granular_balls",
    "compute_feature_entropy_weights", "EWGBDetector",
    "EWGBTAD", "evaluate_detector",
]
