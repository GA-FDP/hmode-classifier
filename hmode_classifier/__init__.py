"""H-mode classifier for DIII-D over the Fusion Data Platform.

Randall Clark's ECE, profile-reflectometer and ensemble L/H-mode classifiers
(Nucl. Fusion 2026; PPCF 68 015022), packaged with pretrained weights.
"""
from pathlib import Path

from .Hmode_ML_Classifier import Hmode_Ml_Predictions

#: The v1 weights shipped with the package.
MODEL_DIR = Path(__file__).resolve().parent / "Models"

__all__ = ["Hmode_Ml_Predictions", "MODEL_DIR"]
