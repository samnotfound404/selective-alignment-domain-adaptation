"""
CC-SAT-DA: Class-Conditioned Selective Alignment Transfer for Domain Adaptation
Cross-Domain Skin Lesion Classification (ISIC 2018 dermoscopy -> PAD-UFES-20 smartphone photography)
"""

__version__ = "0.2.0"
__author__ = "Bhavesh and Contributors"

from .models.cc_satda import CCSATDANet
from .models.satda_baseline import SATDANetwork, DynamicFeatureSelector
from .utils.calibration import compute_class_priors, calibrate_logits
from .utils.metrics import compute_classification_metrics
