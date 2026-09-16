"""Automated dataset validation, quality classification, and reporting."""

from automation.policy import ThresholdProfile, get_threshold_profile
from automation.service import AutoValidationService, run_auto_validation
from automation.types import DatasetVerdict, ValidationBatchResult

__all__ = [
    "AutoValidationService",
    "run_auto_validation",
    "ThresholdProfile",
    "get_threshold_profile",
    "DatasetVerdict",
    "ValidationBatchResult",
]
