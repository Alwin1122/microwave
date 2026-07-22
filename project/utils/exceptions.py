"""
utils/exceptions.py

Purpose:
    Centralized custom exception hierarchy used across the microwave imaging
    framework. Using specific exception types (instead of generic Exception)
    allows the GUI and calling code to catch and display meaningful,
    context-specific error messages to the user.

Description:
    All exceptions inherit from MicrowaveFrameworkError so callers can choose
    to catch broadly (MicrowaveFrameworkError) or narrowly (e.g.
    UnsupportedFileFormatError) depending on their needs.
"""


class MicrowaveFrameworkError(Exception):
    """Base class for all errors raised by this framework."""


class UnsupportedFileFormatError(MicrowaveFrameworkError):
    """Raised when a file extension is not one of the supported formats
    (.mat, .s1p, .s2p, .s4p, .s8p)."""


class CorruptedFileError(MicrowaveFrameworkError):
    """Raised when a file exists and has a supported extension, but its
    contents cannot be parsed (corrupted header, truncated binary data,
    unreadable structure, etc.)."""


class MissingVariableError(MicrowaveFrameworkError):
    """Raised when a required variable / field is missing from a MATLAB
    dataset (e.g. no S-parameter matrix or frequency vector found)."""


class EmptyDatasetError(MicrowaveFrameworkError):
    """Raised when a dataset is successfully parsed but contains no usable
    samples (zero-length frequency vector or empty S-parameter array)."""


class InvalidFrequencyError(MicrowaveFrameworkError):
    """Raised when frequency values are missing, non-numeric, negative,
    not monotonically increasing, or otherwise invalid."""


class InvalidSParameterError(MicrowaveFrameworkError):
    """Raised when S-parameter data has an invalid shape, contains NaN/Inf
    values, or does not match the number of frequency points/ports."""


class DatasetValidationError(MicrowaveFrameworkError):
    """Raised by the validator when a dataset fails one or more validation
    checks. Carries a list of individual issues in `errors`."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or [message]
