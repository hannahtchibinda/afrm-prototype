"""Exceptions raised by AFRM."""


class AfrmError(Exception):
    """Base class for all errors AFRM expects and reports cleanly."""


class DataError(AfrmError):
    """Input data or configuration failed validation."""


class ConvergenceError(AfrmError):
    """The fuel-plan iteration didn't settle within its limit."""
