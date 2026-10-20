"""DiDCheck: pretreatment variance diagnostics for two-group panel designs."""
from .diagnostics import diagnose_arrays, variance_test
from .results import DiagnosticResult, TestResult
from .simulation import simulate_panel

__version__ = "0.1.0"
__all__ = ["variance_test", "diagnose_arrays", "simulate_panel", "DiagnosticResult", "TestResult", "__version__"]
