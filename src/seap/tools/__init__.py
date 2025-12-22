"""
Utility tools for SEAP package.

This module contains various utility tools for comparison and analysis.
"""

# Import modules conditionally to avoid import errors during package initialization
try:
    from . import compare
    _compare_available = True
except ImportError:
    _compare_available = False

try:
    from . import wan_comp_band
    _wan_comp_band_available = True
except ImportError:
    _wan_comp_band_available = False

__all__ = []
if _compare_available:
    __all__.append("compare")
if _wan_comp_band_available:
    __all__.append("wan_comp_band")
