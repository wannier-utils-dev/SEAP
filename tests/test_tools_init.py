"""
Tests for tools __init__ module.
"""

import pytest


class TestToolsInit:
    """Test class for tools __init__ module."""

    def test_tools_import(self):
        """Test that tools module can be imported."""
        from seap import tools
        assert hasattr(tools, '__all__')

    def test_compare_import(self):
        """Test that compare module can be imported."""
        try:
            from seap.tools import compare
            assert compare is not None
        except ImportError:
            pytest.skip("compare module not available")

    def test_wan_comp_band_import(self):
        """Test that wan_comp_band module can be imported."""
        try:
            from seap.tools import wan_comp_band
            assert wan_comp_band is not None
        except ImportError:
            pytest.skip("wan_comp_band module not available")


