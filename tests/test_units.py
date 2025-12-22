"""
Tests for unit conversion constants.
"""

from seap.common.units import AU2ANG, ANG2AU


class TestUnits:
    """Test class for unit conversion constants."""

    def test_au2ang_positive(self):
        """Test that AU2ANG is a positive value."""
        assert AU2ANG > 0

    def test_ang2au_positive(self):
        """Test that ANG2AU is a positive value."""
        assert ANG2AU > 0

    def test_au2ang_ang2au_reciprocal(self):
        """Test that AU2ANG and ANG2AU are reciprocal."""
        # They should be approximately reciprocal (within numerical precision)
        product = AU2ANG * ANG2AU
        assert abs(product - 1.0) < 1e-10

    def test_au2ang_value(self):
        """Test that AU2ANG has a reasonable value (Bohr radius in Angstrom)."""
        # Bohr radius is approximately 0.529 Angstrom
        assert 0.5 < AU2ANG < 0.6

    def test_ang2au_value(self):
        """Test that ANG2AU has a reasonable value."""
        # Should be approximately 1.89 (1 / 0.529)
        assert 1.5 < ANG2AU < 2.0


