"""
Tests for lattice coordinate conversion functions.
"""

import numpy as np
from seap.common.lattice import crys2cart, cart2crys


class TestLatticeConversion:
    """Test class for lattice coordinate conversion functions."""

    def test_crys2cart_simple(self):
        """Test crystal to Cartesian conversion with simple cubic lattice."""
        # Simple cubic lattice with unit cell size 1
        lattice = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        
        # Crystal coordinates (fractional)
        pos_crys = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5],
                            [1.0, 1.0, 1.0]])
        
        # Expected Cartesian coordinates
        expected = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5],
                            [1.0, 1.0, 1.0]])
        
        result = crys2cart(pos_crys, lattice)
        np.testing.assert_array_almost_equal(result, expected)

    def test_cart2crys_simple(self):
        """Test Cartesian to crystal conversion with simple cubic lattice."""
        # Simple cubic lattice
        lattice = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        
        # Cartesian coordinates
        pos_cart = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5],
                            [1.0, 1.0, 1.0]])
        
        # Expected crystal coordinates
        expected = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5],
                            [1.0, 1.0, 1.0]])
        
        result = cart2crys(pos_cart, lattice)
        np.testing.assert_array_almost_equal(result, expected)

    def test_round_trip_conversion(self):
        """Test that crys2cart and cart2crys are inverse operations."""
        # Orthorhombic lattice
        lattice = np.array([[2.0, 0.0, 0.0],
                           [0.0, 3.0, 0.0],
                           [0.0, 0.0, 4.0]])
        
        # Original crystal coordinates
        pos_crys_original = np.array([[0.0, 0.0, 0.0],
                                     [0.25, 0.33, 0.5],
                                     [0.5, 0.5, 0.5]])
        
        # Convert to Cartesian and back
        pos_cart = crys2cart(pos_crys_original, lattice)
        pos_crys_result = cart2crys(pos_cart, lattice)
        
        np.testing.assert_array_almost_equal(
            pos_crys_original, pos_crys_result, decimal=10
        )

    def test_crys2cart_orthorhombic(self):
        """Test crystal to Cartesian conversion with orthorhombic lattice."""
        # Orthorhombic lattice
        lattice = np.array([[2.0, 0.0, 0.0],
                           [0.0, 3.0, 0.0],
                           [0.0, 0.0, 4.0]])
        
        # Crystal coordinates
        pos_crys = np.array([[0.5, 0.5, 0.5]])
        
        # Expected Cartesian coordinates
        expected = np.array([[1.0, 1.5, 2.0]])
        
        result = crys2cart(pos_crys, lattice)
        np.testing.assert_array_almost_equal(result, expected)

    def test_cart2crys_orthorhombic(self):
        """Test Cartesian to crystal conversion with orthorhombic lattice."""
        # Orthorhombic lattice
        lattice = np.array([[2.0, 0.0, 0.0],
                           [0.0, 3.0, 0.0],
                           [0.0, 0.0, 4.0]])
        
        # Cartesian coordinates
        pos_cart = np.array([[1.0, 1.5, 2.0]])
        
        # Expected crystal coordinates
        expected = np.array([[0.5, 0.5, 0.5]])
        
        result = cart2crys(pos_cart, lattice)
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_point(self):
        """Test conversion functions with a single point."""
        lattice = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        
        pos_crys = np.array([[0.25, 0.5, 0.75]])
        pos_cart = crys2cart(pos_crys, lattice)
        pos_crys_back = cart2crys(pos_cart, lattice)
        
        np.testing.assert_array_almost_equal(pos_crys, pos_crys_back)


