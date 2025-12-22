"""
Tests for grid generation functions.
"""

import numpy as np
from seap.prediction.grids import (
    grid_for_scatter,
    cartesian_grid,
    extract_inscribed_ball,
    cartesian2polar,
)


class TestGridForScatter:
    """Test class for grid_for_scatter function."""

    def test_grid_for_scatter_basic(self):
        """Test basic grid_for_scatter."""
        x, y, z = grid_for_scatter([2, 2, 2])
        
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(z, np.ndarray)
        assert len(x) == 8  # 2 * 2 * 2
        assert len(y) == 8
        assert len(z) == 8

    def test_grid_for_scatter_values(self):
        """Test grid_for_scatter returns correct values."""
        x, y, z = grid_for_scatter([2, 2, 2])
        
        # Check that all combinations are present
        expected_x = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        expected_y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        expected_z = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        
        np.testing.assert_array_equal(x, expected_x)
        np.testing.assert_array_equal(y, expected_y)
        np.testing.assert_array_equal(z, expected_z)

    def test_grid_for_scatter_different_sizes(self):
        """Test grid_for_scatter with different sizes."""
        x, y, z = grid_for_scatter([3, 2, 4])
        
        assert len(x) == 3 * 2 * 4
        assert len(y) == 3 * 2 * 4
        assert len(z) == 3 * 2 * 4


class TestCartesianGrid:
    """Test class for cartesian_grid function."""

    def test_cartesian_grid_basic(self):
        """Test basic cartesian_grid."""
        grid = cartesian_grid([2, 2, 2], 2.0)
        
        assert isinstance(grid, np.ndarray)
        assert grid.shape[1] == 3  # 3D coordinates
        assert grid.shape[0] == 8  # 2 * 2 * 2 points

    def test_cartesian_grid_centered(self):
        """Test that cartesian_grid is centered around origin."""
        grid = cartesian_grid([2, 2, 2], 2.0)
        
        # Check that points are symmetric around origin
        assert np.allclose(grid[0], [-0.5, -0.5, -0.5])
        assert np.allclose(grid[-1], [0.5, 0.5, 0.5])

    def test_cartesian_grid_size(self):
        """Test cartesian_grid with different sizes."""
        grid_small = cartesian_grid([2, 2, 2], 1.0)
        grid_large = cartesian_grid([2, 2, 2], 4.0)
        
        # Both should have same number of points
        assert grid_small.shape[0] == grid_large.shape[0]
        # But different coordinate ranges
        assert np.max(np.abs(grid_small)) < np.max(np.abs(grid_large))

    def test_cartesian_grid_odd_divisions(self):
        """Test cartesian_grid with odd number of divisions."""
        grid = cartesian_grid([3, 3, 3], 2.0)
        
        assert grid.shape[0] == 27  # 3 * 3 * 3
        # Check that grid is properly formed
        assert grid.shape[1] == 3


class TestCartesian2Polar:
    """Test class for cartesian2polar function."""

    def test_cartesian2polar_basic(self):
        """Test basic cartesian to polar conversion."""
        pos_cart = np.array([[1.0, 0.0, 0.0]])
        pos_pol = cartesian2polar(pos_cart)
        
        assert isinstance(pos_pol, np.ndarray)
        assert pos_pol.shape == (1, 3)
        assert pos_pol[0, 0] == 1.0  # r should be 1.0

    def test_cartesian2polar_origin(self):
        """Test cartesian2polar with origin point."""
        pos_cart = np.array([[0.0, 0.0, 0.0]])
        pos_pol = cartesian2polar(pos_cart)
        
        assert pos_pol[0, 0] == 0.0  # r should be 0
        assert pos_pol[0, 1] == 0.0  # theta should be 0 (handled NaN)

    def test_cartesian2polar_z_axis(self):
        """Test cartesian2polar with point on z-axis."""
        pos_cart = np.array([[0.0, 0.0, 1.0]])
        pos_pol = cartesian2polar(pos_cart)
        
        assert np.isclose(pos_pol[0, 0], 1.0)  # r = 1
        assert np.isclose(pos_pol[0, 1], 0.0)  # theta = 0 (along z-axis)

    def test_cartesian2polar_multiple_points(self):
        """Test cartesian2polar with multiple points."""
        pos_cart = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        pos_pol = cartesian2polar(pos_cart)
        
        assert pos_pol.shape == (3, 3)
        # All should have r = 1
        assert np.allclose(pos_pol[:, 0], 1.0)

    def test_cartesian2polar_phi_range(self):
        """Test that phi is in [0, 2pi] range."""
        pos_cart = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        pos_pol = cartesian2polar(pos_cart)
        
        # All phi values should be in [0, 2pi]
        assert np.all(pos_pol[:, 2] >= 0)
        assert np.all(pos_pol[:, 2] < 2 * np.pi)


class TestExtractInscribedBall:
    """Test class for extract_inscribed_ball function."""

    def test_extract_inscribed_ball_basic(self):
        """Test basic extract_inscribed_ball."""
        # Create a simple grid
        grid = cartesian_grid([4, 4, 4], 2.0)
        pos_pol, map_idx = extract_inscribed_ball(grid, 2.0)
        
        assert isinstance(pos_pol, np.ndarray)
        assert isinstance(map_idx, list)
        assert pos_pol.shape[1] == 3  # polar coordinates (r, theta, phi)
        assert len(map_idx) == pos_pol.shape[0]

    def test_extract_inscribed_ball_origin_excluded(self):
        """Test that origin is excluded from inscribed ball."""
        grid = cartesian_grid([4, 4, 4], 2.0)
        pos_pol, map_idx = extract_inscribed_ball(grid, 2.0)
        
        # Origin should not be in the result (r=0 points excluded)
        for point in pos_pol:
            assert point[0] > 0  # r > 0

    def test_extract_inscribed_ball_radius(self):
        """Test that extracted points are within inscribed sphere."""
        grid = cartesian_grid([6, 6, 6], 2.0)
        pos_pol, map_idx = extract_inscribed_ball(grid, 2.0)
        
        rmax = 2.0 / 2.0  # size / 2
        # All radial distances should be <= rmax
        assert np.all(pos_pol[:, 0] <= rmax + 1e-10)

    def test_extract_inscribed_ball_small_grid(self):
        """Test extract_inscribed_ball with very small grid."""
        # Create a small grid with only a few points
        grid = np.array([
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
        ])
        pos_pol, map_idx = extract_inscribed_ball(grid, 1.0)
        
        # Should return some points within the sphere
        assert isinstance(pos_pol, np.ndarray)
        assert isinstance(map_idx, list)
        assert len(map_idx) == pos_pol.shape[0]

    def test_extract_inscribed_ball_zero_point(self):
        """Test that zero point is explicitly excluded."""
        # Create grid with explicit zero point
        grid = np.array([
            [0.0, 0.0, 0.0],  # Origin - should be excluded
            [0.1, 0.1, 0.1],  # Non-zero point
        ])
        pos_pol, map_idx = extract_inscribed_ball(grid, 2.0)
        
        # Origin should be excluded (tests line 96: if np.all(pos == 0.0): continue)
        assert len(pos_pol) == 1
        assert pos_pol[0, 0] > 0  # r > 0

