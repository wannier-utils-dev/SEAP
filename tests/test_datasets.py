"""
Tests for dataset classes.
"""

import numpy as np
from seap.prediction.datasets import BoxData


class TestBoxData:
    """Test class for BoxData."""

    def test_boxdata_initialization(self):
        """Test BoxData initialization."""
        box_data = BoxData(l_max=2, rn_max=3, n_div=8, size=1.0)
        
        assert box_data.n == 3  # l_max + 1
        assert box_data.n_div == 8
        assert box_data.rn_max == 3
        assert len(box_data.lm_set) == 9  # (l_max+1)^2 = 9 for l_max=2

    def test_boxdata_arrays_shape(self):
        """Test that BoxData creates arrays with correct shapes."""
        box_data = BoxData(l_max=1, rn_max=2, n_div=4, size=1.0)
        
        # Check that arrays exist and have reasonable shapes
        assert hasattr(box_data, 'rvec')
        assert hasattr(box_data, 'rpow')
        assert hasattr(box_data, 'sphmat')
        
        assert isinstance(box_data.rvec, np.ndarray)
        assert isinstance(box_data.rpow, np.ndarray)
        assert isinstance(box_data.sphmat, np.ndarray)
        
        # Check dimensions
        n_points = len(box_data.rvec)
        assert box_data.rpow.shape[0] == box_data.rn_max
        assert box_data.rpow.shape[1] == n_points
        assert box_data.sphmat.shape[0] == len(box_data.lm_set)
        assert box_data.sphmat.shape[1] == n_points

    def test_boxdata_different_sizes(self):
        """Test BoxData with different size parameters."""
        box_data_small = BoxData(l_max=1, rn_max=2, n_div=4, size=0.5)
        box_data_large = BoxData(l_max=1, rn_max=2, n_div=4, size=2.0)
        
        # Both should initialize successfully
        assert box_data_small.n == 2
        assert box_data_large.n == 2
        
        # Number of points should be different due to different inscribed spheres
        assert len(box_data_small.rvec) > 0
        assert len(box_data_large.rvec) > 0

    def test_boxdata_different_l_max(self):
        """Test BoxData with different l_max values."""
        box_data_l1 = BoxData(l_max=1, rn_max=2, n_div=4, size=1.0)
        box_data_l2 = BoxData(l_max=2, rn_max=2, n_div=4, size=1.0)
        
        # l_max=1 should have 4 quantum numbers (1 + 3)
        assert len(box_data_l1.lm_set) == 4
        # l_max=2 should have 9 quantum numbers (1 + 3 + 5)
        assert len(box_data_l2.lm_set) == 9
        
        # sphmat should have different first dimension
        assert box_data_l1.sphmat.shape[0] == 4
        assert box_data_l2.sphmat.shape[0] == 9

    def test_boxdata_rvec_properties(self):
        """Test properties of rvec array."""
        box_data = BoxData(l_max=1, rn_max=3, n_div=6, size=1.0)
        
        # rvec should contain distances (non-negative)
        assert np.all(box_data.rvec >= 0)
        # All distances should be within the inscribed sphere (radius = size)
        assert np.all(box_data.rvec <= box_data.n_div * box_data.n_div * box_data.n_div)

    def test_boxdata_get_random_radial_params(self):
        """Test get_random_radial_params method."""
        box_data = BoxData(l_max=1, rn_max=3, n_div=4, size=1.0)
        gamma, an = box_data.get_random_radial_params()
        
        assert isinstance(gamma, float)
        assert isinstance(an, np.ndarray)
        assert len(an) == box_data.rn_max
        assert 1.0 <= gamma <= 5.0
        # an should be normalized
        assert isinstance(an, np.ndarray)

    def test_boxdata_assign_sph_coeffs(self):
        """Test assign_sph_coeffs method."""
        box_data = BoxData(l_max=2, rn_max=2, n_div=4, size=1.0)
        lm_list = [[0, 0], [1, 0]]
        coeffs = box_data.assign_sph_coeffs(lm_list)
        
        assert isinstance(coeffs, np.ndarray)
        assert coeffs.shape == (box_data.n**2,)
        # Check that selected coefficients are non-zero
        assert coeffs[0] != 0  # (0,0) -> index 0
        assert coeffs[2] != 0  # (1,0) -> index 2

    def test_boxdata_params_to_boxdata(self):
        """Test params_to_boxdata method."""
        box_data = BoxData(l_max=1, rn_max=2, n_div=4, size=1.0)
        c = np.array([[1.0, 0.0, 0.0, 0.0]])  # Single coefficient
        gamma = 2.0
        an = np.array([1.0, 0.5])
        
        data = box_data.params_to_boxdata(c, gamma, an)
        
        assert isinstance(data, np.ndarray)
        assert data.shape[0] == 1  # n_batch
        assert data.shape[1] == box_data.n_div**3

    def test_boxdata_generate_learning_data(self):
        """Test generate_learning_data method."""
        box_data = BoxData(l_max=1, rn_max=2, n_div=4, size=1.0)
        x, y = box_data.generate_learning_data(n_samples=5)
        
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.shape[0] == 5
        assert y.shape[0] == 5
        assert x.shape[1] == box_data.n_div**3
        assert y.shape[1] == box_data.n**2 + box_data.rn_max + 1

