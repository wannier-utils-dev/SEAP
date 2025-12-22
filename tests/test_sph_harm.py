"""
Tests for spherical harmonics functions.
"""

import numpy as np
from seap.prediction.sph_harm import (
    quantum_number,
    orb_symbol,
    spherical_harmonics,
    generate_S_mat,
)


class TestQuantumNumber:
    """Test class for quantum number generation."""

    def test_quantum_number_n0(self):
        """Test quantum_number with n=0."""
        result = quantum_number(0)
        assert result == []

    def test_quantum_number_n1(self):
        """Test quantum_number with n=1."""
        result = quantum_number(1)
        expected = [[0, 0]]
        assert result == expected

    def test_quantum_number_n2(self):
        """Test quantum_number with n=2."""
        result = quantum_number(2)
        expected = [[0, 0], [1, -1], [1, 0], [1, 1]]
        assert result == expected

    def test_quantum_number_n3(self):
        """Test quantum_number with n=3."""
        result = quantum_number(3)
        # Should have 9 elements: 1 (l=0) + 3 (l=1) + 5 (l=2) = 9
        assert len(result) == 9
        assert [0, 0] in result
        assert [1, -1] in result
        assert [1, 0] in result
        assert [1, 1] in result
        assert [2, -2] in result
        assert [2, -1] in result
        assert [2, 0] in result
        assert [2, 1] in result
        assert [2, 2] in result

    def test_quantum_number_structure(self):
        """Test that quantum_number returns correct structure."""
        result = quantum_number(2)
        for item in result:
            assert isinstance(item, list)
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], int)


class TestOrbSymbol:
    """Test class for orbital symbol mapping."""

    def test_orb_symbol_s(self):
        """Test s orbital symbol."""
        assert orb_symbol[(0, 0)] == 's'

    def test_orb_symbol_p(self):
        """Test p orbital symbols."""
        assert orb_symbol[(1, -1)] == 'py'
        assert orb_symbol[(1, 0)] == 'pz'
        assert orb_symbol[(1, 1)] == 'px'

    def test_orb_symbol_d(self):
        """Test d orbital symbols."""
        assert orb_symbol[(2, -2)] == 'dxy'
        assert orb_symbol[(2, -1)] == 'dyz'
        assert orb_symbol[(2, 0)] == 'dz2'
        assert orb_symbol[(2, 1)] == 'dxz'
        assert orb_symbol[(2, 2)] == 'dx2-y2'

    def test_orb_symbol_keys(self):
        """Test that orb_symbol contains expected keys."""
        expected_keys = [
            (0, 0), (1, -1), (1, 0), (1, 1),
            (2, -2), (2, -1), (2, 0), (2, 1), (2, 2),
            (3, -3), (3, -2), (3, -1), (3, 0),
            (3, 1), (3, 2), (3, 3),
        ]
        for key in expected_keys:
            assert key in orb_symbol


class TestSphericalHarmonics:
    """Test class for spherical harmonics calculation."""

    def test_spherical_harmonics_m0(self):
        """Test spherical harmonics with m=0."""
        theta = np.array([0.0, np.pi / 2, np.pi])
        phi = np.array([0.0, 0.0, 0.0])
        result = spherical_harmonics(0, 0, theta, phi)
        assert isinstance(result, np.ndarray)
        assert result.shape == theta.shape

    def test_spherical_harmonics_positive_m(self):
        """Test spherical harmonics with positive m."""
        theta = np.array([np.pi / 2])
        phi = np.array([0.0])
        result = spherical_harmonics(1, 1, theta, phi)
        assert isinstance(result, np.ndarray)
        assert result.shape == theta.shape

    def test_spherical_harmonics_negative_m(self):
        """Test spherical harmonics with negative m."""
        theta = np.array([np.pi / 2])
        phi = np.array([0.0])
        result = spherical_harmonics(1, -1, theta, phi)
        assert isinstance(result, np.ndarray)
        assert result.shape == theta.shape

    def test_spherical_harmonics_array_input(self):
        """Test spherical harmonics with array inputs."""
        theta = np.linspace(0, np.pi, 10)
        phi = np.linspace(0, 2 * np.pi, 10)
        result = spherical_harmonics(1, 0, theta, phi)
        assert isinstance(result, np.ndarray)
        assert result.shape == theta.shape


class TestGenerateSMat:
    """Test class for S-matrix generation."""

    def test_generate_S_mat_basic(self):
        """Test basic S-matrix generation."""
        lm_set = [[0, 0], [1, 0]]
        num_t = 5
        num_p = 5
        result = generate_S_mat(lm_set, num_t, num_p)
        assert isinstance(result, np.ndarray)
        # Should have shape (num_t * num_p, len(lm_set))
        assert result.shape == (num_t * num_p, len(lm_set))

    def test_generate_S_mat_single_lm(self):
        """Test S-matrix generation with single (l, m) pair."""
        lm_set = [[0, 0]]
        num_t = 3
        num_p = 3
        result = generate_S_mat(lm_set, num_t, num_p)
        assert result.shape == (num_t * num_p, 1)

    def test_generate_S_mat_multiple_lm(self):
        """Test S-matrix generation with multiple (l, m) pairs."""
        lm_set = [[0, 0], [1, -1], [1, 0], [1, 1]]
        num_t = 4
        num_p = 4
        result = generate_S_mat(lm_set, num_t, num_p)
        assert result.shape == (num_t * num_p, len(lm_set))

    def test_import_fallback(self):
        """Test that import fallback works correctly."""
        # Test that the module imports successfully
        from seap.prediction import sph_harm as sph
        assert hasattr(sph, 'spherical_harmonics')
        assert hasattr(sph, 'quantum_number')
        assert hasattr(sph, 'generate_S_mat')
        assert hasattr(sph, 'orb_symbol')

    def test_import_fallback_mechanism(self):
        """Test that import fallback mechanism works."""
        # Test that the try-except import block is covered
        from seap.prediction import sph_harm as sph
        # The module should have _sph_harm imported (either sph_harm_y or sph_harm)
        # We can't directly test the import, but we can test that functions work
        result = sph.spherical_harmonics(0, 0, np.array([0.0]), np.array([0.0]))
        assert isinstance(result, np.ndarray)

