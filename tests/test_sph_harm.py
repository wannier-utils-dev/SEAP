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
        # The module should expose a working spherical_harmonics regardless
        # of which scipy backend (sph_harm or sph_harm_y) was imported by
        # the wrapper. We can't directly test the import, but we can test
        # that the function works.
        result = sph.spherical_harmonics(0, 0, np.array([0.0]), np.array([0.0]))
        assert isinstance(result, np.ndarray)


class TestSphericalHarmonicsValues:
    """Regression tests for the numerical values of the real spherical
    harmonics produced by spherical_harmonics().

    These tests guard against API-mapping regressions of the form that
    motivated the SciPy 1.15 fix: a silent rebinding of the underlying
    scipy function (sph_harm vs. sph_harm_y) without updating the
    argument order or the meaning of theta/phi caused every l > 0
    channel to return zero on the standard physics convention. The
    earlier shape/type tests in this file would not have caught that.

    Reference values use the standard real spherical harmonics
    convention,
        Y_{l, m>0} = sqrt(2) * (-1)^m * Re[Y_l^m(complex)]
        Y_{l, m<0} = sqrt(2) * (-1)^m * Im[Y_l^|m|(complex)]
        Y_{l, 0}   =                Re[Y_l^0(complex)]
    matching the wrapper in seap.prediction.sph_harm. Reference angles
    are chosen so that closed-form values from the standard tables
    apply directly.
    """

    TOL = 1e-10

    def test_Y00_constant(self):
        """Y_{0,0} = 1 / (2 sqrt(pi)) regardless of (theta, phi)."""
        expected = 1.0 / (2.0 * np.sqrt(np.pi))
        for theta, phi in [(0.0, 0.0), (np.pi / 3, np.pi / 4), (np.pi, 1.7)]:
            val = spherical_harmonics(0, 0, np.array([theta]), np.array([phi]))
            assert abs(val[0] - expected) < self.TOL

    def test_Y10_along_axes(self):
        """Y_{1,0}(theta) = sqrt(3 / (4 pi)) cos(theta)."""
        c = np.sqrt(3.0 / (4.0 * np.pi))
        # theta=0 -> +c; theta=pi -> -c; theta=pi/2 -> 0.
        v = spherical_harmonics(1, 0,
                                np.array([0.0, np.pi, np.pi / 2]),
                                np.array([0.0, 0.0, 0.0]))
        assert abs(v[0] - c) < self.TOL
        assert abs(v[1] + c) < self.TOL
        assert abs(v[2]) < self.TOL

    def test_Y1pm1_in_xy_plane(self):
        """At theta = pi/2:
            Y_{1, 1}(phi)  = sqrt(3 / (4 pi)) cos(phi)
            Y_{1,-1}(phi)  = sqrt(3 / (4 pi)) sin(phi)
        """
        c = np.sqrt(3.0 / (4.0 * np.pi))
        phi = np.array([0.0, np.pi / 2])
        theta = np.full_like(phi, np.pi / 2)
        v_p = spherical_harmonics(1, 1, theta, phi)
        v_m = spherical_harmonics(1, -1, theta, phi)
        # Y_{1, 1}: phi=0 -> +c; phi=pi/2 -> 0.
        assert abs(v_p[0] - c) < self.TOL
        assert abs(v_p[1]) < self.TOL
        # Y_{1,-1}: phi=0 -> 0; phi=pi/2 -> +c.
        assert abs(v_m[0]) < self.TOL
        assert abs(v_m[1] - c) < self.TOL

    def test_Y20_along_z(self):
        """Y_{2,0}(theta) = (1/4) sqrt(5 / pi) (3 cos^2(theta) - 1)."""
        coef = 0.25 * np.sqrt(5.0 / np.pi)
        v = spherical_harmonics(
            2, 0,
            np.array([0.0, np.pi / 2]),
            np.array([0.0, 0.0]))
        assert abs(v[0] - coef * 2.0) < self.TOL          # theta=0
        assert abs(v[1] - coef * (-1.0)) < self.TOL       # theta=pi/2

    def test_Y2pm2_in_xy_plane(self):
        """At theta = pi/2:
            Y_{2, 2}(phi) = (1/4) sqrt(15 / pi) cos(2 phi)
            Y_{2,-2}(phi) = (1/4) sqrt(15 / pi) sin(2 phi)
        """
        coef = 0.25 * np.sqrt(15.0 / np.pi)
        phi = np.array([0.0, np.pi / 4])
        theta = np.full_like(phi, np.pi / 2)
        v_p = spherical_harmonics(2, 2, theta, phi)
        v_m = spherical_harmonics(2, -2, theta, phi)
        # Y_{2, 2}: phi=0 -> +coef; phi=pi/4 -> 0.
        assert abs(v_p[0] - coef) < self.TOL
        assert abs(v_p[1]) < self.TOL
        # Y_{2,-2}: phi=0 -> 0; phi=pi/4 -> +coef.
        assert abs(v_m[0]) < self.TOL
        assert abs(v_m[1] - coef) < self.TOL

    def test_no_silent_zero_channel(self):
        """Every (l, m) channel for 0 <= l <= 3 must produce non-zero
        values at some angle. The SciPy 1.15 sph_harm_y API change
        previously caused all m != 0 channels to return zero
        identically; this regression test catches a recurrence."""
        rng = np.random.default_rng(seed=1234)
        theta = rng.uniform(0.05, np.pi - 0.05, size=128)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=128)
        for l in range(4):
            for m in range(-l, l + 1):
                vals = spherical_harmonics(l, m, theta, phi)
                assert isinstance(vals, np.ndarray)
                assert np.any(np.abs(vals) > 1e-6), (
                    f"All values are zero for (l, m) = ({l}, {m}); "
                    "spherical_harmonics wrapper has regressed.")

    def test_sum_rule_per_l(self):
        """Unsoeld's sum rule for real spherical harmonics:
            sum_{m = -l..l} Y_{l, m}(theta, phi)^2 = (2 l + 1) / (4 pi)
        for any (theta, phi). This is independent of the (m, n)
        argument order convention, so it is a strong cross-check on
        the wrapper across SciPy versions."""
        rng = np.random.default_rng(seed=5678)
        theta = rng.uniform(0.05, np.pi - 0.05, size=32)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=32)
        for l in range(4):
            total = np.zeros_like(theta)
            for m in range(-l, l + 1):
                total += spherical_harmonics(l, m, theta, phi) ** 2
            expected = (2 * l + 1) / (4.0 * np.pi)
            assert np.allclose(total, expected, atol=1e-10), (
                f"Sum rule violated for l = {l}: "
                f"got mean {total.mean():.6e}, expected {expected:.6e}.")

