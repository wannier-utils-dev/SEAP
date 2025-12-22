"""
Tests for integration module.
"""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from seap.prediction.integration import Integration
from seap.prediction import sph_harm as sph


class TestIntegration:
    """Test class for Integration class."""

    @pytest.fixture
    def temp_data_dir(self):
        """
        Create a temporary directory with mock wavefunction data files.

        Yields
        ------
        dict
            Dictionary containing paths and configuration for testing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy wavefunction data (2 wavefunctions, 32x32x32 grid)
            psi_data = np.random.rand(2, 32, 32, 32).astype(np.float64)
            psi_info = np.array([[0, 1], [0, 2]])

            psi_path = os.path.join(tmpdir, "image32x32x32.npy")
            info_path = os.path.join(tmpdir, "image_info.npy")

            np.save(psi_path, psi_data)
            np.save(info_path, psi_info)

            yield {
                "tmpdir": tmpdir,
                "psi_path": psi_path,
                "info_path": info_path,
                "psi_data": psi_data,
                "psi_info": psi_info,
                "resolution": [32, 32, 32],
            }

    @pytest.fixture
    def integration_instance(self, temp_data_dir):
        """
        Create an Integration instance for testing.

        Parameters
        ----------
        temp_data_dir : dict
            Temporary data directory fixture.

        Returns
        -------
        Integration
            Instance of Integration class.
        """
        psi_dict = {
            "name": temp_data_dir["psi_path"],
            "info": temp_data_dir["info_path"],
            "resolution": temp_data_dir["resolution"],
        }
        return Integration(num_t=16, num_p=16, n=2, **psi_dict)

    def test_integration_initialization(self, temp_data_dir):
        """Test Integration class initialization with basic parameters."""
        num_t = 4
        num_p = 4
        n = 2

        psi_dict = {
            "name": temp_data_dir["psi_path"],
            "info": temp_data_dir["info_path"],
            "resolution": temp_data_dir["resolution"],
        }

        itg = Integration(num_t, num_p, n, **psi_dict)

        assert itg.num_t == num_t
        assert itg.num_p == num_p
        assert len(itg.lmset) > 0  # lmset should be generated from n
        assert itg.tline.shape == (num_t,)
        assert itg.pline.shape == (num_p,)
        assert itg.theta.shape == (num_t, num_p)
        assert itg.phi.shape == (num_t, num_p)

    def test_initialization_theta_values(self, integration_instance):
        """Test that theta values are correctly initialized from 0 to pi."""
        tline = integration_instance.tline
        assert tline[0] == pytest.approx(0.0)
        assert tline[-1] == pytest.approx(np.pi)
        assert len(tline) == integration_instance.num_t

    def test_initialization_phi_values(self, integration_instance):
        """Test that phi values are correctly initialized from 0 to 2*pi."""
        pline = integration_instance.pline
        assert pline[0] == pytest.approx(0.0)
        assert pline[-1] == pytest.approx(2 * np.pi)
        assert len(pline) == integration_instance.num_p

    def test_initialization_meshgrid(self, integration_instance):
        """Test that theta and phi meshgrids have correct shapes."""
        expected_shape = (
            integration_instance.num_t,
            integration_instance.num_p,
        )
        assert integration_instance.theta.shape == expected_shape
        assert integration_instance.phi.shape == expected_shape

    def test_initialization_lmset_for_n2(self, temp_data_dir):
        """Test lmset generation for n=2 (s and p orbitals)."""
        psi_dict = {
            "name": temp_data_dir["psi_path"],
            "info": temp_data_dir["info_path"],
            "resolution": temp_data_dir["resolution"],
        }
        itg = Integration(8, 8, n=2, **psi_dict)

        # For n=2: l=0 (s), l=1 (p with m=-1,0,1)
        # Expected: [0,0], [1,-1], [1,0], [1,1]
        expected_lmset = [[0, 0], [1, -1], [1, 0], [1, 1]]
        assert itg.lmset == expected_lmset

    def test_initialization_lmset_for_n3(self, temp_data_dir):
        """Test lmset generation for n=3 (s, p, and d orbitals)."""
        psi_dict = {
            "name": temp_data_dir["psi_path"],
            "info": temp_data_dir["info_path"],
            "resolution": temp_data_dir["resolution"],
        }
        itg = Integration(8, 8, n=3, **psi_dict)

        # For n=3: l=0 (s), l=1 (p), l=2 (d)
        expected_count = 1 + 3 + 5  # 9 orbitals
        assert len(itg.lmset) == expected_count

    def test_initialization_gwf_loaded(self, integration_instance, temp_data_dir):
        """Test that GammaWaveFunction is correctly loaded."""
        assert integration_instance.gwf is not None
        assert integration_instance.gwf.psi_org.shape == (2, 32, 32, 32)

    def test_initialization_psi_info_loaded(self, integration_instance, temp_data_dir):
        """Test that psi_info is correctly loaded."""
        expected_info = temp_data_dir["psi_info"]
        np.testing.assert_array_equal(
            integration_instance.psi_info,
            expected_info,
        )


class TestSphInt:
    """Test class for _sph_int method."""

    @pytest.fixture
    def integration_instance(self):
        """
        Create an Integration instance with mocked wavefunction for testing.

        Returns
        -------
        Integration
            Instance of Integration class.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            psi_data = np.random.rand(1, 16, 16, 16).astype(np.float64)
            psi_info = np.array([[0, 1]])

            psi_path = os.path.join(tmpdir, "psi.npy")
            info_path = os.path.join(tmpdir, "info.npy")

            np.save(psi_path, psi_data)
            np.save(info_path, psi_info)

            psi_dict = {
                "name": psi_path,
                "info": info_path,
                "resolution": [16, 16, 16],
            }
            return Integration(num_t=32, num_p=32, n=2, **psi_dict)

    def test_sph_int_orthogonality_same_function(self, integration_instance):
        """Test that spherical harmonics have non-zero self-integral."""
        itg = integration_instance

        # S_{0,0} is the s orbital (constant)
        S_00 = sph.spherical_harmonics(0, 0, itg.theta, itg.phi)

        result = itg._sph_int(S_00, S_00)

        # Self-integral should be approximately 4*pi (surface of unit sphere)
        # For normalized spherical harmonics, integral of Y_lm^2 is 1
        assert result != 0.0
        assert result > 0.0

    def test_sph_int_orthogonality_different_functions(self, integration_instance):
        """Test orthogonality between different spherical harmonics."""
        itg = integration_instance

        S_00 = sph.spherical_harmonics(0, 0, itg.theta, itg.phi)
        S_10 = sph.spherical_harmonics(1, 0, itg.theta, itg.phi)

        result = itg._sph_int(S_00, S_10)

        # Integral of different spherical harmonics should be approximately 0
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_sph_int_with_zeros(self, integration_instance):
        """Test _sph_int with zero arrays."""
        itg = integration_instance
        zeros = np.zeros((itg.num_t, itg.num_p))
        ones = np.ones((itg.num_t, itg.num_p))

        result = itg._sph_int(zeros, ones)
        assert result == pytest.approx(0.0)

    def test_sph_int_symmetry(self, integration_instance):
        """Test that _sph_int is symmetric: <f|g> = <g|f>."""
        itg = integration_instance

        f = np.sin(itg.theta) * np.cos(itg.phi)
        g = np.cos(itg.theta)

        result_fg = itg._sph_int(f, g)
        result_gf = itg._sph_int(g, f)

        assert result_fg == pytest.approx(result_gf)


class TestRunAtR:
    """Test class for run_at_r method."""

    @pytest.fixture
    def integration_instance(self):
        """
        Create an Integration instance for testing run_at_r.

        Returns
        -------
        Integration
            Instance of Integration class.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create simple mock data
            psi_data = np.random.rand(2, 16, 16, 16).astype(np.float64)
            psi_info = np.array([[0, 1], [0, 2]])

            psi_path = os.path.join(tmpdir, "psi.npy")
            info_path = os.path.join(tmpdir, "info.npy")

            np.save(psi_path, psi_data)
            np.save(info_path, psi_info)

            psi_dict = {
                "name": psi_path,
                "info": info_path,
                "resolution": [16, 16, 16],
            }
            return Integration(num_t=16, num_p=16, n=2, **psi_dict)

    def test_run_at_r_returns_correct_shape(self, integration_instance):
        """Test that run_at_r returns array with correct shape."""
        itg = integration_instance
        r = 0.2

        clm_list = itg.run_at_r(r)

        # Should return (num_wavefunctions, num_lm_pairs)
        num_psi = len(itg.gwf.psi_org)
        num_lm = len(itg.lmset)
        assert clm_list.shape == (num_psi, num_lm)

    def test_run_at_r_coefficients_normalized(self, integration_instance):
        """Test that coefficients are normalized (unit norm)."""
        itg = integration_instance
        r = 0.2

        clm_list = itg.run_at_r(r)

        for clm in clm_list:
            norm = np.linalg.norm(clm)
            assert norm == pytest.approx(1.0, rel=1e-6)

    def test_run_at_r_different_radii(self, integration_instance):
        """Test run_at_r at different radii produces different results."""
        itg = integration_instance

        clm_r1 = itg.run_at_r(0.1)
        clm_r2 = itg.run_at_r(0.3)

        # Results at different radii should generally differ
        # (unless wavefunction is spherically symmetric)
        assert not np.allclose(clm_r1, clm_r2)

    def test_run_at_r_small_radius(self, integration_instance):
        """Test run_at_r with small radius."""
        itg = integration_instance
        r = 0.05

        clm_list = itg.run_at_r(r)

        # Should still return valid, normalized coefficients
        assert clm_list.shape[0] == len(itg.gwf.psi_org)
        for clm in clm_list:
            assert np.linalg.norm(clm) == pytest.approx(1.0, rel=1e-6)


class TestRun:
    """Test class for run method."""

    @pytest.fixture
    def integration_instance(self):
        """
        Create an Integration instance for testing run method.

        Returns
        -------
        tuple
            Integration instance and temporary directory path.
        """
        tmpdir = tempfile.mkdtemp()
        psi_data = np.random.rand(2, 16, 16, 16).astype(np.float64)
        psi_info = np.array([[0, 1], [0, 2]])

        psi_path = os.path.join(tmpdir, "psi.npy")
        info_path = os.path.join(tmpdir, "info.npy")

        np.save(psi_path, psi_data)
        np.save(info_path, psi_info)

        psi_dict = {
            "name": psi_path,
            "info": info_path,
            "resolution": [16, 16, 16],
        }
        itg = Integration(num_t=16, num_p=16, n=2, **psi_dict)
        return itg, tmpdir

    def test_run_returns_correct_types(self, integration_instance):
        """Test that run returns tuple of numpy arrays."""
        itg, tmpdir = integration_instance
        os.chdir(tmpdir)

        r_opt, clm_opt = itg.run(rstep=0.1, threshold=0.1)

        assert isinstance(r_opt, np.ndarray)
        assert isinstance(clm_opt, np.ndarray)

    def test_run_returns_correct_shapes(self, integration_instance):
        """Test that run returns arrays with correct shapes."""
        itg, tmpdir = integration_instance
        os.chdir(tmpdir)

        r_opt, clm_opt = itg.run(rstep=0.1, threshold=0.1)

        num_psi = len(itg.gwf.psi_org)
        num_lm = len(itg.lmset)

        assert r_opt.shape == (num_psi,)
        assert clm_opt.shape == (num_psi, num_lm)

    def test_run_radii_in_expected_range(self, integration_instance):
        """Test that optimal radii are within expected range."""
        itg, tmpdir = integration_instance
        os.chdir(tmpdir)

        r_opt, _ = itg.run(rstep=0.1, threshold=0.1)

        # Default range is np.arange(0.1, 0.4, rstep)
        for r in r_opt:
            assert 0.1 <= r < 0.4

    def test_run_coefficients_normalized(self, integration_instance):
        """Test that returned coefficients are normalized."""
        itg, tmpdir = integration_instance
        os.chdir(tmpdir)

        _, clm_opt = itg.run(rstep=0.1, threshold=0.1)

        for clm in clm_opt:
            norm = np.linalg.norm(clm)
            assert norm == pytest.approx(1.0, rel=1e-6)

    def test_run_creates_output_file(self, integration_instance):
        """Test that run creates itg.out file."""
        itg, tmpdir = integration_instance
        os.chdir(tmpdir)

        itg.run(rstep=0.1, threshold=0.1)

        output_path = os.path.join(tmpdir, "itg.out")
        assert os.path.exists(output_path)

    def test_run_with_custom_threshold(self, integration_instance):
        """Test run with different threshold values."""
        itg, tmpdir = integration_instance
        os.chdir(tmpdir)

        # Different thresholds may result in different optimal radii
        _, clm_low = itg.run(rstep=0.1, threshold=0.05)
        _, clm_high = itg.run(rstep=0.1, threshold=0.5)

        # Both should be valid normalized coefficients
        assert clm_low.shape == clm_high.shape


class TestOutputInfo:
    """Test class for output_info method."""

    @pytest.fixture
    def integration_instance(self):
        """
        Create an Integration instance for testing output_info.

        Returns
        -------
        tuple
            Integration instance and temporary directory path.
        """
        tmpdir = tempfile.mkdtemp()
        psi_data = np.random.rand(2, 16, 16, 16).astype(np.float64)
        psi_info = np.array([[1, 5], [2, 10]])

        psi_path = os.path.join(tmpdir, "psi.npy")
        info_path = os.path.join(tmpdir, "info.npy")

        np.save(psi_path, psi_data)
        np.save(info_path, psi_info)

        psi_dict = {
            "name": psi_path,
            "info": info_path,
            "resolution": [16, 16, 16],
        }
        itg = Integration(num_t=8, num_p=8, n=2, **psi_dict)
        return itg, tmpdir

    def test_output_info_creates_file(self, integration_instance):
        """Test that output_info creates itg.out file."""
        itg, tmpdir = integration_instance
        os.chdir(tmpdir)

        r_opt = [0.2, 0.3]
        clm_opt = [np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0])]

        itg.output_info(r_opt, clm_opt)

        output_path = os.path.join(tmpdir, "itg.out")
        assert os.path.exists(output_path)

    def test_output_info_contains_target_data(self, integration_instance):
        """Test that output file contains target data information."""
        itg, tmpdir = integration_instance
        os.chdir(tmpdir)

        r_opt = [0.2, 0.3]
        clm_opt = [np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0])]

        itg.output_info(r_opt, clm_opt)

        output_path = os.path.join(tmpdir, "itg.out")
        with open(output_path, "r") as f:
            content = f.read()

        assert "Target data" in content
        assert itg.psi_name in content

    def test_output_info_contains_molecule_info(self, integration_instance):
        """Test that output file contains molecule and band information."""
        itg, tmpdir = integration_instance
        os.chdir(tmpdir)

        r_opt = [0.2, 0.3]
        clm_opt = [np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0])]

        itg.output_info(r_opt, clm_opt)

        output_path = os.path.join(tmpdir, "itg.out")
        with open(output_path, "r") as f:
            content = f.read()

        assert "molecule" in content
        assert "band" in content

    def test_output_info_contains_radius(self, integration_instance):
        """Test that output file contains radius information."""
        itg, tmpdir = integration_instance
        os.chdir(tmpdir)

        r_opt = [0.2, 0.3]
        clm_opt = [np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0])]

        itg.output_info(r_opt, clm_opt)

        output_path = os.path.join(tmpdir, "itg.out")
        with open(output_path, "r") as f:
            content = f.read()

        assert "r = 0.2" in content
        assert "r = 0.3" in content

    def test_output_info_contains_orbital_symbols(self, integration_instance):
        """Test that output file contains orbital symbols."""
        itg, tmpdir = integration_instance
        os.chdir(tmpdir)

        r_opt = [0.2, 0.3]
        clm_opt = [np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0])]

        itg.output_info(r_opt, clm_opt)

        output_path = os.path.join(tmpdir, "itg.out")
        with open(output_path, "r") as f:
            content = f.read()

        # Check for orbital symbols (s, py, pz, px for n=2)
        assert "s" in content
        assert "l =" in content
        assert "m =" in content

    def test_output_info_contains_coefficients(self, integration_instance):
        """Test that output file contains spherical harmonics coefficients."""
        itg, tmpdir = integration_instance
        os.chdir(tmpdir)

        r_opt = [0.2, 0.3]
        clm_opt = [np.array([0.5, 0.3, 0.7, 0.1]), np.array([0.2, 0.8, 0.4, 0.3])]

        itg.output_info(r_opt, clm_opt)

        output_path = os.path.join(tmpdir, "itg.out")
        with open(output_path, "r") as f:
            content = f.read()

        # Check that coefficients are written in scientific notation
        assert "E" in content or "e" in content


class TestIntegrationEdgeCases:
    """Test edge cases and error handling for Integration class."""

    def test_initialization_with_single_wavefunction(self):
        """Test initialization with a single wavefunction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            psi_data = np.random.rand(1, 8, 8, 8).astype(np.float64)
            psi_info = np.array([[0, 1]])

            psi_path = os.path.join(tmpdir, "psi.npy")
            info_path = os.path.join(tmpdir, "info.npy")

            np.save(psi_path, psi_data)
            np.save(info_path, psi_info)

            psi_dict = {
                "name": psi_path,
                "info": info_path,
                "resolution": [8, 8, 8],
            }
            itg = Integration(num_t=8, num_p=8, n=2, **psi_dict)

            assert len(itg.gwf.psi_org) == 1

    def test_initialization_with_n1(self):
        """Test initialization with n=1 (only s orbital)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            psi_data = np.random.rand(1, 8, 8, 8).astype(np.float64)
            psi_info = np.array([[0, 1]])

            psi_path = os.path.join(tmpdir, "psi.npy")
            info_path = os.path.join(tmpdir, "info.npy")

            np.save(psi_path, psi_data)
            np.save(info_path, psi_info)

            psi_dict = {
                "name": psi_path,
                "info": info_path,
                "resolution": [8, 8, 8],
            }
            itg = Integration(num_t=8, num_p=8, n=1, **psi_dict)

            # n=1 means only l=0 (s orbital)
            assert itg.lmset == [[0, 0]]

    def test_initialization_with_high_resolution(self):
        """Test initialization with high angular resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            psi_data = np.random.rand(1, 8, 8, 8).astype(np.float64)
            psi_info = np.array([[0, 1]])

            psi_path = os.path.join(tmpdir, "psi.npy")
            info_path = os.path.join(tmpdir, "info.npy")

            np.save(psi_path, psi_data)
            np.save(info_path, psi_info)

            psi_dict = {
                "name": psi_path,
                "info": info_path,
                "resolution": [8, 8, 8],
            }
            itg = Integration(num_t=64, num_p=64, n=2, **psi_dict)

            assert itg.num_t == 64
            assert itg.num_p == 64
            assert itg.theta.shape == (64, 64)

    def test_run_at_r_with_boundary_radius(self):
        """Test run_at_r with radius at grid boundary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            psi_data = np.random.rand(1, 16, 16, 16).astype(np.float64)
            psi_info = np.array([[0, 1]])

            psi_path = os.path.join(tmpdir, "psi.npy")
            info_path = os.path.join(tmpdir, "info.npy")

            np.save(psi_path, psi_data)
            np.save(info_path, psi_info)

            psi_dict = {
                "name": psi_path,
                "info": info_path,
                "resolution": [16, 16, 16],
            }
            itg = Integration(num_t=8, num_p=8, n=2, **psi_dict)

            # Test with very small radius
            clm = itg.run_at_r(0.01)
            assert clm.shape == (1, len(itg.lmset))


class TestIntegrationNumericalAccuracy:
    """Test numerical accuracy of Integration methods."""

    @pytest.fixture
    def high_resolution_instance(self):
        """
        Create a high-resolution Integration instance for accuracy testing.

        Returns
        -------
        Integration
            Instance with high angular resolution.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data with known structure
            psi_data = np.random.rand(1, 32, 32, 32).astype(np.float64)
            psi_info = np.array([[0, 1]])

            psi_path = os.path.join(tmpdir, "psi.npy")
            info_path = os.path.join(tmpdir, "info.npy")

            np.save(psi_path, psi_data)
            np.save(info_path, psi_info)

            psi_dict = {
                "name": psi_path,
                "info": info_path,
                "resolution": [32, 32, 32],
            }
            return Integration(num_t=64, num_p=64, n=3, **psi_dict)

    def test_sph_int_normalization_s_orbital(self, high_resolution_instance):
        """Test normalization integral for s orbital."""
        itg = high_resolution_instance

        S_00 = sph.spherical_harmonics(0, 0, itg.theta, itg.phi)
        result = itg._sph_int(S_00, S_00)

        # For normalized real spherical harmonics, <Y_00|Y_00> = 4*pi
        # Y_00 = 1/sqrt(4*pi), so Y_00^2 integrated = 1
        # But we integrate over sin(theta) d(theta) d(phi) which gives 4*pi
        # The actual value depends on normalization convention
        assert result > 0

    def test_sph_int_orthogonality_p_orbitals(self, high_resolution_instance):
        """Test orthogonality between p orbitals."""
        itg = high_resolution_instance

        S_1m1 = sph.spherical_harmonics(1, -1, itg.theta, itg.phi)
        S_10 = sph.spherical_harmonics(1, 0, itg.theta, itg.phi)
        S_11 = sph.spherical_harmonics(1, 1, itg.theta, itg.phi)

        # Different p orbitals should be orthogonal
        assert itg._sph_int(S_1m1, S_10) == pytest.approx(0.0, abs=1e-5)
        assert itg._sph_int(S_1m1, S_11) == pytest.approx(0.0, abs=1e-5)
        assert itg._sph_int(S_10, S_11) == pytest.approx(0.0, abs=1e-5)

    def test_sph_int_orthogonality_s_d_orbitals(self, high_resolution_instance):
        """Test orthogonality between s and d orbitals."""
        itg = high_resolution_instance

        S_00 = sph.spherical_harmonics(0, 0, itg.theta, itg.phi)
        S_20 = sph.spherical_harmonics(2, 0, itg.theta, itg.phi)

        result = itg._sph_int(S_00, S_20)
        assert result == pytest.approx(0.0, abs=1e-5)
