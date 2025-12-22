"""
Tests for lasso module.
"""

import os
import tempfile

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from seap.prediction.lasso import LassoRegression


def create_dummy_psi_files(tmpdir, num_psi=2, resolution=32):
    """
    Create dummy wavefunction files for testing.

    Parameters
    ----------
    tmpdir : str
        Temporary directory path.
    num_psi : int, optional
        Number of wavefunctions, by default 2.
    resolution : int, optional
        Grid resolution, by default 32.

    Returns
    -------
    dict
        Dictionary containing paths and resolution for psi data.
    """
    psi_data = np.random.rand(num_psi, resolution, resolution, resolution)
    psi_info = np.array([[i, i] for i in range(num_psi)])

    psi_path = os.path.join(tmpdir, 'image32x32x32.npy')
    info_path = os.path.join(tmpdir, 'image_info.npy')

    np.save(psi_path, psi_data)
    np.save(info_path, psi_info)

    return {
        'name': psi_path,
        'info': info_path,
        'resolution': [resolution, resolution, resolution],
    }


class TestLassoRegression:
    """Test class for LassoRegression class."""

    def test_lasso_regression_initialization(self):
        """Test LassoRegression class initialization."""
        num_t = 4
        num_p = 4
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            assert lr.num_t == num_t
            assert lr.num_p == num_p
            assert len(lr.lmset) > 0
            assert lr.S_mat.shape[0] == num_t * num_p
            assert lr.S_mat.shape[1] == len(lr.lmset)

    def test_lasso_regression_initialization_with_different_n(self):
        """Test LassoRegression initialization with different quantum numbers."""
        num_t = 8
        num_p = 8

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir)

            # Test with n=1 (only s orbital)
            lr1 = LassoRegression(num_t, num_p, n=1, **psi_dict)
            assert len(lr1.lmset) == 1  # Only (0, 0)
            assert lr1.lmset == [[0, 0]]

            # Test with n=2 (s and p orbitals)
            lr2 = LassoRegression(num_t, num_p, n=2, **psi_dict)
            assert len(lr2.lmset) == 4  # (0,0), (1,-1), (1,0), (1,1)

            # Test with n=3 (s, p, and d orbitals)
            lr3 = LassoRegression(num_t, num_p, n=3, **psi_dict)
            assert len(lr3.lmset) == 9

    def test_s_mat_shape_consistency(self):
        """Test that S_mat shape is consistent with num_t, num_p, and lmset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir)

            for num_t in [4, 8, 16]:
                for num_p in [4, 8, 16]:
                    for n in [1, 2, 3]:
                        lr = LassoRegression(num_t, num_p, n, **psi_dict)
                        assert lr.S_mat.shape[0] == num_t * num_p
                        assert lr.S_mat.shape[1] == len(lr.lmset)

    def test_gwf_initialization(self):
        """Test that GammaWaveFunction is properly initialized."""
        num_t = 4
        num_p = 4
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=3)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            assert lr.gwf is not None
            assert lr.gwf.psi_org.shape[0] == 3

    def test_run_at_r_returns_correct_shape(self):
        """Test that run_at_r returns arrays with correct shapes."""
        num_t = 8
        num_p = 8
        n = 2
        num_psi = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=num_psi)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)
            clm_prime, opt_alpha = lr.run_at_r(r=0.2, num_split=2)

            # Check shapes
            assert clm_prime.shape[0] == num_psi
            assert clm_prime.shape[1] == len(lr.lmset)
            assert opt_alpha.shape[0] == num_psi

    def test_run_at_r_coefficients_normalized(self):
        """Test that run_at_r returns normalized coefficients."""
        num_t = 8
        num_p = 8
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=2)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)
            clm_prime, _ = lr.run_at_r(r=0.2, num_split=2)

            # Check that coefficients are normalized (norm should be 1 or 0)
            for coef in clm_prime:
                norm = np.linalg.norm(coef)
                # Norm should be close to 1 (normalized) or could be NaN
                # if all coefficients were zero
                assert norm == pytest.approx(1.0, rel=1e-5) or np.isnan(norm)

    def test_run_at_r_with_different_radii(self):
        """Test run_at_r with different radius values."""
        num_t = 8
        num_p = 8
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=2)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            for r in [0.1, 0.2, 0.3]:
                clm_prime, opt_alpha = lr.run_at_r(r=r, num_split=2)
                assert clm_prime is not None
                assert opt_alpha is not None

    def test_decide_param_returns_positive_alpha(self):
        """Test that _decide_param returns a positive alpha value."""
        num_t = 8
        num_p = 8
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=2)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            # Create dummy Q data
            Q = np.random.rand(num_t * num_p)

            alpha = lr._decide_param(Q, num_split=2)

            assert alpha > 0
            assert isinstance(alpha, float)

    def test_decide_param_with_different_num_split(self):
        """Test _decide_param with different number of splits."""
        num_t = 16
        num_p = 16
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=2)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            Q = np.random.rand(num_t * num_p)

            for num_split in [2, 3, 4]:
                alpha = lr._decide_param(Q, num_split=num_split)
                assert alpha > 0

    def test_run_method_returns_correct_structure(self):
        """Test that run method returns correct structure."""
        num_t = 8
        num_p = 8
        n = 2
        num_psi = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=num_psi)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            # Change to tmpdir to avoid creating files in the test directory
            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                r_opt, alpha_opt, clm_opt = lr.run(rstep=0.1, num_split=2)

                # Check that all results have correct length
                assert len(r_opt) == num_psi
                assert len(alpha_opt) == num_psi
                assert len(clm_opt) == num_psi

                # Check that r_opt values are within expected range
                for r in r_opt:
                    assert 0.1 <= r <= 0.4

                # Check that alpha_opt values are positive
                for alpha in alpha_opt:
                    assert alpha > 0

                # Check that clm_opt arrays have correct shape
                for clm in clm_opt:
                    assert len(clm) == len(lr.lmset)
            finally:
                os.chdir(original_cwd)

    def test_run_method_creates_output_file(self):
        """Test that run method creates output file."""
        num_t = 8
        num_p = 8
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=2)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                lr.run(rstep=0.1, num_split=2)

                # Check that output file was created
                output_file = os.path.join(tmpdir, 'lasso.out')
                assert os.path.exists(output_file)

                # Check that output file has content
                with open(output_file, 'r') as f:
                    content = f.read()
                    assert 'Target data' in content
                    assert 'molecule' in content
                    assert 'band' in content
            finally:
                os.chdir(original_cwd)

    def test_output_info_format(self):
        """Test that _output_info generates correctly formatted output."""
        num_t = 8
        num_p = 8
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=2)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                r_opt = [0.1, 0.2]
                alpha_opt = [0.01, 0.02]
                clm_opt = [np.ones(len(lr.lmset)), np.ones(len(lr.lmset)) * 0.5]

                lr._output_info(r_opt, clm_opt, alpha_opt)

                with open('lasso.out', 'r') as f:
                    content = f.read()

                # Check format elements
                assert 'r =' in content
                assert 'alpha =' in content
                assert 'predicted coefficients' in content
                assert 'l =' in content
                assert 'm =' in content
            finally:
                os.chdir(original_cwd)

    def test_run_with_single_wavefunction(self):
        """Test run method with single wavefunction."""
        num_t = 8
        num_p = 8
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=1)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                r_opt, alpha_opt, clm_opt = lr.run(rstep=0.15, num_split=2)

                assert len(r_opt) == 1
                assert len(alpha_opt) == 1
                assert len(clm_opt) == 1
            finally:
                os.chdir(original_cwd)

    def test_run_with_different_rstep(self):
        """Test run method with different rstep values."""
        num_t = 8
        num_p = 8
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=2)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                # Test with different rstep values
                for rstep in [0.05, 0.1, 0.15]:
                    r_opt, alpha_opt, clm_opt = lr.run(rstep=rstep, num_split=2)
                    assert len(r_opt) == 2
            finally:
                os.chdir(original_cwd)


class TestLassoRegressionEdgeCases:
    """Test class for edge cases in LassoRegression."""

    def test_small_grid_resolution(self):
        """Test with small grid resolution."""
        num_t = 4
        num_p = 4
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=1, resolution=16)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)
            assert lr.S_mat.shape[0] == num_t * num_p

    def test_large_quantum_number(self):
        """Test with larger quantum number n."""
        num_t = 16
        num_p = 16
        n = 4

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=1)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            # n=4 should give l=0,1,2,3 -> 1+3+5+7 = 16 orbitals
            expected_lmset_len = sum(2 * l + 1 for l in range(n))
            assert len(lr.lmset) == expected_lmset_len

    def test_asymmetric_theta_phi_grid(self):
        """Test with asymmetric theta and phi grid."""
        num_t = 8
        num_p = 16
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=2)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)
            assert lr.S_mat.shape[0] == num_t * num_p

    def test_psi_info_loaded_correctly(self):
        """Test that psi_info is loaded correctly."""
        num_t = 4
        num_p = 4
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create custom psi_info
            num_psi = 3
            psi_data = np.random.rand(num_psi, 32, 32, 32)
            psi_info = np.array([[10, 20], [30, 40], [50, 60]])

            psi_path = os.path.join(tmpdir, 'psi.npy')
            info_path = os.path.join(tmpdir, 'info.npy')

            np.save(psi_path, psi_data)
            np.save(info_path, psi_info)

            psi_dict = {
                'name': psi_path,
                'info': info_path,
                'resolution': [32, 32, 32],
            }

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            np.testing.assert_array_equal(lr.psi_info, psi_info)


class TestLassoRegressionIntegration:
    """Integration tests for LassoRegression."""

    def test_full_workflow(self):
        """Test the complete workflow of LassoRegression."""
        num_t = 8
        num_p = 8
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=2)

            # Initialize
            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            # Run at specific radius
            clm_at_r, alpha_at_r = lr.run_at_r(r=0.2, num_split=2)

            # Verify results
            assert clm_at_r.shape[0] == 2
            assert len(alpha_at_r) == 2

            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                # Run full optimization
                r_opt, alpha_opt, clm_opt = lr.run(rstep=0.1, num_split=2)

                # Verify final results
                assert len(r_opt) == 2
                assert len(alpha_opt) == 2
                assert len(clm_opt) == 2

                # Verify output file exists
                assert os.path.exists('lasso.out')
            finally:
                os.chdir(original_cwd)

    def test_reproducibility_with_same_seed(self):
        """Test that results are reproducible with the same random seed."""
        num_t = 8
        num_p = 8
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data with fixed seed
            np.random.seed(42)
            psi_data = np.random.rand(2, 32, 32, 32)
            psi_info = np.array([[0, 0], [1, 1]])

            psi_path = os.path.join(tmpdir, 'psi.npy')
            info_path = os.path.join(tmpdir, 'info.npy')

            np.save(psi_path, psi_data)
            np.save(info_path, psi_info)

            psi_dict = {
                'name': psi_path,
                'info': info_path,
                'resolution': [32, 32, 32],
            }

            lr1 = LassoRegression(num_t, num_p, n, **psi_dict)
            lr2 = LassoRegression(num_t, num_p, n, **psi_dict)

            # S_mat should be identical
            np.testing.assert_array_almost_equal(lr1.S_mat, lr2.S_mat)


class TestLassoRegressionWithMock:
    """Test LassoRegression with mocked dependencies."""

    def test_lasso_cv_called_with_correct_params(self):
        """Test that LassoCV is called with correct parameters."""
        num_t = 8
        num_p = 8
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=1)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            with patch('seap.prediction.lasso.LassoCV') as mock_lasso_cv:
                mock_clf = MagicMock()
                mock_clf.alpha_ = 0.01
                mock_lasso_cv.return_value = mock_clf

                Q = np.random.rand(num_t * num_p)
                result = lr._decide_param(Q, num_split=3)

                # Verify LassoCV was called
                mock_lasso_cv.assert_called_once()
                call_kwargs = mock_lasso_cv.call_args[1]
                assert call_kwargs['cv'] == 3
                assert 'alphas' in call_kwargs
                assert result == 0.01

    def test_lasso_called_with_correct_alpha(self):
        """Test that Lasso is called with the optimal alpha."""
        num_t = 8
        num_p = 8
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=1)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            with patch('seap.prediction.lasso.Lasso') as mock_lasso:
                mock_clf = MagicMock()
                mock_clf.coef_ = np.ones(len(lr.lmset))
                mock_lasso.return_value = mock_clf

                with patch.object(lr, '_decide_param', return_value=0.05):
                    lr.run_at_r(r=0.2, num_split=2)

                    # Verify Lasso was called with optimal alpha
                    mock_lasso.assert_called_with(alpha=0.05, max_iter=10000)


class TestLassoRegressionParameterValidation:
    """Test parameter validation for LassoRegression."""

    def test_various_num_split_values(self):
        """Test run_at_r with various num_split values."""
        num_t = 16
        num_p = 16
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=1)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            # Test with different num_split values
            for num_split in [2, 3, 5]:
                clm, alpha = lr.run_at_r(r=0.2, num_split=num_split)
                assert clm is not None
                assert alpha is not None

    def test_various_r_values_in_range(self):
        """Test run_at_r with various r values within valid range."""
        num_t = 8
        num_p = 8
        n = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            psi_dict = create_dummy_psi_files(tmpdir, num_psi=1)

            lr = LassoRegression(num_t, num_p, n, **psi_dict)

            # Test with various r values (must be within interpolation range)
            for r in [0.1, 0.15, 0.2, 0.25, 0.3]:
                clm, alpha = lr.run_at_r(r=r, num_split=2)
                assert clm.shape[0] == 1
                assert len(alpha) == 1
