"""
Tests for postppx module.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from seap.core.postppx import PostPPX, output_info, output_csv


class TestPostPPXInit:
    """Test class for PostPPX initialization and basic methods."""

    @pytest.fixture
    def mock_xsf_params(self):
        """Create mock XSF parameters."""
        return {
            'xsf_name': 'test.xsf',
            'prim_vec': [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            'num_atom': 2,
            'prim_coord': [
                ['H', '0.0', '0.0', '0.0'],
                ['H', '1.5', '1.5', '1.5'],
            ],
            'grid_info': [4, 4, 4],
            'pos_org': [0.0, 0.0, 0.0],
            'span_vec': [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            'grid_data': np.random.rand(64),
        }

    @pytest.fixture
    def mock_postppx(self, mock_xsf_params):
        """Create a PostPPX instance with mocked dependencies."""
        with patch('seap.core.postppx.xsf.get_params_from_xsf') as mock_get_params:
            mock_get_params.return_value = mock_xsf_params
            with patch(
                'seap.core.postppx.atoms2molecules'
            ) as mock_atoms2molecules:
                # Mock clustering results
                mock_atoms2molecules.return_value = (
                    [{0}, {1}],  # cluster_indices
                    np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),  # cluster_positions
                )
                ppx = PostPPX('test.xsf', bond_length=1.8)
        return ppx

    def test_init_basic(self, mock_xsf_params):
        """Test basic initialization of PostPPX."""
        with patch('seap.core.postppx.xsf.get_params_from_xsf') as mock_get_params:
            mock_get_params.return_value = mock_xsf_params
            with patch(
                'seap.core.postppx.atoms2molecules'
            ) as mock_atoms2molecules:
                mock_atoms2molecules.return_value = (
                    [{0}, {1}],
                    np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
                )
                ppx = PostPPX('test.xsf', bond_length=1.8)

        assert ppx._num_atom == 2
        np.testing.assert_array_equal(ppx._gridvecs, [4, 4, 4])
        np.testing.assert_array_equal(ppx._origin, [0.0, 0.0, 0.0])
        assert ppx._m1 == 3
        assert ppx._m2 == 3
        assert ppx._m3 == 3

    def test_init_custom_supercell(self, mock_xsf_params):
        """Test initialization with custom supercell multipliers."""
        with patch('seap.core.postppx.xsf.get_params_from_xsf') as mock_get_params:
            mock_get_params.return_value = mock_xsf_params
            with patch(
                'seap.core.postppx.atoms2molecules'
            ) as mock_atoms2molecules:
                mock_atoms2molecules.return_value = (
                    [{0}, {1}],
                    np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
                )
                ppx = PostPPX('test.xsf', bond_length=1.8, m1=5, m2=5, m3=5)

        assert ppx._m1 == 5
        assert ppx._m2 == 5
        assert ppx._m3 == 5

    def test_supervecs(self, mock_postppx):
        """Test _supervecs method."""
        result = mock_postppx._supervecs(2, 2, 2)
        expected = np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_supervecs_asymmetric(self, mock_postppx):
        """Test _supervecs with asymmetric multipliers."""
        result = mock_postppx._supervecs(1, 2, 3)
        expected = np.array([[3.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 9.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_element_symbols(self, mock_postppx):
        """Test _element_symbols method."""
        symbols = mock_postppx._element_symbols()
        np.testing.assert_array_equal(symbols, ['H', 'H'])

    def test_atomic_positions_cartesian(self, mock_postppx):
        """Test _atomic_positions in Cartesian coordinates."""
        pos = mock_postppx._atomic_positions(crys=False)
        expected = np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
        np.testing.assert_array_almost_equal(pos, expected)

    def test_atomic_positions_crystal(self, mock_postppx):
        """Test _atomic_positions in crystal coordinates."""
        pos = mock_postppx._atomic_positions(crys=True)
        # With 3x3x3 lattice, (1.5, 1.5, 1.5) -> (0.5, 0.5, 0.5)
        expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        np.testing.assert_array_almost_equal(pos, expected)

    def test_new_gridvecs(self, mock_postppx):
        """Test _new_gridvecs method."""
        result = mock_postppx._new_gridvecs(2, 2, 2)
        np.testing.assert_array_equal(result, [8, 8, 8])

    def test_new_gridvecs_asymmetric(self, mock_postppx):
        """Test _new_gridvecs with asymmetric multipliers."""
        result = mock_postppx._new_gridvecs(1, 2, 3)
        np.testing.assert_array_equal(result, [4, 8, 12])


class TestPostPPXDataMethods:
    """Test class for PostPPX data manipulation methods."""

    @pytest.fixture
    def mock_postppx(self):
        """Create a PostPPX instance with mocked dependencies."""
        mock_params = {
            'xsf_name': 'test.xsf',
            'prim_vec': [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            'num_atom': 1,
            'prim_coord': [['H', '0.0', '0.0', '0.0']],
            'grid_info': [2, 2, 2],
            'pos_org': [0.0, 0.0, 0.0],
            'span_vec': [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            'grid_data': np.arange(8, dtype=float),
        }
        with patch('seap.core.postppx.xsf.get_params_from_xsf') as mock_get_params:
            mock_get_params.return_value = mock_params
            with patch(
                'seap.core.postppx.atoms2molecules'
            ) as mock_atoms2molecules:
                mock_atoms2molecules.return_value = (
                    [{0}],
                    np.array([[0.0, 0.0, 0.0]]),
                )
                ppx = PostPPX('test.xsf', bond_length=1.8, m1=2, m2=2, m3=2)
        return ppx

    def test_data_in_primitive_no_pbc(self, mock_postppx):
        """Test _data_in_primitive without periodic boundary conditions."""
        data = np.arange(8, dtype=float)
        result = mock_postppx._data_in_primitive(data, pbc=False)

        assert result.shape == (2, 2, 2)
        assert np.all(np.isfinite(result))

    def test_data_in_primitive_with_pbc(self, mock_postppx):
        """Test _data_in_primitive with periodic boundary conditions."""
        data = np.arange(8, dtype=float)
        result = mock_postppx._data_in_primitive(data, pbc=True)

        assert result.shape == (3, 3, 3)
        assert np.all(np.isfinite(result))

    def test_data_in_super_no_pbc(self, mock_postppx):
        """Test _data_in_super without periodic boundary conditions."""
        data = np.arange(8, dtype=float)
        result = mock_postppx._data_in_super(data, 2, 2, 2, pbc=False)

        assert result.shape == (4, 4, 4)
        assert np.all(np.isfinite(result))

    def test_data_in_super_with_pbc(self, mock_postppx):
        """Test _data_in_super with periodic boundary conditions."""
        data = np.arange(8, dtype=float)
        result = mock_postppx._data_in_super(data, 2, 2, 2, pbc=True)

        assert result.shape == (5, 5, 5)
        assert np.all(np.isfinite(result))

    def test_interpolator_in_primitive(self, mock_postppx):
        """Test _interpolator_in_primitive method."""
        data = np.arange(8, dtype=float)
        interpolator = mock_postppx._interpolator_in_primitive(data)

        # Test interpolation at a point
        point = np.array([0.5, 0.5, 0.5])
        result = interpolator(point)
        assert np.isfinite(result)

    def test_interpolator_in_super(self, mock_postppx):
        """Test _interpolator_in_super method."""
        data = np.arange(8, dtype=float)
        interpolator = mock_postppx._interpolator_in_super(data, 2, 2, 2)

        # Test interpolation at a point
        point = np.array([0.5, 0.5, 0.5])
        result = interpolator(point)
        assert np.isfinite(result)


class TestPostPPXClustering:
    """Test class for PostPPX clustering functionality."""

    @pytest.fixture
    def mock_xsf_params(self):
        """Create mock XSF parameters."""
        return {
            'xsf_name': 'test.xsf',
            'prim_vec': [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            'num_atom': 4,
            'prim_coord': [
                ['C', '0.0', '0.0', '0.0'],
                ['C', '1.0', '0.0', '0.0'],  # Close to first C
                ['O', '4.0', '4.0', '4.0'],
                ['O', '4.5', '4.0', '4.0'],  # Close to first O
            ],
            'grid_info': [4, 4, 4],
            'pos_org': [0.0, 0.0, 0.0],
            'span_vec': [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            'grid_data': np.random.rand(64),
        }

    def test_clustering_called_with_bond_length(self, mock_xsf_params):
        """Test that clustering is called with correct bond length."""
        with patch('seap.core.postppx.xsf.get_params_from_xsf') as mock_get_params:
            mock_get_params.return_value = mock_xsf_params
            with patch(
                'seap.core.postppx.atoms2molecules'
            ) as mock_atoms2molecules:
                mock_atoms2molecules.return_value = (
                    [{0, 1}, {2, 3}],
                    np.array([[0.1, 0.0, 0.0], [0.85, 0.8, 0.8]]),
                )
                PostPPX('test.xsf', bond_length=2.0)

                # Check that atoms2molecules was called
                mock_atoms2molecules.assert_called_once()
                call_args = mock_atoms2molecules.call_args
                assert call_args[1]['threshold'] == 2.0


class TestOutputFunctions:
    """Test class for output utility functions."""

    def test_output_info(self):
        """Test output_info function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                kband = 1
                imol = 0
                rho_ratio = 0.5
                center = np.array([1.0, 2.0, 3.0])

                output_info(kband, imol, rho_ratio, center)

                # Check that file was created
                assert os.path.exists('output_ppin.out')

                # Check file content
                with open('output_ppin.out', 'r') as f:
                    content = f.read()

                assert 'band     : 1' in content
                assert 'molecure : 0' in content
                assert 'cart. coord.' in content
                assert 'rho/total =   0.500000' in content
            finally:
                os.chdir(original_cwd)

    def test_output_info_multiple_calls(self):
        """Test output_info appends to existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                output_info(1, 0, 0.3, np.array([0.0, 0.0, 0.0]))
                output_info(2, 1, 0.7, np.array([1.0, 1.0, 1.0]))

                with open('output_ppin.out', 'r') as f:
                    content = f.read()

                # Both entries should be present
                assert 'band     : 1' in content
                assert 'band     : 2' in content
            finally:
                os.chdir(original_cwd)

    def test_output_csv(self):
        """Test output_csv function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                calc_info = [
                    [0, '001', np.array([1.0, 2.0, 3.0])],
                    [1, '002', np.array([4.0, 5.0, 6.0])],
                ]

                output_csv(calc_info)

                # Check that file was created
                assert os.path.exists('center.csv')

                # Read and verify CSV content
                df = pd.read_csv('center.csv')

                assert len(df) == 2
                assert list(df.columns) == [
                    'band_index',
                    'molecule_index',
                    'center_x',
                    'center_y',
                    'center_z',
                ]
                assert df['band_index'].tolist() == [1, 2]
                assert df['molecule_index'].tolist() == [0, 1]
                assert df['center_x'].tolist() == [1.0, 4.0]
                assert df['center_y'].tolist() == [2.0, 5.0]
                assert df['center_z'].tolist() == [3.0, 6.0]
            finally:
                os.chdir(original_cwd)

    def test_output_csv_empty(self):
        """Test output_csv with empty data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                calc_info = []
                output_csv(calc_info)

                df = pd.read_csv('center.csv')
                assert len(df) == 0
            finally:
                os.chdir(original_cwd)


class TestPostPPXOutputCubeData:
    """Test class for PostPPX cube data output."""

    @pytest.fixture
    def mock_postppx(self):
        """Create a PostPPX instance with mocked dependencies."""
        mock_params = {
            'xsf_name': 'test.xsf',
            'prim_vec': [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            'num_atom': 1,
            'prim_coord': [['H', '0.0', '0.0', '0.0']],
            'grid_info': [4, 4, 4],
            'pos_org': [0.0, 0.0, 0.0],
            'span_vec': [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            'grid_data': np.random.rand(64),
        }
        with patch('seap.core.postppx.xsf.get_params_from_xsf') as mock_get_params:
            mock_get_params.return_value = mock_params
            with patch(
                'seap.core.postppx.atoms2molecules'
            ) as mock_atoms2molecules:
                mock_atoms2molecules.return_value = (
                    [{0}],
                    np.array([[0.0, 0.0, 0.0]]),
                )
                ppx = PostPPX('test.xsf', bond_length=1.8)
        return ppx

    def test_output_cube_data_xsf(self, mock_postppx):
        """Test _output_cube_data_xsf method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xsf_name = os.path.join(tmpdir, 'cube_test.xsf')
            gridvecs = np.array([4, 4, 4])
            origin = np.array([0.0, 0.0, 0.0])
            cube_length = 2.0
            data = np.random.rand(64)

            with patch('seap.core.postppx.xsf.output_xsf') as mock_output:
                mock_postppx._output_cube_data_xsf(
                    xsf_name, gridvecs, origin, cube_length, data
                )

                mock_output.assert_called_once()
                call_args = mock_output.call_args[0][0]

                assert call_args['xsf_name'] == xsf_name
                np.testing.assert_array_equal(call_args['grid_info'], gridvecs)
                np.testing.assert_array_equal(call_args['pos_org'], origin)
                np.testing.assert_array_equal(
                    call_args['span_vec'],
                    np.diag([cube_length, cube_length, cube_length]),
                )


class TestPostPPXFindVoids:
    """Test class for PostPPX find_voids functionality."""

    @pytest.fixture
    def mock_postppx_base(self):
        """Create base mock parameters for PostPPX."""
        return {
            'xsf_name': 'test.xsf',
            'prim_vec': [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            'num_atom': 2,
            'prim_coord': [
                ['Si', '0.0', '0.0', '0.0'],
                ['Si', '2.0', '2.0', '2.0'],
            ],
            'grid_info': [4, 4, 4],
            'pos_org': [0.0, 0.0, 0.0],
            'span_vec': [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            'grid_data': np.random.rand(64),
        }

    def test_find_voids_disabled(self, mock_postppx_base, capsys):
        """Test initialization without interstitial_length."""
        with patch('seap.core.postppx.xsf.get_params_from_xsf') as mock_get_params:
            mock_get_params.return_value = mock_postppx_base
            with patch(
                'seap.core.postppx.atoms2molecules'
            ) as mock_atoms2molecules:
                mock_atoms2molecules.return_value = (
                    [{0}, {1}],
                    np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
                )
                ppx = PostPPX('test.xsf', bond_length=1.8)

        captured = capsys.readouterr()
        assert 'not add voids' in captured.out
        assert ppx._void_centers == []

    def test_find_voids_exception_handling(self, mock_postppx_base, capsys):
        """Test find_voids handles exceptions gracefully."""
        with patch('seap.core.postppx.xsf.get_params_from_xsf') as mock_get_params:
            mock_get_params.return_value = mock_postppx_base
            with patch(
                'seap.core.postppx.atoms2molecules'
            ) as mock_atoms2molecules:
                mock_atoms2molecules.return_value = (
                    [{0}, {1}],
                    np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
                )
                with patch(
                    'seap.core.postppx.get_interstitial_sites'
                ) as mock_interstitial:
                    mock_interstitial.side_effect = Exception("Test error")
                    ppx = PostPPX(
                        'test.xsf', bond_length=1.8, interstitial_length=2.0
                    )

        captured = capsys.readouterr()
        assert 'Failed to find voids' in captured.err
        assert ppx._void_centers == []


class TestPostPPXGetCenters:
    """Test class for PostPPX get_centers method."""

    @pytest.fixture
    def mock_postppx(self):
        """Create a PostPPX instance with mocked dependencies."""
        mock_params = {
            'xsf_name': 'test.xsf',
            'prim_vec': [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            'num_atom': 2,
            'prim_coord': [
                ['H', '0.0', '0.0', '0.0'],
                ['H', '1.5', '1.5', '1.5'],
            ],
            'grid_info': [4, 4, 4],
            'pos_org': [0.0, 0.0, 0.0],
            'span_vec': [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            'grid_data': np.ones(64),  # Uniform density
        }
        with patch('seap.core.postppx.xsf.get_params_from_xsf') as mock_get_params:
            mock_get_params.return_value = mock_params
            with patch(
                'seap.core.postppx.atoms2molecules'
            ) as mock_atoms2molecules:
                mock_atoms2molecules.return_value = (
                    [{0}, {1}],
                    np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
                )
                ppx = PostPPX('test.xsf', bond_length=1.8, m1=2, m2=2, m3=2)
        return ppx

    def test_get_centers_returns_correct_shape(self, mock_postppx):
        """Test that get_centers returns arrays with correct shapes."""
        data = np.random.rand(64)
        centers, rho_ratios = mock_postppx.get_centers(data)

        # Should have 2 clusters
        assert centers.shape == (2, 3)
        assert rho_ratios.shape == (2,)

    def test_get_centers_ratios_sum_to_one(self, mock_postppx):
        """Test that rho_ratios approximately sum to 1."""
        data = np.random.rand(64) + 0.1  # Ensure positive values
        centers, rho_ratios = mock_postppx.get_centers(data)

        # Ratios should sum to approximately 1
        np.testing.assert_almost_equal(np.sum(rho_ratios), 1.0, decimal=5)

    def test_get_centers_finite_values(self, mock_postppx):
        """Test that get_centers returns finite values."""
        data = np.random.rand(64) + 0.1
        centers, rho_ratios = mock_postppx.get_centers(data)

        assert np.all(np.isfinite(centers))
        assert np.all(np.isfinite(rho_ratios))
        assert np.all(rho_ratios >= 0)


class TestPostPPXIntegration:
    """Integration tests for PostPPX class."""

    def test_full_workflow_with_mocks(self):
        """Test full workflow with mocked dependencies."""
        mock_params = {
            'xsf_name': 'test.xsf',
            'prim_vec': [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            'num_atom': 2,
            'prim_coord': [
                ['C', '0.0', '0.0', '0.0'],
                ['C', '2.0', '2.0', '2.0'],
            ],
            'grid_info': [4, 4, 4],
            'pos_org': [0.0, 0.0, 0.0],
            'span_vec': [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            'grid_data': np.random.rand(64),
        }

        with patch('seap.core.postppx.xsf.get_params_from_xsf') as mock_get_params:
            mock_get_params.return_value = mock_params
            with patch(
                'seap.core.postppx.atoms2molecules'
            ) as mock_atoms2molecules:
                mock_atoms2molecules.return_value = (
                    [{0}, {1}],
                    np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
                )

                # Initialize PostPPX
                ppx = PostPPX('test.xsf', bond_length=1.8, m1=2, m2=2, m3=2)

                # Test get_centers
                centers, rho_ratios = ppx.get_centers(mock_params['grid_data'])

                # Verify results
                assert len(centers) == 2
                assert len(rho_ratios) == 2
                assert np.sum(rho_ratios) <= 1.0 + 1e-5

                # Test atomic positions
                pos_cart = ppx._atomic_positions(crys=False)
                assert pos_cart.shape == (2, 3)

                # Test element symbols
                symbols = ppx._element_symbols()
                assert list(symbols) == ['C', 'C']


