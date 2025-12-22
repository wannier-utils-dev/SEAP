"""
Tests for xsf module.
"""

import os
import tempfile

import numpy as np

from seap.common.xsf import (
    get_data_for_new_grid,
    get_data_for_new_grid_npbc,
    get_params_from_xsf,
    get_values,
    get_values_new,
    grid2pos,
    output_xsf,
)


class TestGetParamsFromXsf:
    """Test class for get_params_from_xsf function."""

    def test_get_params_from_xsf_basic(self, capsys):
        """Test basic XSF file reading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xsf_file = os.path.join(tmpdir, 'test.xsf')

            # Create a minimal XSF file
            xsf_content = """CRYSTAL
PRIMVEC
   1.000000000   0.000000000   0.000000000
   0.000000000   2.000000000   0.000000000
   0.000000000   0.000000000   3.000000000
PRIMCOORD
    2  1
 H    0.0    0.0    0.0
 O    0.5    0.5    0.5
BEGIN_BLOCK_DATAGRID_3D
3D_PWSCF
DATAGRID_3D_UNKNOWN
    2    2    2
  0.000000  0.000000  0.000000
  1.000000  0.000000  0.000000
  0.000000  2.000000  0.000000
  0.000000  0.000000  3.000000
  1.0  2.0
  3.0  4.0

  5.0  6.0
  7.0  8.0

END_DATAGRID_3D
END_BLOCK_DATAGRID_3D
"""
            with open(xsf_file, 'w') as f:
                f.write(xsf_content)

            params = get_params_from_xsf(xsf_file)

            # Verify parameters
            assert params['xsf_name'] == xsf_file
            assert params['num_atom'] == 2
            assert params['grid_info'] == [2, 2, 2]
            assert params['pos_org'] == [0.0, 0.0, 0.0]

            # Check primitive vectors
            assert params['prim_vec'][0] == [1.0, 0.0, 0.0]
            assert params['prim_vec'][1] == [0.0, 2.0, 0.0]
            assert params['prim_vec'][2] == [0.0, 0.0, 3.0]

            # Check spanning vectors
            assert params['span_vec'][0] == [1.0, 0.0, 0.0]
            assert params['span_vec'][1] == [0.0, 2.0, 0.0]
            assert params['span_vec'][2] == [0.0, 0.0, 3.0]

            # Check atomic coordinates
            assert params['prim_coord'][0] == ['H', '0.0', '0.0', '0.0']
            assert params['prim_coord'][1] == ['O', '0.5', '0.5', '0.5']

            # Check grid data
            expected_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            np.testing.assert_array_almost_equal(
                params['grid_data'], expected_data
            )

            # Check that file name is printed
            captured = capsys.readouterr()
            assert f'XSF file name = {xsf_file}' in captured.out

    def test_get_params_from_xsf_single_atom(self, capsys):
        """Test XSF file reading with single atom."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xsf_file = os.path.join(tmpdir, 'single_atom.xsf')

            xsf_content = """CRYSTAL
PRIMVEC
   5.000000000   0.000000000   0.000000000
   0.000000000   5.000000000   0.000000000
   0.000000000   0.000000000   5.000000000
PRIMCOORD
    1  1
 C    1.0    2.0    3.0
BEGIN_BLOCK_DATAGRID_3D
3D_PWSCF
DATAGRID_3D_UNKNOWN
    2    2    2
  0.100000  0.200000  0.300000
  5.000000  0.000000  0.000000
  0.000000  5.000000  0.000000
  0.000000  0.000000  5.000000
  0.1  0.2
  0.3  0.4

  0.5  0.6
  0.7  0.8

END_DATAGRID_3D
END_BLOCK_DATAGRID_3D
"""
            with open(xsf_file, 'w') as f:
                f.write(xsf_content)

            params = get_params_from_xsf(xsf_file)

            assert params['num_atom'] == 1
            assert params['prim_coord'][0] == ['C', '1.0', '2.0', '3.0']
            assert params['pos_org'] == [0.1, 0.2, 0.3]


class TestOutputXsf:
    """Test class for output_xsf function."""

    def test_output_xsf_basic(self):
        """Test basic XSF file writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xsf_file = os.path.join(tmpdir, 'output.xsf')

            params = {
                'xsf_name': xsf_file,
                'prim_vec': [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
                'num_atom': 2,
                'prim_coord': [
                    ['H', '0.0', '0.0', '0.0'],
                    ['O', '0.5', '0.5', '0.5'],
                ],
                'grid_info': [2, 2, 2],
                'pos_org': [0.0, 0.0, 0.0],
                'span_vec': [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
                'grid_data': np.array(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                ),
            }

            output_xsf(params)

            # Check file was created
            assert os.path.exists(xsf_file)

            # Read and verify content
            with open(xsf_file, 'r') as f:
                content = f.read()

            assert 'CRYSTAL' in content
            assert 'PRIMVEC' in content
            assert 'PRIMCOORD' in content
            assert 'BEGIN_BLOCK_DATAGRID_3D' in content
            assert 'END_DATAGRID_3D' in content
            assert 'END_BLOCK_DATAGRID_3D' in content
            assert '3D_PWSCF' in content
            assert 'DATAGRID_3D_UNKNOWN' in content

    def test_output_xsf_round_trip(self, capsys):
        """Test reading back written XSF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            xsf_file = os.path.join(tmpdir, 'round_trip.xsf')

            original_params = {
                'xsf_name': xsf_file,
                'prim_vec': [
                    [2.5, 0.0, 0.0],
                    [0.0, 3.5, 0.0],
                    [0.0, 0.0, 4.5],
                ],
                'num_atom': 1,
                'prim_coord': [['Fe', '1.25', '1.75', '2.25']],
                'grid_info': [2, 2, 2],
                'pos_org': [0.1, 0.2, 0.3],
                'span_vec': [
                    [2.5, 0.0, 0.0],
                    [0.0, 3.5, 0.0],
                    [0.0, 0.0, 4.5],
                ],
                'grid_data': np.array(
                    [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
                ),
            }

            output_xsf(original_params)
            read_params = get_params_from_xsf(xsf_file)

            # Verify key parameters match
            assert read_params['num_atom'] == original_params['num_atom']
            assert read_params['grid_info'] == original_params['grid_info']

            # Check primitive vectors (with tolerance for formatting)
            for i in range(3):
                np.testing.assert_array_almost_equal(
                    read_params['prim_vec'][i],
                    original_params['prim_vec'][i],
                    decimal=5,
                )

            # Check grid data (with tolerance)
            np.testing.assert_array_almost_equal(
                read_params['grid_data'],
                original_params['grid_data'],
                decimal=5,
            )


class TestGetDataForNewGrid:
    """Test class for get_data_for_new_grid function."""

    def test_same_grid_size(self):
        """Test interpolation with same grid size."""
        grid_info = [4, 4, 4]
        data = np.arange(64, dtype=float)
        grid_info_new = [4, 4, 4]

        result = get_data_for_new_grid(data, grid_info, grid_info_new)

        assert result.shape == (4, 4, 4)
        # Check that result contains finite values
        assert np.all(np.isfinite(result))

    def test_downsampling(self):
        """Test downsampling to smaller grid."""
        grid_info = [4, 4, 4]
        data = np.random.rand(64)
        grid_info_new = [2, 2, 2]

        result = get_data_for_new_grid(data, grid_info, grid_info_new)

        assert result.shape == (2, 2, 2)
        assert np.all(np.isfinite(result))

    def test_output_shape(self):
        """Test that output shape matches grid_info_new."""
        grid_info = [4, 4, 4]
        data = np.random.rand(64)
        grid_info_new = [3, 3, 3]

        result = get_data_for_new_grid(data, grid_info, grid_info_new)

        assert result.shape == (3, 3, 3)
        assert np.all(np.isfinite(result))


class TestGetDataForNewGridNpbc:
    """Test class for get_data_for_new_grid_npbc function."""

    def test_basic_interpolation(self):
        """Test basic interpolation without periodic boundary conditions."""
        grid_info = [4, 4, 4]
        data = np.arange(64, dtype=float)
        grid_info_new = [4, 4, 4]

        result = get_data_for_new_grid_npbc(data, grid_info, grid_info_new)

        assert result.shape == (4, 4, 4)
        assert np.all(np.isfinite(result))

    def test_upsampling_npbc(self):
        """Test upsampling without periodic boundary conditions."""
        grid_info = [2, 2, 2]
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        grid_info_new = [4, 4, 4]

        result = get_data_for_new_grid_npbc(data, grid_info, grid_info_new)

        assert result.shape == (4, 4, 4)
        assert np.all(np.isfinite(result))

    def test_no_shift_grid(self):
        """Test interpolation without grid shift."""
        grid_info = [4, 4, 4]
        data = np.arange(64, dtype=float)
        grid_info_new = [4, 4, 4]

        result = get_data_for_new_grid_npbc(
            data, grid_info, grid_info_new, shift_grid=False
        )

        assert result.shape == (4, 4, 4)
        assert np.all(np.isfinite(result))


class TestGrid2Pos:
    """Test class for grid2pos function."""

    def test_origin_conversion(self):
        """Test conversion of origin point."""
        grid_coords = np.array([0, 0, 0])
        grid_vec = np.array([10, 10, 10])
        spanning_vec = np.array([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]])
        rorg = np.array([0.0, 0.0, 0.0])

        result = grid2pos(grid_coords, grid_vec, spanning_vec, rorg)

        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_with_origin_offset(self):
        """Test conversion with origin offset."""
        grid_coords = np.array([0, 0, 0])
        grid_vec = np.array([10, 10, 10])
        spanning_vec = np.array([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]])
        rorg = np.array([1.0, 2.0, 3.0])

        result = grid2pos(grid_coords, grid_vec, spanning_vec, rorg)

        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_grid_point_conversion(self):
        """Test conversion of non-origin grid point."""
        grid_coords = np.array([1, 1, 1])
        grid_vec = np.array([2, 2, 2])
        spanning_vec = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        rorg = np.array([0.0, 0.0, 0.0])

        result = grid2pos(grid_coords, grid_vec, spanning_vec, rorg)

        # With grid_vec = [2, 2, 2], dspanning_vec = spanning_vec / [1, 1, 1] = spanning_vec
        # result = dot([1, 1, 1], [[2,0,0],[0,2,0],[0,0,2]]) + [0,0,0] = [2, 2, 2]
        np.testing.assert_array_almost_equal(result, [2.0, 2.0, 2.0])

    def test_multiple_grid_points(self):
        """Test conversion of multiple grid points."""
        grid_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        grid_vec = np.array([3, 3, 3])
        spanning_vec = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
        rorg = np.array([0.0, 0.0, 0.0])

        result = grid2pos(grid_coords, grid_vec, spanning_vec, rorg)

        # dspanning_vec = [[4,0,0],[0,4,0],[0,0,4]] / [2, 2, 2] = [[2,0,0],[0,2,0],[0,0,2]]
        expected = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ])
        np.testing.assert_array_almost_equal(result, expected)


class TestGetValues:
    """Test class for get_values function."""

    def test_basic_interpolation(self):
        """Test basic value interpolation."""
        grid_vec = np.array([2, 2, 2])
        data = np.arange(8, dtype=float).reshape(2, 2, 2)

        # Create coordinate array
        coords = np.zeros((2, 2, 2, 3))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    coords[i, j, k] = [i, j, k]

        result = get_values(coords, grid_vec, data)

        assert result.shape == (2, 2, 2)
        assert np.all(np.isfinite(result))

    def test_interpolation_shape(self):
        """Test that output shape matches input grid."""
        grid_vec = np.array([4, 4, 4])
        data = np.random.rand(4, 4, 4)

        coords = np.zeros((4, 4, 4, 3))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    coords[i, j, k] = [i * 0.25, j * 0.25, k * 0.25]

        result = get_values(coords, grid_vec, data)

        assert result.shape == (4, 4, 4)


class TestGetValuesNew:
    """Test class for get_values_new function."""

    def test_single_point(self):
        """Test interpolation at a single point."""
        grid_vec = np.array([2, 2, 2])
        data = np.arange(8, dtype=float)
        points = np.array([[0.0, 0.0, 0.0]])

        result = get_values_new(points, grid_vec, data)

        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_multiple_points(self):
        """Test interpolation at multiple points."""
        grid_vec = np.array([2, 2, 2])
        data = np.arange(8, dtype=float)
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
        ])

        result = get_values_new(points, grid_vec, data)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_corner_points(self):
        """Test interpolation at grid corner points."""
        grid_vec = np.array([2, 2, 2])
        # Create simple data where each corner has distinct value
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        points = np.array([
            [0.0, 0.0, 0.0],
        ])

        result = get_values_new(points, grid_vec, data)

        # At origin, should get value close to first data point
        assert result.shape == (1,)
        np.testing.assert_almost_equal(result[0], 0.0, decimal=5)


class TestIntegration:
    """Integration tests for xsf module."""

    def test_read_modify_write_cycle(self, capsys):
        """Test complete read-modify-write cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create original XSF file
            original_file = os.path.join(tmpdir, 'original.xsf')
            xsf_content = """CRYSTAL
PRIMVEC
   3.000000000   0.000000000   0.000000000
   0.000000000   3.000000000   0.000000000
   0.000000000   0.000000000   3.000000000
PRIMCOORD
    1  1
 Si    1.5    1.5    1.5
BEGIN_BLOCK_DATAGRID_3D
3D_PWSCF
DATAGRID_3D_UNKNOWN
    2    2    2
  0.000000  0.000000  0.000000
  3.000000  0.000000  0.000000
  0.000000  3.000000  0.000000
  0.000000  0.000000  3.000000
  1.0  2.0
  3.0  4.0

  5.0  6.0
  7.0  8.0

END_DATAGRID_3D
END_BLOCK_DATAGRID_3D
"""
            with open(original_file, 'w') as f:
                f.write(xsf_content)

            # Read original
            params = get_params_from_xsf(original_file)

            # Modify grid data
            params['grid_data'] = params['grid_data'] * 2

            # Write to new file
            modified_file = os.path.join(tmpdir, 'modified.xsf')
            params['xsf_name'] = modified_file
            output_xsf(params)

            # Read back modified file
            modified_params = get_params_from_xsf(modified_file)

            # Verify modification
            expected_data = np.array(
                [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
            )
            np.testing.assert_array_almost_equal(
                modified_params['grid_data'], expected_data, decimal=5
            )

    def test_grid_interpolation_downsampling(self):
        """Test that downsampling produces valid interpolated values."""
        grid_info = [8, 8, 8]
        data = np.random.rand(512) * 100  # Random data in [0, 100]
        grid_info_new = [4, 4, 4]

        result = get_data_for_new_grid(data, grid_info, grid_info_new)

        # Check output shape and values
        assert result.shape == (4, 4, 4)
        assert np.all(np.isfinite(result))

