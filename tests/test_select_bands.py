"""
Tests for select_bands module.
"""

import os
import tempfile
import numpy as np

from seap.core.select_bands import (
    count_cmol,
    output_mol_info,
    get_data_from_pwout,
    find_nearest_bands,
    get_initial_ppin,
    write_band_dat,
)


class TestCountCmol:
    """Test class for count_cmol function."""

    def test_count_cmol_single_carbon_molecule(self):
        """Test counting a single carbon-containing molecule."""
        elements = ['C', 'H', 'H', 'H', 'H']
        molecules = [{0, 1, 2, 3, 4}]
        
        result = count_cmol(elements, molecules)
        
        assert result == 1

    def test_count_cmol_multiple_carbon_molecules(self):
        """Test counting multiple carbon-containing molecules."""
        elements = ['C', 'H', 'C', 'H', 'O', 'H']
        molecules = [{0, 1}, {2, 3}, {4, 5}]
        
        result = count_cmol(elements, molecules)
        
        assert result == 2

    def test_count_cmol_no_carbon(self):
        """Test counting molecules with no carbon."""
        elements = ['O', 'H', 'H', 'N', 'H', 'H', 'H']
        molecules = [{0, 1, 2}, {3, 4, 5, 6}]
        
        result = count_cmol(elements, molecules)
        
        assert result == 0

    def test_count_cmol_empty_molecules(self):
        """Test with empty molecules list."""
        elements = []
        molecules = []
        
        result = count_cmol(elements, molecules)
        
        assert result == 0

    def test_count_cmol_mixed_molecules(self):
        """Test with mixed carbon and non-carbon molecules."""
        elements = ['C', 'H', 'O', 'H', 'H', 'C', 'C', 'H']
        molecules = [{0, 1}, {2, 3, 4}, {5, 6, 7}]
        
        result = count_cmol(elements, molecules)
        
        assert result == 2


class TestOutputMolInfo:
    """Test class for output_mol_info function."""

    def test_output_mol_info_basic(self):
        """Test basic mol.info file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                elements = ['C', 'H', 'O']
                molecules = [{0, 1}, {2}]
                positions = np.array([
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                    [0.25, 0.25, 0.25],
                ])
                
                output_mol_info(elements, molecules, positions)
                
                assert os.path.exists('mol.info')
                
                with open('mol.info', 'r') as f:
                    content = f.read()
                
                assert 'molecule : 0' in content
                assert 'molecule : 1' in content
                assert 'C' in content or 'H' in content
                assert 'O' in content
            finally:
                os.chdir(original_cwd)

    def test_output_mol_info_single_molecule(self):
        """Test mol.info output with single molecule."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                elements = ['C', 'H', 'H']
                molecules = [{0, 1, 2}]
                positions = np.array([
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                ])
                
                output_mol_info(elements, molecules, positions)
                
                with open('mol.info', 'r') as f:
                    lines = f.readlines()
                
                # Check that we have molecule header and 3 atom lines
                assert lines[0].strip() == 'molecule : 0'
                atom_lines = [l for l in lines if ':' in l and 'molecule' not in l]
                assert len(atom_lines) == 3
            finally:
                os.chdir(original_cwd)

    def test_output_mol_info_empty_molecules(self):
        """Test mol.info output with empty molecules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                elements = []
                molecules = []
                positions = np.array([]).reshape(0, 3)
                
                output_mol_info(elements, molecules, positions)
                
                assert os.path.exists('mol.info')
                
                with open('mol.info', 'r') as f:
                    content = f.read()
                
                assert content == ''
            finally:
                os.chdir(original_cwd)


class TestGetDataFromPwout:
    """Test class for get_data_from_pwout function."""

    def test_get_data_from_pwout_basic(self):
        """Test basic parsing of PWscf output file."""
        pwout_content = """
     Program PWSCF v.x.x starts on ...
     
     bravais-lattice index     =            0
     number of Kohn-Sham states=           10
     
     number of k points=     2
        cart. coord. in units 2pi/alat
        k(    1) = (   0.0000000   0.0000000   0.0000000), wk =   0.5000000
        k(    2) = (   0.5000000   0.5000000   0.5000000), wk =   0.5000000
     
          k = 0.0000 0.0000 0.0000 (   123 PWs)   bands (ev):

    -5.0000  -3.0000  -1.0000   1.0000   3.0000   5.0000   7.0000   9.0000
    11.0000  13.0000

          k = 0.5000 0.5000 0.5000 (   123 PWs)   bands (ev):

    -4.0000  -2.0000   0.0000   2.0000   4.0000   6.0000   8.0000  10.0000
    12.0000  14.0000

     the Fermi energy is     2.0000 ev
"""
        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.out'
        ) as f:
            f.write(pwout_content)
            temp_path = f.name
        
        try:
            eigvals, ef, kpoints = get_data_from_pwout(temp_path)
            
            assert isinstance(eigvals, np.ndarray)
            assert isinstance(kpoints, np.ndarray)
            assert ef == 2.0
            assert eigvals.shape[0] == 2  # 2 k-points
            assert eigvals.shape[1] == 10  # 10 bands
            assert kpoints.shape[0] == 2
            # Check first k-point is Gamma
            assert np.allclose(kpoints[0], [0.0, 0.0, 0.0])
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_get_data_from_pwout_single_kpoint(self):
        """Test parsing with single k-point."""
        pwout_content = """
     number of Kohn-Sham states=            4
     
     number of k points=     1
        cart. coord. in units 2pi/alat
        k(    1) = (   0.0000000   0.0000000   0.0000000), wk =   1.0000000
     
          k = 0.0000 0.0000 0.0000 (   123 PWs)   bands (ev):

    -10.0000  -5.0000   5.0000  10.0000

     the Fermi energy is     0.0000 ev
"""
        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.out'
        ) as f:
            f.write(pwout_content)
            temp_path = f.name
        
        try:
            eigvals, ef, kpoints = get_data_from_pwout(temp_path)
            
            assert eigvals.shape == (1, 4)
            assert ef == 0.0
            assert len(kpoints) == 1
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_get_data_from_pwout_many_bands(self):
        """Test parsing with more than 8 bands (multi-line output)."""
        pwout_content = """
     number of Kohn-Sham states=           12
     
     number of k points=     1
        cart. coord. in units 2pi/alat
        k(    1) = (   0.0000000   0.0000000   0.0000000), wk =   1.0000000
     
          k = 0.0000 0.0000 0.0000 (   123 PWs)   bands (ev):

    -12.0  -10.0   -8.0   -6.0   -4.0   -2.0    0.0    2.0
      4.0    6.0    8.0   10.0

     the Fermi energy is    -1.0000 ev
"""
        with tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.out'
        ) as f:
            f.write(pwout_content)
            temp_path = f.name
        
        try:
            eigvals, ef, kpoints = get_data_from_pwout(temp_path)
            
            assert eigvals.shape == (1, 12)
            assert ef == -1.0
            # Check eigenvalues are in correct order
            assert eigvals[0, 0] == -12.0
            assert eigvals[0, -1] == 10.0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestFindNearestBands:
    """Test class for find_nearest_bands function."""

    def test_find_nearest_bands_fermi_in_middle(self):
        """Test finding bands when Fermi energy is in the middle."""
        eigvals = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
        ef = 0.0
        nmol = 2
        
        result = find_nearest_bands(eigvals, ef, nmol)
        
        assert len(result) == 2
        # Should select bands closest to Fermi energy (indices 2 and 3)
        assert set(result) == {2, 3}

    def test_find_nearest_bands_fermi_at_low_energy(self):
        """Test finding bands when Fermi energy is below all bands."""
        eigvals = np.array([1.0, 2.0, 3.0, 4.0])
        ef = 0.0
        nmol = 2
        
        result = find_nearest_bands(eigvals, ef, nmol)
        
        assert len(result) == 2
        # Should select lowest bands (indices 0, 1)
        assert set(result) == {0, 1}

    def test_find_nearest_bands_fermi_at_high_energy(self):
        """Test finding bands when Fermi energy is above all bands."""
        eigvals = np.array([-4.0, -3.0, -2.0, -1.0])
        ef = 0.0
        nmol = 2
        
        result = find_nearest_bands(eigvals, ef, nmol)
        
        assert len(result) == 2
        # Should select highest bands (indices 2, 3)
        assert set(result) == {2, 3}

    def test_find_nearest_bands_single_band(self):
        """Test finding a single nearest band."""
        eigvals = np.array([-3.0, -1.0, 1.0, 3.0])
        ef = 0.0
        nmol = 1
        
        result = find_nearest_bands(eigvals, ef, nmol)
        
        assert len(result) == 1
        # Should select band closest to Fermi (index 1 or 2)
        assert result[0] in [1, 2]

    def test_find_nearest_bands_asymmetric(self):
        """Test with asymmetric band distribution."""
        eigvals = np.array([-10.0, -5.0, -1.0, 0.5, 10.0])
        ef = 0.0
        nmol = 3
        
        result = find_nearest_bands(eigvals, ef, nmol)
        
        assert len(result) == 3
        # Should select indices 2, 3, and 1 (closest to Fermi)
        assert 2 in result
        assert 3 in result

    def test_find_nearest_bands_all_bands(self):
        """Test selecting all bands."""
        eigvals = np.array([-2.0, -1.0, 1.0, 2.0])
        ef = 0.0
        nmol = 4
        
        result = find_nearest_bands(eigvals, ef, nmol)
        
        assert len(result) == 4
        assert set(result) == {0, 1, 2, 3}


class TestGetInitialPpin:
    """Test class for get_initial_ppin function."""

    def test_get_initial_ppin_basic(self):
        """Test basic PP input file generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                path = '/test/path'
                iband = 4  # 0-indexed, will be written as band 5
                primvecs = np.array([
                    [10.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [0.0, 0.0, 10.0],
                ])
                
                get_initial_ppin(path, iband, primvecs)
                
                # Check file was created
                expected_file = 'pwscf.pp_005.in'
                assert os.path.exists(expected_file)
                
                with open(expected_file, 'r') as f:
                    content = f.read()
                
                assert 'inputpp' in content
                assert 'plot' in content
                assert '/test/path/work' in content
                assert 'kband(1)' in content
            finally:
                os.chdir(original_cwd)

    def test_get_initial_ppin_band_numbering(self):
        """Test that band numbering is correct (1-indexed in output)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                path = '/test/path'
                iband = 0  # 0-indexed, will be written as band 1
                primvecs = np.eye(3) * 10.0
                
                get_initial_ppin(path, iband, primvecs)
                
                expected_file = 'pwscf.pp_001.in'
                assert os.path.exists(expected_file)
                
                with open(expected_file, 'r') as f:
                    content = f.read()
                
                # kband should be 1 (1-indexed)
                assert 'kband(1)' in content
                assert '= 1' in content
            finally:
                os.chdir(original_cwd)

    def test_get_initial_ppin_multiple_bands(self):
        """Test generating PP input for multiple bands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                path = '/test/path'
                primvecs = np.eye(3) * 10.0
                
                # Generate for bands 0, 1, 2
                for iband in [0, 1, 2]:
                    get_initial_ppin(path, iband, primvecs)
                
                # Check all files were created
                assert os.path.exists('pwscf.pp_001.in')
                assert os.path.exists('pwscf.pp_002.in')
                assert os.path.exists('pwscf.pp_003.in')
            finally:
                os.chdir(original_cwd)


class TestWriteBandDat:
    """Test class for write_band_dat function."""

    def test_write_band_dat_basic(self):
        """Test basic band.dat file writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                band_indices = [0, 1, 2]  # 0-indexed
                
                write_band_dat(band_indices)
                
                assert os.path.exists('band.dat')
                
                with open('band.dat', 'r') as f:
                    lines = f.readlines()
                
                assert int(lines[0].strip()) == 3  # Number of bands
                # Bands are written as 1-indexed with zero-padding
                assert lines[1].strip() == '001'
                assert lines[2].strip() == '002'
                assert lines[3].strip() == '003'
            finally:
                os.chdir(original_cwd)

    def test_write_band_dat_non_consecutive(self):
        """Test band.dat writing with non-consecutive bands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                band_indices = [2, 5, 10]  # 0-indexed
                
                write_band_dat(band_indices)
                
                with open('band.dat', 'r') as f:
                    lines = f.readlines()
                
                assert int(lines[0].strip()) == 3
                assert lines[1].strip() == '003'  # 2+1
                assert lines[2].strip() == '006'  # 5+1
                assert lines[3].strip() == '011'  # 10+1
            finally:
                os.chdir(original_cwd)

    def test_write_band_dat_single_band(self):
        """Test band.dat writing with single band."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                band_indices = [5]  # 0-indexed
                
                write_band_dat(band_indices)
                
                with open('band.dat', 'r') as f:
                    lines = f.readlines()
                
                assert int(lines[0].strip()) == 1
                assert lines[1].strip() == '006'
            finally:
                os.chdir(original_cwd)

    def test_write_band_dat_large_band_index(self):
        """Test band.dat writing with large band indices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                band_indices = [99, 100, 999]  # 0-indexed
                
                write_band_dat(band_indices)
                
                with open('band.dat', 'r') as f:
                    lines = f.readlines()
                
                assert int(lines[0].strip()) == 3
                assert lines[1].strip() == '100'
                assert lines[2].strip() == '101'
                assert lines[3].strip() == '1000'
            finally:
                os.chdir(original_cwd)

    def test_write_band_dat_empty_list(self):
        """Test band.dat writing with empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                band_indices = []
                
                write_band_dat(band_indices)
                
                with open('band.dat', 'r') as f:
                    lines = f.readlines()
                
                assert int(lines[0].strip()) == 0
                assert len(lines) == 1
            finally:
                os.chdir(original_cwd)


class TestFindNearestBandsEdgeCases:
    """Additional edge case tests for find_nearest_bands function."""

    def test_find_nearest_bands_exact_fermi_match(self):
        """Test when Fermi energy exactly matches an eigenvalue."""
        eigvals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        ef = 0.0
        nmol = 2
        
        result = find_nearest_bands(eigvals, ef, nmol)
        
        assert len(result) == 2
        # Band at index 2 should be included
        assert 2 in result

    def test_find_nearest_bands_duplicate_eigenvalues(self):
        """Test with duplicate eigenvalues."""
        eigvals = np.array([-2.0, -1.0, -1.0, 1.0, 1.0, 2.0])
        ef = 0.0
        nmol = 4
        
        result = find_nearest_bands(eigvals, ef, nmol)
        
        assert len(result) == 4

    def test_find_nearest_bands_large_nmol(self):
        """Test requesting more bands than available."""
        eigvals = np.array([-2.0, -1.0, 1.0, 2.0])
        ef = 0.0
        nmol = 3
        
        result = find_nearest_bands(eigvals, ef, nmol)
        
        assert len(result) == 3

    def test_find_nearest_bands_fermi_between_close_bands(self):
        """Test with Fermi energy between two close bands."""
        eigvals = np.array([-5.0, -0.1, 0.1, 5.0])
        ef = 0.0
        nmol = 2
        
        result = find_nearest_bands(eigvals, ef, nmol)
        
        assert len(result) == 2
        # Should select indices 1 and 2 (closest to Fermi)
        assert set(result) == {1, 2}


