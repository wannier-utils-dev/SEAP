"""
Tests for qe module (Quantum ESPRESSO interface).
"""

import os
import tempfile
import pytest
import numpy as np
from seap.interface.qe import PWIn, PPIn


class TestPPIn:
    """Test class for PPIn class."""

    def test_ppin_initialization_with_region(self):
        """Test PPIn initialization with set_region=True."""
        ppin = PPIn(set_region=True)
        
        assert ppin.set_region is True
        assert 'inputpp' in ppin.__dict__
        assert 'plot' in ppin.__dict__
        assert 'iflag' in ppin.plot
        assert 'e1(1)' in ppin.plot
        assert 'x0(1)' in ppin.plot

    def test_ppin_initialization_without_region(self):
        """Test PPIn initialization with set_region=False."""
        ppin = PPIn(set_region=False)
        
        assert ppin.set_region is False
        assert 'inputpp' in ppin.__dict__
        assert 'plot' in ppin.__dict__
        assert 'iflag' in ppin.plot
        assert 'e1(1)' not in ppin.plot  # Should not have e1 when set_region=False
        assert 'x0(1)' not in ppin.plot  # Should not have x0 when set_region=False

    def test_set_prefix(self):
        """Test set_prefix method."""
        ppin = PPIn()
        ppin.set_prefix("'test_prefix'")
        
        assert ppin.inputpp['prefix'] == "'test_prefix'"

    def test_set_outdir(self):
        """Test set_outdir method."""
        ppin = PPIn()
        ppin.set_outdir("'test_dir'")
        
        assert ppin.inputpp['outdir'] == "'test_dir'"

    def test_set_kband_single(self):
        """Test set_kband with single index."""
        ppin = PPIn()
        ppin.set_kband(5)
        
        assert ppin.inputpp['kband(1)'] == 5
        assert 'kband(2)' not in ppin.inputpp

    def test_set_kband_double(self):
        """Test set_kband with two indices."""
        ppin = PPIn()
        ppin.set_kband(5, 10)
        
        assert ppin.inputpp['kband(1)'] == 5
        assert ppin.inputpp['kband(2)'] == 10

    def test_set_fileout_with_region(self):
        """Test set_fileout with set_region=True."""
        ppin = PPIn(set_region=True)
        ppin.set_fileout('test')
        
        assert ppin.plot['fileout'] == 'test.wfc.xsf'

    def test_set_fileout_without_region(self):
        """Test set_fileout with set_region=False."""
        ppin = PPIn(set_region=False)
        ppin.set_fileout('test')
        
        assert ppin.plot['fileout'] == 'test.xsf'

    def test_set_cubic_e(self):
        """Test set_cubic_e method."""
        ppin = PPIn(set_region=True)
        ppin.set_cubic_e(2.0)
        
        assert ppin.plot['e1(1)'] == 2.0
        assert ppin.plot['e2(2)'] == 2.0
        assert ppin.plot['e3(3)'] == 2.0

    def test_set_cubic_e_without_region(self):
        """Test set_cubic_e raises error when set_region=False."""
        ppin = PPIn(set_region=False)
        
        with pytest.raises(SystemExit):
            ppin.set_cubic_e(2.0)

    def test_set_origin(self):
        """Test set_origin method."""
        ppin = PPIn(set_region=True)
        ppin.set_origin([0.5, 0.5, 0.5])
        
        assert ppin.plot['x0(1)'] == 0.5
        assert ppin.plot['x0(2)'] == 0.5
        assert ppin.plot['x0(3)'] == 0.5

    def test_set_origin_without_region(self):
        """Test set_origin raises error when set_region=False."""
        ppin = PPIn(set_region=False)
        
        with pytest.raises(SystemExit):
            ppin.set_origin([0.5, 0.5, 0.5])

    def test_set_mesh(self):
        """Test set_mesh method."""
        ppin = PPIn()
        ppin.set_mesh([32, 64, 128])
        
        assert ppin.plot['nx'] == 32
        assert ppin.plot['ny'] == 64
        assert ppin.plot['nz'] == 128

    def test_stringify_group(self):
        """Test _stringify_group method."""
        ppin = PPIn()
        group = {'key1': 'value1', 'key2': 123}
        result = ppin._stringify_group('test_group', group)
        
        assert '&test_group' in result
        assert 'key1' in result
        assert 'value1' in result
        assert 'key2' in result
        assert '123' in result
        assert '/\n' in result

    def test_stringify_group_with_none(self):
        """Test _stringify_group with None values."""
        ppin = PPIn()
        group = {'key1': 'value1', 'key2': None}
        result = ppin._stringify_group('test_group', group)
        
        assert 'key1' in result
        assert 'key2' not in result  # None values should be skipped

    def test_write(self):
        """Test write method."""
        ppin = PPIn()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = 'test_pp.in'
            ppin.write(output_file, folder=tmpdir)
            
            output_path = os.path.join(tmpdir, output_file)
            assert os.path.exists(output_path)
            
            with open(output_path, 'r') as f:
                content = f.read()
            
            assert '&inputpp' in content
            assert '&plot' in content
            assert '/\n' in content

    def test_write_with_subfolder(self):
        """Test write method with subfolder creation."""
        ppin = PPIn()
        
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                subfolder = 'subfolder'
                output_file = 'test_pp.in'
                ppin.write(output_file, folder=subfolder)
                
                output_path = os.path.join(subfolder, output_file)
                assert os.path.exists(output_path)
            finally:
                os.chdir(original_cwd)
                # Clean up subfolder if created in original directory
                subfolder_path = os.path.join(original_cwd, 'subfolder')
                if os.path.exists(subfolder_path):
                    import shutil
                    shutil.rmtree(subfolder_path)


class TestPWIn:
    """Test class for PWIn class."""

    def test_pwin_file_not_found(self):
        """Test PWIn with non-existent file."""
        # Should not raise exception, but print error message
        pw_in = PWIn('nonexistent_file.in')
        # Check that attributes are not set when file doesn't exist
        assert not hasattr(pw_in, 'atoms') or pw_in.atoms is None

    def test_pwin_basic_reading(self):
        """Test PWIn with a basic input file."""
        # Create a minimal PWscf input file
        input_content = """&control
  calculation = 'scf'
/
&system
  ibrav = 0
  nat = 2
  ntyp = 1
  A = 10.0
/
&electrons
  conv_thr = 1.0d-6
/
CELL_PARAMETERS alat
  1.0  0.0  0.0
  0.0  1.0  0.0
  0.0  0.0  1.0
ATOMIC_SPECIES
  H  1.0  H.pbe.UPF
ATOMIC_POSITIONS crystal
  H  0.0  0.0  0.0
  H  0.5  0.5  0.5
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.in') as f:
            f.write(input_content)
            temp_path = f.name
        
        try:
            pw_in = PWIn(temp_path)
            
            assert pw_in.ibrav == 0
            assert len(pw_in.atoms) == 2
            assert len(pw_in.atypes) == 1
            assert 'H' in pw_in.atypes
            assert pw_in.atomic_pos_type == 'crystal'
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_pwin_get_atomic_positions_same_unit(self):
        """Test get_atomic_positions with same unit."""
        input_content = """&control
  calculation = 'scf'
/
&system
  ibrav = 0
  nat = 2
  ntyp = 1
  A = 10.0
/
&electrons
  conv_thr = 1.0d-6
/
CELL_PARAMETERS alat
  1.0  0.0  0.0
  0.0  1.0  0.0
  0.0  0.0  1.0
ATOMIC_SPECIES
  H  1.0  H.pbe.UPF
ATOMIC_POSITIONS crystal
  H  0.0  0.0  0.0
  H  0.5  0.5  0.5
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.in') as f:
            f.write(input_content)
            temp_path = f.name
        
        try:
            pw_in = PWIn(temp_path)
            positions = pw_in.get_atomic_positions(ounit='crystal')
            
            assert isinstance(positions, np.ndarray)
            assert positions.shape[0] == 2
            assert positions.shape[1] == 3
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_pwin_get_atomic_positions_convert(self):
        """Test get_atomic_positions with unit conversion."""
        input_content = """&control
  calculation = 'scf'
/
&system
  ibrav = 0
  nat = 1
  ntyp = 1
  A = 10.0
/
&electrons
  conv_thr = 1.0d-6
/
CELL_PARAMETERS alat
  1.0  0.0  0.0
  0.0  1.0  0.0
  0.0  0.0  1.0
ATOMIC_SPECIES
  H  1.0  H.pbe.UPF
ATOMIC_POSITIONS crystal
  H  0.0  0.0  0.0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.in') as f:
            f.write(input_content)
            temp_path = f.name
        
        try:
            pw_in = PWIn(temp_path)
            # Convert from crystal to angstrom
            positions = pw_in.get_atomic_positions(ounit='angstrom')
            
            assert isinstance(positions, np.ndarray)
            assert positions.shape == (1, 3)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_pwin_bravais_lattice_ibrav6(self):
        """Test PWIn with ibrav=6 (tetragonal)."""
        input_content = """&control
  calculation = 'scf'
/
&system
  ibrav = 6
  nat = 1
  ntyp = 1
  a = 5.0
  c = 10.0
/
&electrons
  conv_thr = 1.0d-6
/
ATOMIC_SPECIES
  H  1.0  H.pbe.UPF
ATOMIC_POSITIONS crystal
  H  0.0  0.0  0.0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.in') as f:
            f.write(input_content)
            temp_path = f.name
        
        try:
            pw_in = PWIn(temp_path)
            
            assert pw_in.ibrav == 6
            assert pw_in.cell_parameters.shape == (3, 3)
            assert pw_in.alat == 5.0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_pwin_bravais_lattice_ibrav5(self):
        """Test PWIn with ibrav=5 (trigonal)."""
        input_content = """&control
  calculation = 'scf'
/
&system
  ibrav = 5
  nat = 1
  ntyp = 1
  a = 5.0
  cosab = 0.5
/
&electrons
  conv_thr = 1.0d-6
/
ATOMIC_SPECIES
  H  1.0  H.pbe.UPF
ATOMIC_POSITIONS crystal
  H  0.0  0.0  0.0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.in') as f:
            f.write(input_content)
            temp_path = f.name
        
        try:
            pw_in = PWIn(temp_path)
            
            assert pw_in.ibrav == 5
            assert pw_in.cell_parameters.shape == (3, 3)
            assert pw_in.alat == 5.0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_pwin_bravais_lattice_ibrav_minus3(self):
        """Test PWIn with ibrav=-3 (rhombohedral)."""
        input_content = """&control
  calculation = 'scf'
/
&system
  ibrav = -3
  nat = 1
  ntyp = 1
  a = 5.0
/
&electrons
  conv_thr = 1.0d-6
/
ATOMIC_SPECIES
  H  1.0  H.pbe.UPF
ATOMIC_POSITIONS crystal
  H  0.0  0.0  0.0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.in') as f:
            f.write(input_content)
            temp_path = f.name
        
        try:
            pw_in = PWIn(temp_path)
            
            assert pw_in.ibrav == -3
            assert pw_in.cell_parameters.shape == (3, 3)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_pwin_cell_parameters_angstrom(self):
        """Test PWIn with CELL_PARAMETERS in angstrom."""
        input_content = """&control
  calculation = 'scf'
/
&system
  ibrav = 0
  nat = 1
  ntyp = 1
  A = 10.0
/
&electrons
  conv_thr = 1.0d-6
/
CELL_PARAMETERS angstrom
  10.0  0.0  0.0
  0.0  10.0  0.0
  0.0  0.0  10.0
ATOMIC_SPECIES
  H  1.0  H.pbe.UPF
ATOMIC_POSITIONS crystal
  H  0.0  0.0  0.0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.in') as f:
            f.write(input_content)
            temp_path = f.name
        
        try:
            pw_in = PWIn(temp_path)
            
            assert pw_in.ibrav == 0
            assert pw_in.cell_parameters.shape == (3, 3)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_pwin_atomic_positions_bohr(self):
        """Test PWIn with ATOMIC_POSITIONS in bohr."""
        input_content = """&control
  calculation = 'scf'
/
&system
  ibrav = 0
  nat = 1
  ntyp = 1
  A = 10.0
/
&electrons
  conv_thr = 1.0d-6
/
CELL_PARAMETERS alat
  1.0  0.0  0.0
  0.0  1.0  0.0
  0.0  0.0  1.0
ATOMIC_SPECIES
  H  1.0  H.pbe.UPF
ATOMIC_POSITIONS bohr
  H  0.0  0.0  0.0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.in') as f:
            f.write(input_content)
            temp_path = f.name
        
        try:
            pw_in = PWIn(temp_path)
            
            assert pw_in.atomic_pos_type == 'bohr'
            positions = pw_in.get_atomic_positions(ounit='angstrom')
            assert isinstance(positions, np.ndarray)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_pwin_atomic_positions_alat(self):
        """Test PWIn with ATOMIC_POSITIONS in alat."""
        input_content = """&control
  calculation = 'scf'
/
&system
  ibrav = 0
  nat = 1
  ntyp = 1
  A = 10.0
/
&electrons
  conv_thr = 1.0d-6
/
CELL_PARAMETERS alat
  1.0  0.0  0.0
  0.0  1.0  0.0
  0.0  0.0  1.0
ATOMIC_SPECIES
  H  1.0  H.pbe.UPF
ATOMIC_POSITIONS alat
  H  0.0  0.0  0.0
"""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.in') as f:
            f.write(input_content)
            temp_path = f.name
        
        try:
            pw_in = PWIn(temp_path)
            
            assert pw_in.atomic_pos_type == 'alat'
            positions = pw_in.get_atomic_positions(ounit='crystal')
            assert isinstance(positions, np.ndarray)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

