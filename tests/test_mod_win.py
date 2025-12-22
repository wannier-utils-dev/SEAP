"""
Tests for mod_win module.
"""

import os
import tempfile
from seap.core.mod_win import main
from unittest.mock import patch


class TestModWin:
    """Test class for mod_win module."""

    def test_main_with_dis_flag(self):
        """Test main function with --dis flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                # Create pred directory
                os.makedirs('pred', exist_ok=True)
                
                # Create band.dat file
                with open('pred/band.dat', 'w') as f:
                    f.write('2\n')  # num_wann
                    f.write('3\n')  # band 1
                    f.write('4\n')  # band 2
                
                # Create nbnd file
                with open('pred/nbnd', 'w') as f:
                    f.write('10\n')  # total bands
                
                # Create pwscf.win file
                with open('pwscf.win', 'w') as f:
                    f.write('num_bands = 10\n')
                    f.write('num_wann = 2\n')
                    f.write('exclude_bands = 1-4\n')
                    f.write('num_iter = 0\n')
                    f.write('begin projections\n')
                    f.write('  H: s\n')
                    f.write('end projections\n')
                
                # Create pw2wan.in file
                with open('pw2wan.in', 'w') as f:
                    f.write('write_unk = .true.\n')
                    f.write('other_line = value\n')
                
                # Create proj.out file
                os.makedirs('pred', exist_ok=True)
                with open('pred/proj.out', 'w') as f:
                    f.write('begin projections\n')
                    f.write('  H: px\n')
                    f.write('end projections\n')
                
                # Run main with --dis flag
                with patch('sys.argv', ['mod_win.py', '--dis']):
                    main()
                
                # Check pwscf.win was modified
                assert os.path.exists('pwscf.win')
                with open('pwscf.win', 'r') as f:
                    content = f.read()
                    assert 'num_bands = 8' in content  # 10 - 2 (bands below min)
                    assert 'num_wann = 2' in content
                    assert 'H: px' in content  # From proj.out
                
                # Check pw2wan.in was modified (write_unk should be removed)
                with open('pw2wan.in', 'r') as f:
                    content = f.read()
                    assert 'write_unk' not in content
                    assert 'other_line = value' in content
            finally:
                os.chdir(original_cwd)

    def test_main_with_maxloc_flag(self):
        """Test main function with --maxloc flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                # Create pred directory
                os.makedirs('pred', exist_ok=True)
                
                # Create band.dat file
                with open('pred/band.dat', 'w') as f:
                    f.write('2\n')
                    f.write('5\n')
                    f.write('6\n')
                
                # Create nbnd file
                with open('pred/nbnd', 'w') as f:
                    f.write('10\n')
                
                # Create pwscf.win file
                with open('pwscf.win', 'w') as f:
                    f.write('num_bands = 10\n')
                    f.write('num_wann = 2\n')
                    f.write('num_iter = 0\n')
                    f.write('begin projections\n')
                    f.write('  H: s\n')
                    f.write('end projections\n')
                
                # Create pw2wan.in file
                with open('pw2wan.in', 'w') as f:
                    f.write('other_line = value\n')
                
                # Create proj.out file
                with open('pred/proj.out', 'w') as f:
                    f.write('begin projections\n')
                    f.write('  H: px\n')
                    f.write('end projections\n')
                
                # Run main with --maxloc flag
                with patch('sys.argv', ['mod_win.py', '--maxloc']):
                    main()
                
                # Check num_iter was set to 300
                with open('pwscf.win', 'r') as f:
                    content = f.read()
                    assert 'num_iter = 300' in content
            finally:
                os.chdir(original_cwd)

    def test_main_with_wplot_flag(self):
        """Test main function with --wplot flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                # Create pred directory
                os.makedirs('pred', exist_ok=True)
                
                # Create band.dat file
                with open('pred/band.dat', 'w') as f:
                    f.write('2\n')
                    f.write('5\n')
                    f.write('6\n')
                
                # Create nbnd file
                with open('pred/nbnd', 'w') as f:
                    f.write('10\n')
                
                # Create pwscf.win file
                with open('pwscf.win', 'w') as f:
                    f.write('num_bands = 10\n')
                    f.write('num_wann = 2\n')
                    f.write('wannier_plot = .true.\n')
                    f.write('begin projections\n')
                    f.write('  H: s\n')
                    f.write('end projections\n')
                
                # Create pw2wan.in file
                with open('pw2wan.in', 'w') as f:
                    f.write('write_unk = .true.\n')
                
                # Create proj.out file
                with open('pred/proj.out', 'w') as f:
                    f.write('begin projections\n')
                    f.write('  H: px\n')
                    f.write('end projections\n')
                
                # Run main with --wplot flag
                with patch('sys.argv', ['mod_win.py', '--wplot']):
                    main()
                
                # Check wannier_plot_supercell was added
                with open('pwscf.win', 'r') as f:
                    content = f.read()
                    assert 'wannier_plot_supercell = 3' in content
                
                # Check write_unk was kept
                with open('pw2wan.in', 'r') as f:
                    content = f.read()
                    assert 'write_unk = .true.' in content
            finally:
                os.chdir(original_cwd)

    def test_main_exclude_bands_calculation(self):
        """Test exclude_bands calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                # Create pred directory
                os.makedirs('pred', exist_ok=True)
                
                # Create band.dat file with non-consecutive bands
                with open('pred/band.dat', 'w') as f:
                    f.write('3\n')
                    f.write('2\n')  # band 2
                    f.write('5\n')  # band 5
                    f.write('8\n')  # band 8
                
                # Create nbnd file
                with open('pred/nbnd', 'w') as f:
                    f.write('10\n')
                
                # Create pwscf.win file
                with open('pwscf.win', 'w') as f:
                    f.write('num_bands = 10\n')
                    f.write('num_wann = 3\n')
                    f.write('exclude_bands = 1-1\n')
                    f.write('begin projections\n')
                    f.write('  H: s\n')
                    f.write('end projections\n')
                
                # Create pw2wan.in file
                with open('pw2wan.in', 'w') as f:
                    f.write('other_line = value\n')
                
                # Create proj.out file
                with open('pred/proj.out', 'w') as f:
                    f.write('begin projections\n')
                    f.write('  H: px\n')
                    f.write('end projections\n')
                
                # Run main without --dis flag
                with patch('sys.argv', ['mod_win.py']):
                    main()
                
                # Check exclude_bands was calculated correctly
                with open('pwscf.win', 'r') as f:
                    content = f.read()
                    # Should exclude bands 1, 3-4, 6-7, 9-10
                    assert 'exclude_bands' in content
                    assert 'num_bands = 3' in content  # num_wann
            finally:
                os.chdir(original_cwd)

    def test_main_no_exclude_bands(self):
        """Test main when no bands need to be excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                # Create pred directory
                os.makedirs('pred', exist_ok=True)
                
                # Create band.dat file with all bands
                with open('pred/band.dat', 'w') as f:
                    f.write('3\n')
                    f.write('1\n')
                    f.write('2\n')
                    f.write('3\n')
                
                # Create nbnd file
                with open('pred/nbnd', 'w') as f:
                    f.write('3\n')
                
                # Create pwscf.win file
                with open('pwscf.win', 'w') as f:
                    f.write('num_bands = 3\n')
                    f.write('num_wann = 3\n')
                    f.write('begin projections\n')
                    f.write('  H: s\n')
                    f.write('end projections\n')
                
                # Create pw2wan.in file
                with open('pw2wan.in', 'w') as f:
                    f.write('other_line = value\n')
                
                # Create proj.out file
                with open('pred/proj.out', 'w') as f:
                    f.write('begin projections\n')
                    f.write('  H: px\n')
                    f.write('end projections\n')
                
                # Run main
                with patch('sys.argv', ['mod_win.py']):
                    main()
                
                # Check exclude_bands was not added
                with open('pwscf.win', 'r') as f:
                    content = f.read()
                    # exclude_bands should not appear if nexcludes == 0
                    lines = content.split('\n')
                    # If exclude_bands was in original, it should be removed
                    assert 'num_bands = 3' in content
            finally:
                os.chdir(original_cwd)

    def test_main_skip_lines(self):
        """Test that certain lines are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                # Create pred directory
                os.makedirs('pred', exist_ok=True)
                
                # Create band.dat file
                with open('pred/band.dat', 'w') as f:
                    f.write('2\n')
                    f.write('5\n')
                    f.write('6\n')
                
                # Create nbnd file
                with open('pred/nbnd', 'w') as f:
                    f.write('10\n')
                
                # Create pwscf.win file with lines to skip
                with open('pwscf.win', 'w') as f:
                    f.write('num_bands = 10\n')
                    f.write('num_wann = 2\n')
                    f.write('fermi_surface_plot = .true.\n')
                    f.write('write_tb = .true.\n')
                    f.write('wannier_plot_supercell = 2\n')
                    f.write('begin projections\n')
                    f.write('  H: s\n')
                    f.write('end projections\n')
                
                # Create pw2wan.in file
                with open('pw2wan.in', 'w') as f:
                    f.write('other_line = value\n')
                
                # Create proj.out file
                with open('pred/proj.out', 'w') as f:
                    f.write('begin projections\n')
                    f.write('  H: px\n')
                    f.write('end projections\n')
                
                # Run main
                with patch('sys.argv', ['mod_win.py']):
                    main()
                
                # Check skipped lines are not in output
                with open('pwscf.win', 'r') as f:
                    content = f.read()
                    assert 'fermi_surface_plot' not in content
                    assert 'write_tb' not in content
                    # wannier_plot_supercell should be removed if wplot is False
                    assert 'wannier_plot_supercell = 2' not in content
            finally:
                os.chdir(original_cwd)

