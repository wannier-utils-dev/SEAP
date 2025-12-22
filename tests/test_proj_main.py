"""
Tests for proj module main function.
"""

import os
import tempfile
import pandas as pd
from seap.core.proj import main


class TestProjMain:
    """Test class for proj main function."""

    def test_main_with_mock_files(self):
        """Test main function with mocked CSV files."""
        # Create temporary CSV files
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                
                # Create center.csv
                center_data = {
                    'band_index': [0, 1],
                    'molecule_index': [0, 0],
                    'center_x': [0.0, 0.5],
                    'center_y': [0.0, 0.5],
                    'center_z': [0.0, 0.5],
                }
                center_df = pd.DataFrame(center_data)
                center_df.to_csv('center.csv', index=False)
                
                # Create orbital.csv
                orbital_data = {
                    'band_index': [0, 1],
                    'molecule_index': [0, 0],
                    'orbital': ['s', 'px'],
                }
                orbital_df = pd.DataFrame(orbital_data)
                orbital_df.to_csv('orbital.csv', index=False)
                
                # Run main function
                main()
                
                # Check that proj.out was created
                assert os.path.exists('proj.out')
                
                # Read and verify the output
                with open('proj.out', 'r') as f:
                    content = f.read()
                
                assert 'begin projections' in content
                assert 'end projections' in content
                assert 's' in content
                assert 'px' in content
            finally:
                os.chdir(original_cwd)

