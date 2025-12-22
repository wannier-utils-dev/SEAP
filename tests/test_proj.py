"""
Tests for proj module.
"""

import os
import tempfile
import pandas as pd
from seap.core.proj import generate_projection_string, write_projection_to_file


class TestGenerateProjectionString:
    """Test class for generate_projection_string function."""

    def test_generate_projection_string_basic(self):
        """Test basic projection string generation."""
        # Create a sample DataFrame
        data = {
            'band_index': [0, 1],
            'molecule_index': [0, 0],
            'center_x': [0.0, 0.5],
            'center_y': [0.0, 0.5],
            'center_z': [0.0, 0.5],
            'orbital': ['s', 'px']
        }
        df = pd.DataFrame(data)
        
        result = generate_projection_string(df)
        
        assert isinstance(result, str)
        assert 'begin projections' in result
        assert 'end projections' in result
        assert 'Ang' in result
        assert 's' in result
        assert 'px' in result

    def test_generate_projection_string_formatting(self):
        """Test projection string formatting."""
        data = {
            'band_index': [0],
            'molecule_index': [0],
            'center_x': [1.23456789],
            'center_y': [2.34567890],
            'center_z': [3.45678901],
            'orbital': ['s']
        }
        df = pd.DataFrame(data)
        
        result = generate_projection_string(df)
        
        # Check that coordinates are formatted with 8 decimal places
        assert '1.23456789' in result or '1.23456788' in result  # Allow for rounding
        assert 'c=' in result

    def test_generate_projection_string_empty(self):
        """Test projection string generation with empty DataFrame."""
        df = pd.DataFrame(columns=['band_index', 'molecule_index', 'center_x', 'center_y', 'center_z', 'orbital'])
        
        result = generate_projection_string(df)
        
        assert 'begin projections' in result
        assert 'end projections' in result
        assert 'Ang' in result


class TestWriteProjectionToFile:
    """Test class for write_projection_to_file function."""

    def test_write_projection_to_file_basic(self):
        """Test basic file writing."""
        proj_string = "begin projections\nAng\nc=0.0,0.0,0.0:s\nend projections\n\n"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.out') as f:
            temp_path = f.name
        
        try:
            write_projection_to_file(proj_string, temp_path)
            
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                content = f.read()
            assert content == proj_string
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_write_projection_to_file_empty(self):
        """Test writing empty projection string."""
        proj_string = ""
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.out') as f:
            temp_path = f.name
        
        try:
            write_projection_to_file(proj_string, temp_path)
            
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                content = f.read()
            assert content == ""
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


