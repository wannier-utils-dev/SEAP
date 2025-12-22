"""
Tests for file I/O functions in utils module.
"""

import os
import tempfile
import numpy as np


class TestReadLoss:
    """Test class for read_loss function."""

    def test_read_loss_basic(self):
        """Test basic read_loss functionality."""
        from seap.prediction.utils import read_loss
        
        # Create a temporary file with loss data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.dat') as f:
            f.write("Header line\n")
            f.write("3\n")  # Number of Wannier functions
            f.write("0 0.0001\n")
            f.write("1 0.0002\n")
            f.write("2 0.0003\n")
            temp_path = f.name
        
        try:
            data_idxs, losses = read_loss(temp_path)
            
            assert len(data_idxs) == 3
            assert len(losses) == 3
            assert data_idxs == [0, 1, 2]
            assert losses == [0.0001, 0.0002, 0.0003]
        finally:
            os.unlink(temp_path)

    def test_read_loss_empty(self):
        """Test read_loss with empty file."""
        from seap.prediction.utils import read_loss
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.dat') as f:
            f.write("Header line\n")
            f.write("0\n")  # Number of Wannier functions
            temp_path = f.name
        
        try:
            data_idxs, losses = read_loss(temp_path)
            
            assert len(data_idxs) == 0
            assert len(losses) == 0
        finally:
            os.unlink(temp_path)


class TestReadWaninfo:
    """Test class for read_waninfo function."""

    def test_read_waninfo_basic(self):
        """Test basic read_waninfo functionality."""
        from seap.prediction.utils import read_waninfo
        
        # Create a temporary file with Wannier info
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.dat') as f:
            f.write("Header line\n")
            f.write("2\n")  # Number of Wannier functions
            f.write("0 100 0 0.0001 1.5\n")
            f.write("1 100 1 0.0002 2.0\n")
            temp_path = f.name
        
        try:
            data_idx, mp_idx, wn_idx, loss_list, spread_list = read_waninfo(temp_path)
            
            assert len(data_idx) == 2
            assert len(mp_idx) == 2
            assert len(wn_idx) == 2
            assert len(loss_list) == 2
            assert len(spread_list) == 2
            assert data_idx == [0, 1]
            assert mp_idx == [100, 100]
            assert wn_idx == [0, 1]
            assert loss_list == [0.0001, 0.0002]
            assert spread_list == [1.5, 2.0]
        finally:
            os.unlink(temp_path)


class TestGetWannierData:
    """Test class for get_wannier_data function."""

    def test_get_wannier_data_basic(self):
        """Test basic get_wannier_data functionality."""
        from seap.prediction.utils import get_wannier_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy data file
            n_materials = 2
            n_wannier = 3
            # Create data with some zeros (blank images)
            data = np.random.rand(n_materials, n_wannier, 4, 4, 4)
            data[0, 1] = 0.0  # Make one blank
            
            data_path = os.path.join(tmpdir, "images_center.npy")
            np.save(data_path, data)
            
            data_idx, wannier_data = get_wannier_data(tmpdir)
            
            assert isinstance(data_idx, list)
            assert isinstance(wannier_data, list)
            # Should exclude blank images
            assert len(data_idx) == len(wannier_data)
            assert len(data_idx) == n_materials * n_wannier - 1  # One blank excluded

    def test_get_wannier_data_all_blank(self):
        """Test get_wannier_data with all blank images."""
        from seap.prediction.utils import get_wannier_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data with all zeros
            data = np.zeros((2, 2, 4, 4, 4))
            
            data_path = os.path.join(tmpdir, "images_center.npy")
            np.save(data_path, data)
            
            data_idx, wannier_data = get_wannier_data(tmpdir)
            
            assert len(data_idx) == 0
            assert len(wannier_data) == 0


class TestGetOriginalIndex:
    """Test class for get_original_index function."""

    def test_get_original_index_basic(self):
        """Test basic get_original_index functionality."""
        from seap.prediction.utils import get_original_index
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy mp-id file
            mp_ids = np.array([100, 200, 300])
            mp_id_path = os.path.join(tmpdir, "mp-id.npy")
            np.save(mp_id_path, mp_ids)
            
            data_idx = [0, 5, 10]  # Indices
            nwan = 5
            
            mp_idx, wn_idx = get_original_index(tmpdir, data_idx, nwan)
            
            assert len(mp_idx) == len(data_idx)
            assert len(wn_idx) == len(data_idx)
            assert mp_idx == [100, 200, 300]  # Material indices
            assert wn_idx == [0, 0, 0]  # Wannier indices (0, 5, 10 mod 5 = 0, 0, 0)


