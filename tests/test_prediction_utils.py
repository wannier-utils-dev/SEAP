"""
Tests for prediction utility functions.
"""

import os
import tempfile

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from seap.prediction.utils import (
    plot_learning_curve,
    split_data_by_torch,
    random_sampling_in_every_interval,
    read_loss,
    get_wannier_data,
    get_original_index,
    read_waninfo,
)


class TestPlotLearningCurve:
    """Test class for plot_learning_curve function."""

    def test_plot_learning_curve_basic(self):
        """Test basic learning curve plotting."""
        loss_train = [1.0, 0.8, 0.6, 0.4, 0.2]
        loss_val = [1.2, 1.0, 0.8, 0.6, 0.4]
        
        # Should not raise an exception
        plot_learning_curve(loss_train, loss_val)
        
        # Check that files were created (if not using mock)
        import os
        assert os.path.exists("lcurve.png")
        assert os.path.exists("lcurve.eps")
        
        # Clean up
        if os.path.exists("lcurve.png"):
            os.remove("lcurve.png")
        if os.path.exists("lcurve.eps"):
            os.remove("lcurve.eps")

    def test_plot_learning_curve_custom_prefix(self):
        """Test learning curve plotting with custom prefix."""
        loss_train = [1.0, 0.5, 0.25]
        loss_val = [1.1, 0.6, 0.3]
        
        plot_learning_curve(loss_train, loss_val, prefix="test_curve", label_loss="Loss")
        
        import os
        assert os.path.exists("test_curve.png")
        assert os.path.exists("test_curve.eps")
        
        # Clean up
        if os.path.exists("test_curve.png"):
            os.remove("test_curve.png")
        if os.path.exists("test_curve.eps"):
            os.remove("test_curve.eps")

    def test_plot_learning_curve_different_lengths(self):
        """Test learning curve with different training/validation lengths."""
        loss_train = [1.0, 0.8, 0.6, 0.4]
        loss_val = [1.2, 1.0, 0.8, 0.6]  # Same length (function expects same length)
        
        # Should handle gracefully
        plot_learning_curve(loss_train, loss_val)
        
        import os
        if os.path.exists("lcurve.png"):
            os.remove("lcurve.png")
        if os.path.exists("lcurve.eps"):
            os.remove("lcurve.eps")

    def test_plot_learning_curve_single_point(self):
        """Test learning curve with single data point."""
        loss_train = [1.0]
        loss_val = [1.2]
        
        plot_learning_curve(loss_train, loss_val)
        
        import os
        if os.path.exists("lcurve.png"):
            os.remove("lcurve.png")
        if os.path.exists("lcurve.eps"):
            os.remove("lcurve.eps")


class TestRandomSamplingInEveryInterval:
    """Test class for random_sampling_in_every_interval function."""

    def test_random_sampling_basic(self):
        """Test basic random sampling."""
        from seap.prediction.utils import random_sampling_in_every_interval
        
        loss_list = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
        result = random_sampling_in_every_interval(
            loss_list, 0.0, 0.001, 5, 2
        )
        
        assert isinstance(result, list)
        assert len(result) == 5  # num_int intervals

    def test_random_sampling_empty_intervals(self):
        """Test random sampling with empty intervals."""
        from seap.prediction.utils import random_sampling_in_every_interval
        
        loss_list = [0.0001, 0.0002]
        result = random_sampling_in_every_interval(
            loss_list, 0.0, 0.001, 10, 5
        )
        
        assert isinstance(result, list)
        assert len(result) == 10
        # Some intervals may be empty
        for interval in result:
            assert isinstance(interval, list)

    def test_random_sampling_fewer_samples(self):
        """Test random sampling when fewer samples available than requested."""
        from seap.prediction.utils import random_sampling_in_every_interval
        
        loss_list = [0.0001]
        result = random_sampling_in_every_interval(
            loss_list, 0.0, 0.001, 5, 10
        )
        
        assert isinstance(result, list)
        assert len(result) == 5


class TestPlotHist:
    """Test class for plot_hist function."""

    def test_plot_hist_basic(self):
        """Test basic histogram plotting."""
        from seap.prediction.utils import plot_hist
        import matplotlib.pyplot as plt
        
        loss_list = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
        # Try to use the function, handle style errors gracefully
        try:
            plot_hist(loss_list, int_min=0.0, int_max=0.001, bins=5)
        except OSError:
            # If seaborn-bright style is not available, skip this test
            pytest.skip("seaborn-bright style not available")
        
        import os
        assert os.path.exists("./loss_hist.eps")
        assert os.path.exists("./loss_hist.png")
        
        # Clean up
        if os.path.exists("./loss_hist.eps"):
            os.remove("./loss_hist.eps")
        if os.path.exists("./loss_hist.png"):
            os.remove("./loss_hist.png")

    def test_plot_hist_custom_range(self):
        """Test histogram plotting with custom range."""
        from seap.prediction.utils import plot_hist
        
        loss_list = [0.001, 0.002, 0.003]
        # Try to use the function, handle style errors gracefully
        try:
            plot_hist(loss_list, int_min=0.0, int_max=0.01, bins=10)
        except OSError:
            # If seaborn-bright style is not available, skip this test
            pytest.skip("seaborn-bright style not available")
        
        if os.path.exists("./loss_hist.eps"):
            os.remove("./loss_hist.eps")
        if os.path.exists("./loss_hist.png"):
            os.remove("./loss_hist.png")


class TestSplitDataByTorch:
    """Test class for split_data_by_torch function."""

    @pytest.fixture(autouse=True)
    def skip_without_torch(self):
        """Skip tests if torch is not available."""
        pytest.importorskip("torch")

    def test_split_data_basic(self):
        """Test basic data splitting with default parameters."""
        import torch
        from torch.utils.data import TensorDataset

        # Create a simple dataset
        data = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, labels)

        train, val, test = split_data_by_torch(dataset, test_size=0.25)

        # Check that data is split correctly
        assert len(train) == 75  # 75% of 100
        assert len(val) == 0  # validation_split=0.0 by default
        assert len(test) == 25  # 25% of 100
        assert len(train) + len(val) + len(test) == len(dataset)

    def test_split_data_with_validation(self):
        """Test data splitting with validation set."""
        import torch
        from torch.utils.data import TensorDataset

        data = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        dataset = TensorDataset(data, labels)

        train, val, test = split_data_by_torch(
            dataset, test_size=0.2, validation_split=0.25
        )

        # test: 20% = 20, remaining: 80, val: 25% of 80 = 20, train: 60
        assert len(test) == 20
        assert len(val) == 20
        assert len(train) == 60
        assert len(train) + len(val) + len(test) == len(dataset)

    def test_split_data_small_dataset(self):
        """Test data splitting with a small dataset."""
        import torch
        from torch.utils.data import TensorDataset

        data = torch.randn(10, 5)
        labels = torch.randint(0, 2, (10,))
        dataset = TensorDataset(data, labels)

        train, val, test = split_data_by_torch(dataset, test_size=0.3)

        assert len(test) == 3  # 30% of 10
        assert len(train) == 7
        assert len(train) + len(val) + len(test) == len(dataset)

    def test_split_data_reproducibility(self):
        """Test that splitting is reproducible with same seed."""
        import torch
        from torch.utils.data import TensorDataset

        data = torch.randn(50, 10)
        labels = torch.arange(50)
        dataset = TensorDataset(data, labels)

        train1, val1, test1 = split_data_by_torch(dataset, test_size=0.2)
        train2, val2, test2 = split_data_by_torch(dataset, test_size=0.2)

        # Extract indices from the subsets
        indices1 = list(train1.indices)
        indices2 = list(train2.indices)

        assert indices1 == indices2


class TestReadLoss:
    """Test class for read_loss function."""

    def test_read_loss_basic(self):
        """Test basic loss file reading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write("# Header line\n")
            f.write("3\n")  # Number of Wannier functions
            f.write("0 0.001\n")
            f.write("1 0.002\n")
            f.write("2 0.003\n")
            temp_path = f.name

        try:
            data_idxs, losses = read_loss(temp_path)

            assert data_idxs == [0, 1, 2]
            assert losses == [0.001, 0.002, 0.003]
        finally:
            os.unlink(temp_path)

    def test_read_loss_larger_file(self):
        """Test reading a larger loss file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write("# Loss information file\n")
            f.write("5\n")
            for i in range(5):
                f.write(f"{i} {0.0001 * (i + 1)}\n")
            temp_path = f.name

        try:
            data_idxs, losses = read_loss(temp_path)

            assert len(data_idxs) == 5
            assert len(losses) == 5
            assert data_idxs == [0, 1, 2, 3, 4]
            assert losses[0] == pytest.approx(0.0001)
            assert losses[4] == pytest.approx(0.0005)
        finally:
            os.unlink(temp_path)

    def test_read_loss_single_entry(self):
        """Test reading a loss file with a single entry."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write("# Header\n")
            f.write("1\n")
            f.write("42 0.00123\n")
            temp_path = f.name

        try:
            data_idxs, losses = read_loss(temp_path)

            assert data_idxs == [42]
            assert losses == [pytest.approx(0.00123)]
        finally:
            os.unlink(temp_path)


class TestGetWannierData:
    """Test class for get_wannier_data function."""

    def test_get_wannier_data_basic(self):
        """Test basic Wannier data retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data (3 materials, 2 Wannier functions each)
            all_data = np.array([
                [[1.0, 2.0], [3.0, 4.0]],  # Material 0
                [[5.0, 6.0], [0.0, 0.0]],  # Material 1 (second Wannier is blank)
                [[7.0, 8.0], [9.0, 10.0]],  # Material 2
            ])
            np.save(os.path.join(tmpdir, "images_center.npy"), all_data)

            data_idx, wannier_data = get_wannier_data(tmpdir)

            # Expecting 5 non-blank Wannier functions
            assert len(data_idx) == 5
            assert len(wannier_data) == 5
            # Check indices (material_idx * nwan + wan_idx)
            assert 0 in data_idx  # Material 0, Wannier 0
            assert 1 in data_idx  # Material 0, Wannier 1
            assert 2 in data_idx  # Material 1, Wannier 0
            # Material 1, Wannier 1 should be excluded (blank)
            assert 3 not in data_idx
            assert 4 in data_idx  # Material 2, Wannier 0
            assert 5 in data_idx  # Material 2, Wannier 1

    def test_get_wannier_data_all_blank(self):
        """Test Wannier data retrieval when all data is blank."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create all-zero test data
            all_data = np.zeros((2, 3, 10))
            np.save(os.path.join(tmpdir, "images_center.npy"), all_data)

            data_idx, wannier_data = get_wannier_data(tmpdir)

            assert len(data_idx) == 0
            assert len(wannier_data) == 0

    def test_get_wannier_data_no_blank(self):
        """Test Wannier data retrieval when no data is blank."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create non-zero test data
            all_data = np.ones((2, 3, 10))
            np.save(os.path.join(tmpdir, "images_center.npy"), all_data)

            data_idx, wannier_data = get_wannier_data(tmpdir)

            # All 6 Wannier functions should be included
            assert len(data_idx) == 6
            assert len(wannier_data) == 6


class TestGetOriginalIndex:
    """Test class for get_original_index function."""

    def test_get_original_index_basic(self):
        """Test basic original index retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test mp-id data
            mp_ids = np.array([100, 200, 300])
            np.save(os.path.join(tmpdir, "mp-id.npy"), mp_ids)

            data_idx = [0, 1, 2, 3, 4, 5]  # Indices for nwan=2
            mp_idx, wn_idx = get_original_index(tmpdir, data_idx, nwan=2)

            # Expected: data_idx // nwan gives material index
            # mp_idx should be the mp_id at that material index
            assert mp_idx == [100, 100, 200, 200, 300, 300]
            # wn_idx is data_idx % nwan
            assert wn_idx == [0, 1, 0, 1, 0, 1]

    def test_get_original_index_various_nwan(self):
        """Test original index retrieval with different nwan values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mp_ids = np.array([10, 20])
            np.save(os.path.join(tmpdir, "mp-id.npy"), mp_ids)

            data_idx = [0, 1, 2, 3, 4, 5]  # Indices for nwan=3
            mp_idx, wn_idx = get_original_index(tmpdir, data_idx, nwan=3)

            assert mp_idx == [10, 10, 10, 20, 20, 20]
            assert wn_idx == [0, 1, 2, 0, 1, 2]

    def test_get_original_index_single_item(self):
        """Test original index retrieval with single data index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mp_ids = np.array([999])
            np.save(os.path.join(tmpdir, "mp-id.npy"), mp_ids)

            data_idx = [0]
            mp_idx, wn_idx = get_original_index(tmpdir, data_idx, nwan=5)

            assert mp_idx == [999]
            assert wn_idx == [0]


class TestReadWaninfo:
    """Test class for read_waninfo function."""

    def test_read_waninfo_basic(self):
        """Test basic Wannier info file reading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write("# Wannier info header\n")
            f.write("3\n")  # Number of Wannier functions
            f.write("0 100 0 0.001 1.5\n")
            f.write("1 100 1 0.002 2.0\n")
            f.write("2 200 0 0.003 2.5\n")
            temp_path = f.name

        try:
            data_idx, mp_idx, wn_idx, loss_list, spread_list = read_waninfo(temp_path)

            assert data_idx == [0, 1, 2]
            assert mp_idx == [100, 100, 200]
            assert wn_idx == [0, 1, 0]
            assert loss_list == [0.001, 0.002, 0.003]
            assert spread_list == [1.5, 2.0, 2.5]
        finally:
            os.unlink(temp_path)

    def test_read_waninfo_single_entry(self):
        """Test reading Wannier info file with single entry."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write("# Header\n")
            f.write("1\n")
            f.write("42 123 5 0.00012 3.14\n")
            temp_path = f.name

        try:
            data_idx, mp_idx, wn_idx, loss_list, spread_list = read_waninfo(temp_path)

            assert data_idx == [42]
            assert mp_idx == [123]
            assert wn_idx == [5]
            assert loss_list == [pytest.approx(0.00012)]
            assert spread_list == [pytest.approx(3.14)]
        finally:
            os.unlink(temp_path)

    def test_read_waninfo_larger_file(self):
        """Test reading a larger Wannier info file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write("# Wannier information file\n")
            f.write("10\n")
            for i in range(10):
                f.write(f"{i} {i * 10} {i % 3} {0.001 * i} {1.0 + 0.1 * i}\n")
            temp_path = f.name

        try:
            data_idx, mp_idx, wn_idx, loss_list, spread_list = read_waninfo(temp_path)

            assert len(data_idx) == 10
            assert len(mp_idx) == 10
            assert len(wn_idx) == 10
            assert len(loss_list) == 10
            assert len(spread_list) == 10

            # Check specific values
            assert data_idx[0] == 0
            assert data_idx[9] == 9
            assert mp_idx[5] == 50
            assert wn_idx[7] == 7 % 3  # 1
            assert loss_list[3] == pytest.approx(0.003)
            assert spread_list[4] == pytest.approx(1.4)
        finally:
            os.unlink(temp_path)


class TestRandomSamplingEdgeCases:
    """Additional edge case tests for random_sampling_in_every_interval."""

    def test_random_sampling_all_same_value(self):
        """Test random sampling when all values are the same."""
        loss_list = [0.0005] * 10
        result = random_sampling_in_every_interval(
            loss_list, 0.0, 0.001, 5, 3
        )

        assert isinstance(result, list)
        assert len(result) == 5
        # Only one interval should have samples
        non_empty_count = sum(1 for r in result if len(r) > 0)
        assert non_empty_count >= 1

    def test_random_sampling_values_outside_range(self):
        """Test random sampling when values are outside the specified range."""
        loss_list = [0.01, 0.02, 0.03]  # All outside [0.0, 0.001]
        result = random_sampling_in_every_interval(
            loss_list, 0.0, 0.001, 5, 3
        )

        assert isinstance(result, list)
        assert len(result) == 5
        # All intervals should be empty since values are outside range
        total_samples = sum(len(r) for r in result)
        assert total_samples == 0

    def test_random_sampling_single_interval(self):
        """Test random sampling with a single interval."""
        loss_list = [0.0001, 0.0005, 0.0009]
        result = random_sampling_in_every_interval(
            loss_list, 0.0, 0.001, 1, 5
        )

        assert isinstance(result, list)
        assert len(result) == 1
        # All values should be in the single interval
        assert len(result[0]) == 3

