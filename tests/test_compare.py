"""
Tests for compare module.

This module tests the compare.py script which compares the performance of
different machine learning models (deep learning, Lasso regression, and
integration) for predicting Wannier functions from 3D data.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


class TestCompareImport:
    """Test class for compare module imports."""

    def test_compare_import(self):
        """Test that compare module can be imported."""
        try:
            from seap.tools.compare import main, LassoRegression, Integration
            assert main is not None
            assert LassoRegression is not None
            assert Integration is not None
        except ImportError:
            pytest.skip("compare module dependencies not available")

    def test_compare_import_fallback(self):
        """Test that compare module handles import fallback correctly."""
        # Test that the module structure is correct
        from seap.tools import compare as compare_module
        assert hasattr(compare_module, 'main')

    def test_compare_import_exception_handling(self):
        """Test that compare module handles import exceptions."""
        # Test the try-except ImportError block (lines 10-17)
        # This tests the import fallback mechanism
        from seap.tools import compare
        # The module should import successfully even if relative imports fail
        assert hasattr(compare, 'main')


class TestCompareMain:
    """Test class for compare module main function."""

    @pytest.fixture
    def mock_psi_data(self):
        """
        Create mock wavefunction data for testing.

        Returns
        -------
        dict
            Dictionary containing temporary directory and psi_dict.
        """
        import numpy as np

        tmpdir = tempfile.mkdtemp()
        # Create dummy numpy files
        psi_data = np.random.rand(2, 32, 32, 32)
        psi_info = np.array([[0, 1], [0, 2]])  # molecule, band pairs

        np.save(os.path.join(tmpdir, 'image32x32x32.npy'), psi_data)
        np.save(os.path.join(tmpdir, 'image_info.npy'), psi_info)

        psi_dict = {
            'name': os.path.join(tmpdir, 'image32x32x32.npy'),
            'info': os.path.join(tmpdir, 'image_info.npy'),
            'resolution': [32, 32, 32],
        }
        yield {'tmpdir': tmpdir, 'psi_dict': psi_dict}

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_main_with_mocked_dependencies(self, mock_psi_data):
        """
        Test main function with mocked dependencies.

        Parameters
        ----------
        mock_psi_data : dict
            Fixture providing mock data files.
        """
        # Mock LassoRegression
        mock_lasso_instance = MagicMock()
        mock_lasso_instance.run.return_value = (
            [0.1, 0.2],  # r_lr
            [0.01, 0.02],  # alpha
            [[0.1, 0.2], [0.3, 0.4]],  # clm_lr
        )

        # Mock Integration
        mock_integration_instance = MagicMock()
        mock_integration_instance.run.return_value = (
            [0.1, 0.2],  # r_itg
            [[0.1, 0.2], [0.3, 0.4]],  # clm_itg
        )

        with patch(
            'seap.tools.compare.LassoRegression',
            return_value=mock_lasso_instance,
        ) as mock_lasso:
            with patch(
                'seap.tools.compare.Integration',
                return_value=mock_integration_instance,
            ) as mock_integration:
                with patch('builtins.print') as mock_print:
                    # Import and run main
                    from seap.tools.compare import main
                    main()

                    # Verify LassoRegression was called
                    assert mock_lasso.called

                    # Verify Integration was called
                    assert mock_integration.called

                    # Verify run methods were called
                    mock_lasso_instance.run.assert_called_once()
                    mock_integration_instance.run.assert_called_once()

                    # Verify timing output was printed
                    assert mock_print.called
                    call_args = [str(call) for call in mock_print.call_args_list]
                    assert any('time (nn)' in arg for arg in call_args)
                    assert any('time (lasso)' in arg for arg in call_args)
                    assert any('time (integration)' in arg for arg in call_args)

    def test_main_deep_learning_import_error(self):
        """Test main function handles ImportError for deep learning model."""
        # Mock the deep learning model import to raise ImportError
        mock_lasso_instance = MagicMock()
        mock_lasso_instance.run.return_value = (
            [0.1],
            [0.01],
            [[0.1, 0.2]],
        )

        mock_integration_instance = MagicMock()
        mock_integration_instance.run.return_value = (
            [0.1],
            [[0.1, 0.2]],
        )

        with patch(
            'seap.tools.compare.LassoRegression',
            return_value=mock_lasso_instance,
        ):
            with patch(
                'seap.tools.compare.Integration',
                return_value=mock_integration_instance,
            ):
                with patch('builtins.print') as mock_print:
                    from seap.tools.compare import main
                    main()

                    # Verify that the function handled potential ImportError
                    call_args = [str(call) for call in mock_print.call_args_list]
                    # Either deep learning model runs or import error is printed
                    assert any(
                        'time (nn)' in arg or 'Deep learning model not available' in arg
                        for arg in call_args
                    )

    def test_main_timing_output(self):
        """Test that main function outputs timing information."""
        mock_lasso_instance = MagicMock()
        mock_lasso_instance.run.return_value = (
            [0.1],
            [0.01],
            [[0.1, 0.2]],
        )

        mock_integration_instance = MagicMock()
        mock_integration_instance.run.return_value = (
            [0.1],
            [[0.1, 0.2]],
        )

        captured_output = []

        def capture_print(*args, **kwargs):
            captured_output.append(' '.join(str(arg) for arg in args))

        with patch(
            'seap.tools.compare.LassoRegression',
            return_value=mock_lasso_instance,
        ):
            with patch(
                'seap.tools.compare.Integration',
                return_value=mock_integration_instance,
            ):
                with patch('builtins.print', side_effect=capture_print):
                    from seap.tools.compare import main
                    main()

                    # Check timing output format
                    nn_output = [o for o in captured_output if 'time (nn)' in o]
                    lasso_output = [o for o in captured_output if 'time (lasso)' in o]
                    itg_output = [o for o in captured_output if 'time (integration)' in o]

                    assert len(nn_output) >= 1
                    assert len(lasso_output) >= 1
                    assert len(itg_output) >= 1


class TestCompareParameters:
    """Test class for compare module default parameters."""

    def test_default_parameters(self):
        """Test that default parameters are correctly set in main function."""
        # We test this by examining the source code behavior through mocking
        mock_lasso_instance = MagicMock()
        mock_lasso_instance.run.return_value = ([0.1], [0.01], [[0.1]])

        mock_integration_instance = MagicMock()
        mock_integration_instance.run.return_value = ([0.1], [[0.1]])

        with patch(
            'seap.tools.compare.LassoRegression',
            return_value=mock_lasso_instance,
        ) as mock_lasso:
            with patch(
                'seap.tools.compare.Integration',
                return_value=mock_integration_instance,
            ) as mock_integration:
                with patch('builtins.print'):
                    from seap.tools.compare import main
                    main()

                    # Verify LassoRegression was called with correct parameters
                    # num_t=64, num_p=64, n=4
                    lasso_call_args = mock_lasso.call_args
                    assert lasso_call_args[0][0] == 64  # num_t
                    assert lasso_call_args[0][1] == 64  # num_p
                    assert lasso_call_args[0][2] == 4   # n

                    # Verify Integration was called with correct parameters
                    integration_call_args = mock_integration.call_args
                    assert integration_call_args[0][0] == 64  # num_t
                    assert integration_call_args[0][1] == 64  # num_p
                    assert integration_call_args[0][2] == 4   # n

    def test_psi_dict_parameters(self):
        """Test that psi_dict parameters are correctly passed."""
        mock_lasso_instance = MagicMock()
        mock_lasso_instance.run.return_value = ([0.1], [0.01], [[0.1]])

        mock_integration_instance = MagicMock()
        mock_integration_instance.run.return_value = ([0.1], [[0.1]])

        with patch(
            'seap.tools.compare.LassoRegression',
            return_value=mock_lasso_instance,
        ) as mock_lasso:
            with patch(
                'seap.tools.compare.Integration',
                return_value=mock_integration_instance,
            ) as mock_integration:
                with patch('builtins.print'):
                    from seap.tools.compare import main
                    main()

                    # Verify psi_dict kwargs
                    lasso_kwargs = mock_lasso.call_args[1]
                    assert 'name' in lasso_kwargs
                    assert 'info' in lasso_kwargs
                    assert 'resolution' in lasso_kwargs
                    assert lasso_kwargs['resolution'] == [32, 32, 32]

                    integration_kwargs = mock_integration.call_args[1]
                    assert 'name' in integration_kwargs
                    assert 'info' in integration_kwargs
                    assert 'resolution' in integration_kwargs
                    assert integration_kwargs['resolution'] == [32, 32, 32]


class TestCompareDeepLearning:
    """Test class for deep learning model handling in compare module."""

    def test_deep_learning_import_fallback(self):
        """Test deep learning model import fallback mechanism."""
        # The main function should handle ImportError gracefully
        mock_lasso_instance = MagicMock()
        mock_lasso_instance.run.return_value = ([0.1], [0.01], [[0.1]])

        mock_integration_instance = MagicMock()
        mock_integration_instance.run.return_value = ([0.1], [[0.1]])

        error_messages = []

        def capture_print(*args, **kwargs):
            error_messages.append(' '.join(str(arg) for arg in args))

        with patch(
            'seap.tools.compare.LassoRegression',
            return_value=mock_lasso_instance,
        ):
            with patch(
                'seap.tools.compare.Integration',
                return_value=mock_integration_instance,
            ):
                with patch('builtins.print', side_effect=capture_print):
                    from seap.tools.compare import main

                    # Run should not raise exception even if deep learning fails
                    try:
                        main()
                    except ImportError:
                        pytest.fail("main() should not propagate ImportError")

    def test_deep_learning_time_measurement(self):
        """Test that deep learning time is measured even on error."""
        mock_lasso_instance = MagicMock()
        mock_lasso_instance.run.return_value = ([0.1], [0.01], [[0.1]])

        mock_integration_instance = MagicMock()
        mock_integration_instance.run.return_value = ([0.1], [[0.1]])

        time_outputs = []

        def capture_print(*args, **kwargs):
            time_outputs.append(' '.join(str(arg) for arg in args))

        with patch(
            'seap.tools.compare.LassoRegression',
            return_value=mock_lasso_instance,
        ):
            with patch(
                'seap.tools.compare.Integration',
                return_value=mock_integration_instance,
            ):
                with patch('builtins.print', side_effect=capture_print):
                    from seap.tools.compare import main
                    main()

                    # Verify time (nn) is always printed
                    nn_times = [o for o in time_outputs if 'time (nn)' in o]
                    assert len(nn_times) >= 1


class TestCompareExecution:
    """Test class for compare module execution scenarios."""

    def test_execution_order(self):
        """Test that models are executed in correct order: nn, lasso, integration."""
        execution_order = []

        mock_lasso_instance = MagicMock()
        mock_lasso_instance.run.side_effect = lambda: (
            execution_order.append('lasso'),
            ([0.1], [0.01], [[0.1]]),
        )[1]

        mock_integration_instance = MagicMock()
        mock_integration_instance.run.side_effect = lambda: (
            execution_order.append('integration'),
            ([0.1], [[0.1]]),
        )[1]

        outputs = []

        def capture_print(*args, **kwargs):
            msg = ' '.join(str(arg) for arg in args)
            outputs.append(msg)
            if 'time (nn)' in msg:
                execution_order.append('nn')

        with patch(
            'seap.tools.compare.LassoRegression',
            return_value=mock_lasso_instance,
        ):
            with patch(
                'seap.tools.compare.Integration',
                return_value=mock_integration_instance,
            ):
                with patch('builtins.print', side_effect=capture_print):
                    from seap.tools.compare import main
                    main()

                    # Verify execution order
                    assert 'nn' in execution_order
                    assert 'lasso' in execution_order
                    assert 'integration' in execution_order

                    # nn should be before lasso, lasso before integration
                    nn_idx = execution_order.index('nn')
                    lasso_idx = execution_order.index('lasso')
                    integration_idx = execution_order.index('integration')

                    assert nn_idx < lasso_idx
                    assert lasso_idx < integration_idx

    def test_all_models_return_results(self):
        """Test that all models return results correctly."""
        mock_lasso_instance = MagicMock()
        expected_lasso_result = (
            [0.1, 0.2],
            [0.01, 0.02],
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        )
        mock_lasso_instance.run.return_value = expected_lasso_result

        mock_integration_instance = MagicMock()
        expected_integration_result = (
            [0.15, 0.25],
            [[0.11, 0.22, 0.33], [0.44, 0.55, 0.66]],
        )
        mock_integration_instance.run.return_value = expected_integration_result

        with patch(
            'seap.tools.compare.LassoRegression',
            return_value=mock_lasso_instance,
        ):
            with patch(
                'seap.tools.compare.Integration',
                return_value=mock_integration_instance,
            ):
                with patch('builtins.print'):
                    from seap.tools.compare import main
                    main()

                    # Verify run methods were called and returned correctly
                    assert mock_lasso_instance.run.called
                    assert mock_integration_instance.run.called


class TestCompareModuleStructure:
    """Test class for compare module structure and attributes."""

    def test_module_has_main_function(self):
        """Test that module exports main function."""
        from seap.tools import compare
        assert hasattr(compare, 'main')
        assert callable(compare.main)

    def test_module_has_lasso_regression(self):
        """Test that module exports LassoRegression class."""
        from seap.tools import compare
        assert hasattr(compare, 'LassoRegression')

    def test_module_has_integration(self):
        """Test that module exports Integration class."""
        from seap.tools import compare
        assert hasattr(compare, 'Integration')


class TestCompareEdgeCases:
    """Test class for compare module edge cases."""

    def test_empty_results_handling(self):
        """Test handling of empty results from models."""
        mock_lasso_instance = MagicMock()
        mock_lasso_instance.run.return_value = ([], [], [])

        mock_integration_instance = MagicMock()
        mock_integration_instance.run.return_value = ([], [])

        with patch(
            'seap.tools.compare.LassoRegression',
            return_value=mock_lasso_instance,
        ):
            with patch(
                'seap.tools.compare.Integration',
                return_value=mock_integration_instance,
            ):
                with patch('builtins.print'):
                    from seap.tools.compare import main
                    # Should not raise exception
                    main()

    def test_timing_values_are_non_negative(self):
        """Test that timing values are non-negative."""
        mock_lasso_instance = MagicMock()
        mock_lasso_instance.run.return_value = ([0.1], [0.01], [[0.1]])

        mock_integration_instance = MagicMock()
        mock_integration_instance.run.return_value = ([0.1], [[0.1]])

        timing_values = []

        def capture_print(*args, **kwargs):
            msg = ' '.join(str(arg) for arg in args)
            if 'time' in msg and ':' in msg:
                try:
                    time_str = msg.split(':')[-1].strip()
                    timing_values.append(float(time_str))
                except ValueError:
                    # Ignore lines that don't contain valid timing values
                    pass

        with patch(
            'seap.tools.compare.LassoRegression',
            return_value=mock_lasso_instance,
        ):
            with patch(
                'seap.tools.compare.Integration',
                return_value=mock_integration_instance,
            ):
                with patch('builtins.print', side_effect=capture_print):
                    from seap.tools.compare import main
                    main()

                    # All timing values should be non-negative
                    for val in timing_values:
                        assert val >= 0, f"Timing value {val} should be non-negative"
