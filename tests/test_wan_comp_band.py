"""
Tests for wan_comp_band module.
"""

import os
import tempfile

import numpy as np
from unittest.mock import patch

from seap.tools import wan_comp_band


class TestWanCompBandMain:
    """Test class for wan_comp_band main function."""

    def test_main_with_mocked_dependencies(self):
        """Test main function with all dependencies mocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Mock the external functions
                mock_x = np.linspace(0, 1, 10)
                mock_y = np.random.rand(10, 5)  # 10 k-points, 5 bands

                with patch.object(
                    wan_comp_band, 'get_band_data'
                ) as mock_get_band:
                    mock_get_band.return_value = (mock_x, mock_y)

                    with patch.object(
                        wan_comp_band, 'get_ef_from_scfout'
                    ) as mock_get_ef:
                        mock_get_ef.return_value = 5.0

                        with patch.object(
                            wan_comp_band, 'get_klabel'
                        ) as mock_get_klabel:
                            mock_get_klabel.return_value = (
                                [0.0, 0.5, 1.0],
                                ['G', 'X', 'M'],
                            )

                            with patch(
                                'seap.tools.wan_comp_band.plt'
                            ) as mock_plt:
                                # Run main
                                wan_comp_band.main()

                                # Verify plot functions were called
                                mock_plt.rcParams.__setitem__.assert_called()
                                mock_plt.title.assert_called_once()
                                assert mock_plt.plot.call_count == 2
                                mock_plt.ylabel.assert_called_once()
                                mock_plt.xticks.assert_called_once()
                                mock_plt.xlim.assert_called_once()
                                mock_plt.ylim.assert_called_once()
                                mock_plt.vlines.assert_called_once()
                                assert mock_plt.savefig.call_count == 2
            finally:
                os.chdir(original_cwd)

    def test_main_plot_parameters(self):
        """Test that main function uses correct plot parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                mock_x = np.linspace(0, 1, 10)
                mock_y = np.ones((10, 3)) * 10.0  # Constant energy for easy verification

                with patch.object(
                    wan_comp_band, 'get_band_data'
                ) as mock_get_band:
                    mock_get_band.return_value = (mock_x, mock_y)

                    with patch.object(
                        wan_comp_band, 'get_ef_from_scfout'
                    ) as mock_get_ef:
                        mock_get_ef.return_value = 5.0  # E_F = 5.0

                        with patch.object(
                            wan_comp_band, 'get_klabel'
                        ) as mock_get_klabel:
                            mock_get_klabel.return_value = (
                                [0.0, 0.5, 1.0],
                                ['G', 'X', 'M'],
                            )

                            with patch(
                                'seap.tools.wan_comp_band.plt'
                            ) as mock_plt:
                                wan_comp_band.main()

                                # Check xlim was set to [0, 1]
                                mock_plt.xlim.assert_called_with([0, 1])

                                # Check title
                                mock_plt.title.assert_called_with(
                                    "Red=scdm, Black=mycalc"
                                )

                                # Check ylabel
                                mock_plt.ylabel.assert_called_with(
                                    r"$E - E_{\mathrm{F}}$[eV]"
                                )

                                # Check savefig was called with correct filenames
                                savefig_calls = mock_plt.savefig.call_args_list
                                assert len(savefig_calls) == 2
                                assert './wan_band_compare.png' in str(
                                    savefig_calls[0]
                                )
                                assert './wan_band_compare.eps' in str(
                                    savefig_calls[1]
                                )
            finally:
                os.chdir(original_cwd)

    def test_main_band_data_calls(self):
        """Test that get_band_data is called with correct filenames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                mock_x = np.linspace(0, 1, 5)
                mock_y = np.random.rand(5, 2)

                with patch.object(
                    wan_comp_band, 'get_band_data'
                ) as mock_get_band:
                    mock_get_band.return_value = (mock_x, mock_y)

                    with patch.object(
                        wan_comp_band, 'get_ef_from_scfout'
                    ) as mock_get_ef:
                        mock_get_ef.return_value = 0.0

                        with patch.object(
                            wan_comp_band, 'get_klabel'
                        ) as mock_get_klabel:
                            mock_get_klabel.return_value = ([0, 1], ['G', 'M'])

                            with patch('seap.tools.wan_comp_band.plt'):
                                wan_comp_band.main()

                                # Verify get_band_data was called with correct files
                                calls = mock_get_band.call_args_list
                                assert len(calls) == 2
                                assert calls[0][0][0] == 'scdm_band.dat'
                                assert calls[1][0][0] == 'my_band.dat'
            finally:
                os.chdir(original_cwd)

    def test_main_fermi_energy_subtraction(self):
        """Test that Fermi energy is correctly subtracted from band energies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                mock_x = np.array([0.0, 0.5, 1.0])
                mock_y = np.array([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
                fermi_energy = 5.0

                with patch.object(
                    wan_comp_band, 'get_band_data'
                ) as mock_get_band:
                    mock_get_band.return_value = (mock_x, mock_y)

                    with patch.object(
                        wan_comp_band, 'get_ef_from_scfout'
                    ) as mock_get_ef:
                        mock_get_ef.return_value = fermi_energy

                        with patch.object(
                            wan_comp_band, 'get_klabel'
                        ) as mock_get_klabel:
                            mock_get_klabel.return_value = ([0, 1], ['G', 'M'])

                            with patch(
                                'seap.tools.wan_comp_band.plt'
                            ) as mock_plt:
                                wan_comp_band.main()

                                # Check that plot was called with shifted energies
                                plot_calls = mock_plt.plot.call_args_list

                                # First call (SCDM bands)
                                y_plotted_scdm = plot_calls[0][0][1]
                                expected_y = mock_y - fermi_energy
                                np.testing.assert_array_almost_equal(
                                    y_plotted_scdm, expected_y
                                )

                                # Second call (my bands)
                                y_plotted_my = plot_calls[1][0][1]
                                np.testing.assert_array_almost_equal(
                                    y_plotted_my, expected_y
                                )
            finally:
                os.chdir(original_cwd)

    def test_main_ylim_calculation(self):
        """Test that y-axis limits are calculated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                mock_x = np.array([0.0, 0.5, 1.0])
                # y values from 0 to 10
                mock_y = np.array([[0.0, 10.0], [2.0, 8.0], [0.0, 10.0]])
                fermi_energy = 5.0  # After subtraction: -5 to 5

                with patch.object(
                    wan_comp_band, 'get_band_data'
                ) as mock_get_band:
                    mock_get_band.return_value = (mock_x, mock_y)

                    with patch.object(
                        wan_comp_band, 'get_ef_from_scfout'
                    ) as mock_get_ef:
                        mock_get_ef.return_value = fermi_energy

                        with patch.object(
                            wan_comp_band, 'get_klabel'
                        ) as mock_get_klabel:
                            mock_get_klabel.return_value = ([0, 1], ['G', 'M'])

                            with patch(
                                'seap.tools.wan_comp_band.plt'
                            ) as mock_plt:
                                wan_comp_band.main()

                                # Check ylim calculation
                                ylim_call = mock_plt.ylim.call_args[0][0]
                                y_shifted = mock_y - fermi_energy
                                y_min = np.min(y_shifted)  # -5.0
                                y_max = np.max(y_shifted)  # 5.0
                                margin = 0.05 * (y_max - y_min)  # 0.5

                                expected_py_min = y_min - margin
                                expected_py_max = y_max + margin

                                np.testing.assert_almost_equal(
                                    ylim_call[0], expected_py_min
                                )
                                np.testing.assert_almost_equal(
                                    ylim_call[1], expected_py_max
                                )
            finally:
                os.chdir(original_cwd)

    def test_main_vlines_at_klabels(self):
        """Test that vertical lines are drawn at k-label positions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                mock_x = np.linspace(0, 1, 10)
                mock_y = np.random.rand(10, 3)
                klabel_positions = [0.0, 0.33, 0.67, 1.0]
                klabel_names = ['G', 'X', 'M', 'G']

                with patch.object(
                    wan_comp_band, 'get_band_data'
                ) as mock_get_band:
                    mock_get_band.return_value = (mock_x, mock_y)

                    with patch.object(
                        wan_comp_band, 'get_ef_from_scfout'
                    ) as mock_get_ef:
                        mock_get_ef.return_value = 0.0

                        with patch.object(
                            wan_comp_band, 'get_klabel'
                        ) as mock_get_klabel:
                            mock_get_klabel.return_value = (
                                klabel_positions,
                                klabel_names,
                            )

                            with patch(
                                'seap.tools.wan_comp_band.plt'
                            ) as mock_plt:
                                wan_comp_band.main()

                                # Check vlines was called with klabel positions
                                vlines_call = mock_plt.vlines.call_args
                                np.testing.assert_array_equal(
                                    vlines_call[0][0], klabel_positions
                                )
            finally:
                os.chdir(original_cwd)


class TestWanCompBandPlotColors:
    """Test class for plot color and style settings."""

    def test_scdm_band_color_red(self):
        """Test that SCDM band is plotted in red."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                mock_x = np.linspace(0, 1, 5)
                mock_y = np.random.rand(5, 2)

                with patch.object(
                    wan_comp_band, 'get_band_data'
                ) as mock_get_band:
                    mock_get_band.return_value = (mock_x, mock_y)

                    with patch.object(
                        wan_comp_band, 'get_ef_from_scfout'
                    ) as mock_get_ef:
                        mock_get_ef.return_value = 0.0

                        with patch.object(
                            wan_comp_band, 'get_klabel'
                        ) as mock_get_klabel:
                            mock_get_klabel.return_value = ([0, 1], ['G', 'M'])

                            with patch(
                                'seap.tools.wan_comp_band.plt'
                            ) as mock_plt:
                                wan_comp_band.main()

                                # First plot call should be SCDM (red)
                                first_plot_call = mock_plt.plot.call_args_list[0]
                                assert first_plot_call[1]['c'] == 'r'
                                assert first_plot_call[1]['lw'] == 1
            finally:
                os.chdir(original_cwd)

    def test_my_band_color_black(self):
        """Test that custom band is plotted in black."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                mock_x = np.linspace(0, 1, 5)
                mock_y = np.random.rand(5, 2)

                with patch.object(
                    wan_comp_band, 'get_band_data'
                ) as mock_get_band:
                    mock_get_band.return_value = (mock_x, mock_y)

                    with patch.object(
                        wan_comp_band, 'get_ef_from_scfout'
                    ) as mock_get_ef:
                        mock_get_ef.return_value = 0.0

                        with patch.object(
                            wan_comp_band, 'get_klabel'
                        ) as mock_get_klabel:
                            mock_get_klabel.return_value = ([0, 1], ['G', 'M'])

                            with patch(
                                'seap.tools.wan_comp_band.plt'
                            ) as mock_plt:
                                wan_comp_band.main()

                                # Second plot call should be custom (black)
                                second_plot_call = mock_plt.plot.call_args_list[1]
                                assert second_plot_call[1]['c'] == 'k'
                                assert second_plot_call[1]['lw'] == 0.5
            finally:
                os.chdir(original_cwd)


class TestWanCompBandEdgeCases:
    """Test class for edge cases and error handling."""

    def test_main_with_empty_band_data(self):
        """Test main function with empty band data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Empty arrays
                mock_x = np.array([])
                mock_y = np.array([])

                with patch.object(
                    wan_comp_band, 'get_band_data'
                ) as mock_get_band:
                    mock_get_band.return_value = (mock_x, mock_y)

                    with patch.object(
                        wan_comp_band, 'get_ef_from_scfout'
                    ) as mock_get_ef:
                        mock_get_ef.return_value = 0.0

                        with patch.object(
                            wan_comp_band, 'get_klabel'
                        ) as mock_get_klabel:
                            mock_get_klabel.return_value = ([], [])

                            with patch(
                                'seap.tools.wan_comp_band.plt'
                            ) as mock_plt:
                                # This might raise an error with empty data
                                # but we want to verify the function handles it
                                try:
                                    wan_comp_band.main()
                                except (ValueError, IndexError):
                                    # Expected behavior with empty data
                                    pass
            finally:
                os.chdir(original_cwd)

    def test_main_with_single_band(self):
        """Test main function with single band data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                mock_x = np.array([0.0, 0.5, 1.0])
                mock_y = np.array([[1.0], [2.0], [3.0]])  # Single band

                with patch.object(
                    wan_comp_band, 'get_band_data'
                ) as mock_get_band:
                    mock_get_band.return_value = (mock_x, mock_y)

                    with patch.object(
                        wan_comp_band, 'get_ef_from_scfout'
                    ) as mock_get_ef:
                        mock_get_ef.return_value = 0.0

                        with patch.object(
                            wan_comp_band, 'get_klabel'
                        ) as mock_get_klabel:
                            mock_get_klabel.return_value = ([0, 1], ['G', 'M'])

                            with patch(
                                'seap.tools.wan_comp_band.plt'
                            ) as mock_plt:
                                wan_comp_band.main()

                                # Should complete without error
                                assert mock_plt.savefig.call_count == 2
            finally:
                os.chdir(original_cwd)

    def test_main_with_negative_fermi_energy(self):
        """Test main function with negative Fermi energy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                mock_x = np.linspace(0, 1, 5)
                mock_y = np.array([[-5.0, 5.0]] * 5)
                fermi_energy = -10.0  # Negative Fermi energy

                with patch.object(
                    wan_comp_band, 'get_band_data'
                ) as mock_get_band:
                    mock_get_band.return_value = (mock_x, mock_y)

                    with patch.object(
                        wan_comp_band, 'get_ef_from_scfout'
                    ) as mock_get_ef:
                        mock_get_ef.return_value = fermi_energy

                        with patch.object(
                            wan_comp_band, 'get_klabel'
                        ) as mock_get_klabel:
                            mock_get_klabel.return_value = ([0, 1], ['G', 'M'])

                            with patch(
                                'seap.tools.wan_comp_band.plt'
                            ) as mock_plt:
                                wan_comp_band.main()

                                # Check that energy shift is applied correctly
                                plot_calls = mock_plt.plot.call_args_list
                                y_plotted = plot_calls[1][0][1]
                                expected_y = mock_y - fermi_energy
                                np.testing.assert_array_almost_equal(
                                    y_plotted, expected_y
                                )
            finally:
                os.chdir(original_cwd)


class TestWanCompBandFallbackFunctions:
    """Test class for fallback functions when cif2qewan is not available."""

    def test_fallback_get_band_data_exists(self):
        """Test that fallback get_band_data function exists in module."""
        # The function should exist in the module namespace
        assert hasattr(wan_comp_band, 'get_band_data')

    def test_fallback_get_ef_from_scfout_exists(self):
        """Test that fallback get_ef_from_scfout function exists in module."""
        assert hasattr(wan_comp_band, 'get_ef_from_scfout')

    def test_fallback_get_klabel_exists(self):
        """Test that fallback get_klabel function exists in module."""
        assert hasattr(wan_comp_band, 'get_klabel')


class TestWanCompBandIntegration:
    """Integration tests for wan_comp_band module."""

    def test_full_workflow_mock(self):
        """Test complete workflow with mocked dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Create realistic mock data
                num_kpoints = 50
                num_bands = 4
                mock_x = np.linspace(0, 1, num_kpoints)
                mock_y_scdm = np.sin(np.outer(mock_x, np.arange(1, num_bands + 1) * np.pi))
                mock_y_my = mock_y_scdm + 0.1 * np.random.randn(num_kpoints, num_bands)
                fermi_energy = 0.0
                klabel_pos = [0.0, 0.25, 0.5, 0.75, 1.0]
                klabel_names = ['Γ', 'X', 'M', 'Γ', 'R']

                call_count = [0]

                def mock_get_band_data(filename):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return mock_x, mock_y_scdm
                    return mock_x, mock_y_my

                with patch.object(
                    wan_comp_band, 'get_band_data', side_effect=mock_get_band_data
                ):
                    with patch.object(
                        wan_comp_band, 'get_ef_from_scfout', return_value=fermi_energy
                    ):
                        with patch.object(
                            wan_comp_band,
                            'get_klabel',
                            return_value=(klabel_pos, klabel_names),
                        ):
                            with patch(
                                'seap.tools.wan_comp_band.plt'
                            ) as mock_plt:
                                wan_comp_band.main()

                                # Verify all expected calls were made
                                assert mock_plt.title.called
                                assert mock_plt.plot.call_count == 2
                                assert mock_plt.ylabel.called
                                assert mock_plt.xticks.called
                                assert mock_plt.xlim.called
                                assert mock_plt.ylim.called
                                assert mock_plt.vlines.called
                                assert mock_plt.savefig.call_count == 2
            finally:
                os.chdir(original_cwd)


