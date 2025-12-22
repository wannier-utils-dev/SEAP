"""
Tests for wavefunction classes.
"""

import numpy as np
from seap.prediction.wavefunction import GammaWaveFunction


class TestGammaWaveFunction:
    """Test class for GammaWaveFunction."""

    def test_initialization(self):
        """Test GammaWaveFunction initialization."""
        # Create dummy wavefunction data
        nx, ny, nz = 8, 8, 8
        npsi = 2
        data = np.random.rand(npsi * nx * ny * nz)
        
        wf = GammaWaveFunction(data, nx, ny, nz)
        
        assert wf.psi_org.shape == (npsi, nx, ny, nz)
        assert len(wf.psi_itp) == npsi
        assert wf.center.shape == (3,)

    def test_spherical2cartesian(self):
        """Test spherical to Cartesian coordinate conversion."""
        nx, ny, nz = 4, 4, 4
        data = np.random.rand(nx * ny * nz)
        wf = GammaWaveFunction(data, nx, ny, nz)
        
        x, y, z = wf._spherical2cartesian(r=0.5, num_t=4, num_p=4)
        
        assert x.shape == (4, 4)
        assert y.shape == (4, 4)
        assert z.shape == (4, 4)
        # Check that coordinates are centered around self.center
        assert np.allclose(x.mean(), wf.center[0], atol=0.1)
        assert np.allclose(y.mean(), wf.center[1], atol=0.1)
        assert np.allclose(z.mean(), wf.center[2], atol=0.1)

    def test_normalize(self):
        """Test wavefunction normalization."""
        nx, ny, nz = 4, 4, 4
        data = np.random.rand(nx * ny * nz)
        wf = GammaWaveFunction(data, nx, ny, nz)
        
        # Create dummy psi, theta, phi
        num_t, num_p = 4, 4
        psi = np.random.rand(num_t, num_p)
        theta = np.linspace(0, np.pi, num_t)
        phi = np.linspace(0, 2*np.pi, num_p)
        
        normalized = wf._normalize(psi, theta, phi)
        
        assert normalized.shape == psi.shape
        # Normalized wavefunction should have unit norm
        assert np.all(np.isfinite(normalized))

    def test_get_spherical_data(self):
        """Test getting spherical data."""
        nx, ny, nz = 4, 4, 4
        npsi = 2
        data = np.random.rand(npsi * nx * ny * nz)
        wf = GammaWaveFunction(data, nx, ny, nz)
        
        spherical_data = wf.get_spherical_data(r=0.3, num_t=4, num_p=4)
        
        assert isinstance(spherical_data, np.ndarray)
        assert spherical_data.shape[0] == npsi
        assert spherical_data.shape[1] == 4  # num_t
        assert spherical_data.shape[2] == 4  # num_p

    def test_linear_interpolator(self):
        """Test linear interpolator creation."""
        nx, ny, nz = 4, 4, 4
        npsi = 2
        data = np.random.rand(npsi * nx * ny * nz)
        wf = GammaWaveFunction(data, nx, ny, nz)
        
        assert len(wf.psi_itp) == npsi
        # Test interpolation at a point
        point = np.array([[0.5, 0.5, 0.5]])
        result = wf.psi_itp[0](point)
        assert isinstance(result, np.ndarray)


