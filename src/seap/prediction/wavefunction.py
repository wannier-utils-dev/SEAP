import numpy as np
from scipy import interpolate as itp

# This module provides a class for handling gamma wavefunctions.
# The class allows for interpolation of wavefunction data and retrieval of spherical data.
# The class also provides methods for normalizing the wavefunction and converting to Cartesian coordinates.

class GammaWaveFunction:
    def __init__(self, data, nx, ny, nz):
        """
        Initialize the GammaWaveFunction object.

        Parameters
        ----------
        data : np.ndarray
            The wavefunction data to be reshaped and stored.
        nx, ny, nz : int
            The dimensions of the wavefunction data grid.
        """
        # Reshape the input data into a 4D array with dimensions (npsi, nx, ny, nz)
        self.psi_org = data.reshape(-1, nx, ny, nz)
        # Create interpolators for the wavefunction data
        self.psi_itp = self._linear_interpolator()
        # Set the center of the wavefunction in Cartesian coordinates
        self.center = np.array([0.5, 0.5, 0.5])

    def _linear_interpolator(self):
        """
        Create linear interpolators for the wavefunction data.

        Returns
        -------
        list of RegularGridInterpolator
            A list of interpolator functions for each wavefunction.
        """
        # Extract the shape of the wavefunction data
        npsi, nx, ny, nz = self.psi_org.shape

        # Generate linearly spaced grid points for interpolation
        x = np.linspace(0, 1, nx, endpoint=False) + 1.0/nx/2.0
        y = np.linspace(0, 1, ny, endpoint=False) + 1.0/ny/2.0
        z = np.linspace(0, 1, nz, endpoint=False) + 1.0/nz/2.0

        # Initialize a list to store the interpolator functions
        itpfunc = []
        for data in self.psi_org:
            # Create a RegularGridInterpolator for each wavefunction
            func = itp.RegularGridInterpolator((x, y, z), data)
            itpfunc.append(func)
       
        return itpfunc

    def _spherical2cartesian(self, r, num_t, num_p):
        """ 
        Convert spherical coordinates to Cartesian coordinates.

        Parameters
        ----------
        r : float
            Radius of the spherical surface.
        num_t : int
            Number of segments in the theta direction.
        num_p : int
            Number of segments in the phi direction.

        Returns
        -------
        tuple of np.ndarray
            Cartesian coordinates (x, y, z) of each point on the spherical surface.
        """
        # Generate linearly spaced angles for theta and phi
        ts = np.linspace(0, np.pi, num_t)
        ps = np.linspace(0, 2*np.pi, num_p)

        # Create a meshgrid for theta and phi
        theta, phi = np.meshgrid(ts, ps, indexing='ij')

        # Convert spherical coordinates to Cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi) + self.center[0]
        y = r * np.sin(theta) * np.sin(phi) + self.center[1]
        z = r * np.cos(theta) + self.center[2]

        return x, y, z

    def _normalize(self, psi, theta, phi):
        """
        Normalize the wavefunction over the spherical surface.

        Parameters
        ----------
        psi : np.ndarray
            The wavefunction values on the spherical surface.
        theta : np.ndarray
            Array of polar angles in radians.
        phi : np.ndarray
            Array of azimuthal angles in radians.

        Returns
        -------
        np.ndarray
            The normalized wavefunction S^{\\prime}(\\theta, \\phi).
        """
        # Calculate the Jacobian for the spherical coordinates
        jacob = np.sin(theta)
        # Initialize an array to store the integrated values over theta
        psi_p = np.zeros(len(phi), dtype=float)
        for ip, p in enumerate(phi):
            # Square the wavefunction values
            psi_2 = psi[:, ip] * psi[:, ip]
            # Integrate over theta using the trapezoidal rule
            psi_p[ip] = np.trapz(psi_2 * jacob, x=theta)
        # Calculate the absolute value of the radial part
        R_abs = np.sqrt(np.trapz(psi_p, x=phi))
        # Normalize the wavefunction
        S_prime = psi / R_abs
        return S_prime

    def get_spherical_data(self, r, num_t, num_p):
        """
        Retrieve the normalized spherical data from the wavefunction.

        Parameters
        ----------
        r : float
            Radius of the spherical surface.
        num_t : int
            Number of segments in the theta direction.
        num_p : int
            Number of segments in the phi direction.

        Returns
        -------
        np.ndarray
            The normalized spherical data S^{\\prime}(\\theta, \\phi) for each wavefunction.
        """
        # Generate linearly spaced angles for theta and phi
        ts = np.linspace(0, np.pi, num_t)
        ps = np.linspace(0, 2*np.pi, num_p)

        # Convert spherical coordinates to Cartesian coordinates
        x_new, y_new, z_new = self._spherical2cartesian(r, num_t, num_p)

        # Initialize a list to store the normalized spherical data
        S_prime = []
        for f in self.psi_itp:
            # Interpolate the wavefunction values at the new Cartesian coordinates
            psi = f((x_new, y_new, z_new))
            # Normalize the wavefunction and append to the list
            S_prime.append(self._normalize(psi, ts, ps))

        return np.array(S_prime)

if __name__ == '__main__':
    # Define the path to the data file
    path = '.'
    # Load the wavefunction data from a NumPy file
    data = np.load(path + '/image32x32x32.npy')
