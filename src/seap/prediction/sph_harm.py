import numpy as np

# scipy changed the spherical-harmonic API in 1.15:
#   sph_harm    (deprecated): sph_harm(m, n, theta, phi)  with
#                             theta = azimuthal in [0, 2pi],
#                             phi   = polar     in [0, pi].
#   sph_harm_y  (>= 1.15):    sph_harm_y(n, m, theta, phi) with
#                             theta = polar     in [0, pi],
#                             phi   = azimuthal in [0, 2pi].
# Both the (m, n) order *and* the meaning of theta/phi are inverted.
# We expose a single helper _Ylm(l, m, phi_az, theta_pol) that always evaluates
# Y_l^m(theta_pol, phi_az) with the standard physics convention.
try:
    from scipy.special import sph_harm_y as _scipy_sph
    def _Ylm(l, m, phi_az, theta_pol):
        # sph_harm_y(n=l, m=m, theta=theta_pol, phi=phi_az)
        return _scipy_sph(l, m, theta_pol, phi_az)
except ImportError:
    from scipy.special import sph_harm as _scipy_sph
    def _Ylm(l, m, phi_az, theta_pol):
        # sph_harm(m=m, n=l, theta=phi_az, phi=theta_pol)
        return _scipy_sph(m, l, phi_az, theta_pol)


def spherical_harmonics(l, m, theta, phi):
    """
    Calculate the real or imaginary part of the spherical harmonics.

    Parameters
    ----------
    l : int
        Degree of the spherical harmonics.
    m : int
        Order of the spherical harmonics.
    theta : np.ndarray
        Array of polar angles in radians.
    phi : np.ndarray
        Array of azimuthal angles in radians.

    Returns
    -------
    np.ndarray
        The real or imaginary part of the spherical harmonics evaluated at
        the given angles.
    """
    if m > 0:
        # For positive m, return the real part multiplied by sqrt(2) and (-1)^m
        return np.sqrt(2) * (-1)**m * _Ylm(l, m, phi, theta).real
    if m < 0:
        # For negative m, return the imaginary part multiplied by sqrt(2) and (-1)^m
        return np.sqrt(2) * (-1)**m * _Ylm(l, -m, phi, theta).imag
    else:
        # For m = 0, return the real part
        return _Ylm(l, m, phi, theta).real

def quantum_number(n):
    """
    Generate a list of quantum numbers (l, m) for spherical harmonics.

    Parameters
    ----------
    n : int
        Maximum degree of the spherical harmonics.

    Returns
    -------
    list of list
        A list containing pairs of quantum numbers [l, m].
    """
    lm = []
    for l in range(n):
        for m in range(-l, l+1):
            lm.append([l, m])
    return lm

def generate_S_mat(lm_set, num_t, num_p):
    """
    Generate the S-matrix for real spherical harmonics.

    Parameters
    ----------
    lm_set : list of tuple
        List of quantum numbers (l, m) for spherical harmonics.
    num_t : int
        Number of divisions along the theta axis.
    num_p : int
        Number of divisions along the phi axis.

    Returns
    -------
    np.ndarray
        The S-matrix where each column corresponds to a flattened spherical
        harmonics function evaluated over a grid of theta and phi.
    """
    # Generate linearly spaced angles for theta and phi
    ts = np.linspace(0, np.pi, num_t)
    ps = np.linspace(0, 2*np.pi, num_p)

    # Create a meshgrid for theta and phi
    theta, phi = np.meshgrid(ts, ps, indexing='ij')

    S_mat_T = []
    for l, m in lm_set:
        # Calculate the spherical harmonics for each (l, m) pair
        slm_2d = spherical_harmonics(l, m, theta, phi)
        # Flatten the 2D array to 1D
        slm_1d = slm_2d.flatten()
        # Append the flattened array to the list
        S_mat_T.append(slm_1d)

    # Convert the list to a numpy array and transpose it
    return np.array(S_mat_T).T

# Dictionary mapping quantum numbers (l, m) to orbital symbols
orb_symbol = {
    (0,  0): 's',
    (1, -1): 'py',
    (1,  0): 'pz',
    (1,  1): 'px',
    (2, -2): 'dxy',
    (2, -1): 'dyz',
    (2,  0): 'dz2',
    (2,  1): 'dxz',
    (2,  2): 'dx2-y2',
    (3, -3): 'l=3,mr=7',
    (3, -2): 'l=3,mr=5',
    (3, -1): 'l=3,mr=3',
    (3,  0): 'l=3,mr=1',
    (3,  1): 'l=3,mr=2',
    (3,  2): 'l=3,mr=4',
    (3,  3): 'l=3,mr=6',
}

if __name__ == "__main__":
    # Example usage: generate quantum numbers up to l_max
    l_max = 3
    lm_list = quantum_number(l_max)
    print(lm_list)
