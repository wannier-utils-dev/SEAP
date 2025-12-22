# This script generates synthetic data for machine learning.
# It creates a dataset of input-output pairs where the input is a set of parameters
# and the output is a grid of data generated from those parameters.
# The data is generated using a radial function and a set of spherical harmonics.
# The radial function is a function of the distance from the origin, and the spherical harmonics
# are functions of the angular coordinates.
# The data is generated using a random set of parameters, and the data is normalized.
# The data is then saved to a file.

import os
import sys
import random
import itertools

import numpy as np
from scipy.special import factorial

from . import grids
from . import sph_harm as sph


class BoxData:
    """
    Attributes
    ----------
    Here, the number of points contained in the inscribed sphere
    (i.e., len(grid_pol_in_ball)), is denoted as n_points.

    rvec   : ndarray
             shape=(n_points).
             ex) rvec = [r_0, r_1, r_2, r_3, ..., r_(n_points - 1)]
    rpow   : ndarray
             shape=(rn_max, n_points).
             ex) rpow[:, i] = [(r_i)^0, (r_i)^1, (r_i)^2, ..., (r_i)^(rn_max-1)]
    sphmat : ndarray
             Arrays of real spherical harmonics, shape=((l_max+1)**2, n_points):
             sphmat[i, j] = sphmat[(l,m), (theta, phi)].
    """
    def __init__(self, l_max, rn_max, n_div, size=1.0):
        """
        Initialize the BoxData class with the given parameters.

        Parameters
        ----------
        l_max : int
            Maximum degree of spherical harmonics.
        rn_max : int
            Maximum power for radial functions.
        n_div : int
            Number of divisions for the grid.
        size : float, optional
            Size of the grid, by default 1.0.
        """
        self.n = l_max + 1
        self.n_div = n_div
        self.rn_max = rn_max
        self.lm_set = sph.quantum_number(self.n)
        # Prepare cubic grid points and extract points in the inscribed sphere.
        grid_cart = grids.cartesian_grid(np.repeat(self.n_div, 3), size)
        grid_pol_in_ball, self.map_idx = grids.extract_inscribed_ball(grid_cart, size)
        # Arrays used to convert from parameters to grid data.
        self.rvec, self.rpow, self.sphmat = self.set_arrays(grid_pol_in_ball)

    def set_arrays(self, grid_in_pol):
        """
        Set up arrays for radial and spherical harmonic components.

        Parameters
        ----------
        grid_in_pol : list
            List of grid points in polar coordinates.

        Returns
        -------
        tuple
            Tuple containing arrays for radial vectors, radial powers, and spherical harmonics.
        """
        rpow = []
        rvec = []
        sphmat = []
        for r_pol in grid_in_pol:
            r, theta, phi = r_pol
            rvec.append(r)
            rpow.append([r**n for n in range(self.rn_max)])
            sphmat.append([
                sph.spherical_harmonics(l, m, theta, phi) for l, m in self.lm_set
            ])
        return np.array(rvec), np.array(rpow).T, np.array(sphmat).T
    
    def generate_learning_data(self, n_samples):
        """
        Generate synthetic data for machine learning.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        tuple
            Tuple containing input data (x) and target data (y).
        """
        x = np.empty((n_samples, self.n_div**3), dtype=np.float32)
        y = np.empty((n_samples, self.n**2 + self.rn_max + 1), dtype=np.float32)
        for i in range(n_samples):
            random.seed()
            # Generate random parameters for the radial part
            gamma, an = self.get_random_radial_params()
            # Select random spherical harmonics coefficients
            n_c = random.choice(range(1, len(self.lm_set))) 
            lm_selected = random.sample(self.lm_set, k=n_c)
            c = self.assign_sph_coeffs(lm_selected).reshape(1, -1)
            # Generate data based on parameters
            data = self.params_to_boxdata(c, gamma, an)
            x[i] = np.ravel(data)
            y[i] = np.concatenate([np.ravel(c), [gamma], an])
        return x, y

    def get_random_radial_params(self):
        """
        Generate random parameters for the radial function.

        Returns
        -------
        tuple
            Tuple containing gamma and normalized radial coefficients (an).
        """
        gamma = random.uniform(1.0, 5.0)
        an = [random.uniform(-1.0, 1.0) for k in range(self.rn_max)]
        # Normalize the radial coefficients
        norm = 0.0
        for n1, n2 in itertools.product(range(self.rn_max), repeat=2):
            norm += (an[n1] * an[n2] * factorial(n1 + n2 + 2)) / (2 * gamma)**(n1 + n2 + 3)
        an /= np.sqrt(norm)
        return gamma, an

    def assign_sph_coeffs(self, lm_list):
        """
        Assign non-zero coefficients to selected spherical harmonics.

        Parameters
        ----------
        lm_list : list
            List of (l, m) tuples for which to assign coefficients.

        Returns
        -------
        ndarray
            Array of spherical harmonics coefficients.
        """
        coeffs = np.zeros(self.n**2, dtype=np.float32)
        c_cand = np.linspace(-10, 10, 1000)
        c_nonzero = random.choices(c_cand[c_cand != 0], k=len(lm_list))
        for j, lm in enumerate(lm_list):
            l, m = lm
            coeffs[l**2 + m + l] = c_nonzero[j]
        return coeffs / np.linalg.norm(coeffs)

    def params_to_boxdata(self, c, gamma, an):
        """
        Transform parameters to grid data.

        Parameters
        ----------
        c : ndarray
            Coefficients for spherical harmonics.
        gamma : float
            Radial decay parameter.
        an : list
            Radial coefficients.

        Returns
        -------
        ndarray
            Grid data transformed from parameters.
        """
        n_batch = c.shape[0]
        angular_parts = c @ self.sphmat
        radial_parts = np.exp(np.outer(-gamma, self.rvec)) * (an @ self.rpow)
        values = angular_parts * radial_parts
        # Assign values to grid points
        data = np.zeros((n_batch, self.n_div**3))
        for i in range(n_batch):
            np.put(data[i], self.map_idx, values[i])
        return data
