import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

from . import wavefunction as wf
from . import sph_harm as sph

# This module provides a class for performing Lasso regression on a given wavefunction.
# The class computes the coefficients of real spherical harmonics for a given wavefunction data.
# The coefficients are calculated using the formula:
# S^{\\prime}(\\theta,\\phi) = \\sum_{l,m}c^{\\prime}_{l,m}S_{l,m}(\\theta,\\phi)
# The class also provides methods to output the computed coefficients and optimal radius to a file.

class LassoRegression:
    r"""
    A class to perform Lasso regression on wavefunction data to compute the coefficients
    of real spherical harmonics.

    The coefficients are calculated using the formula:
    S^{\prime}(\theta,\phi) = \sum_{l,m}c^{\prime}_{l,m}S_{l,m}(\theta,\phi)
    which can be expressed in matrix form as:
        S^{\prime}_{i} = \sum_{j}S_{i,j}c^{\prime}_{j}, 
        where {i}:={\theta,\phi} and {j}:={l, m}.
    This simplifies to:
        S^{\prime} = S * c^{\prime}

    The objective function to minimize in Lasso regression is:
        min_{c'}||S' - Sc'||^{2}_{2}/(2*n_{samples}) + \alpha*||c'||_{1},
    where \alpha is a hyperparameter controlling the regularization strength.
    """

    def __init__(self, num_t, num_p, n, **psi_dict):
        """
        Initialize the LassoRegression class with the given parameters.

        Parameters
        ----------
        num_t : int
            Number of divisions along the theta axis.
        num_p : int
            Number of divisions along the phi axis.
        n : int
            Quantum number for generating the set of spherical harmonics.
        psi_dict : dict
            Dictionary containing wavefunction information such as name, info, and resolution.
        """
        self.num_t = num_t
        self.num_p = num_p

        # Generate the matrix S_{i,j} using quantum numbers
        self.lmset = sph.quantum_number(n)
        self.S_mat = sph.generate_S_mat(self.lmset, self.num_t, self.num_p)

        # Load wavefunction data and additional information
        nx, ny, nz = psi_dict.get('resolution') 
        self.psi_name = psi_dict.get('name')
        self.gwf = wf.GammaWaveFunction(np.load(self.psi_name), nx, ny, nz)
        self.psi_info = np.load(psi_dict.get('info'))

    def run_at_r(self, r, num_split=4):
        """
        Perform Lasso regression at a specific radius to compute coefficients.

        Parameters
        ----------
        r : float
            Radius at which to perform the regression.
        num_split : int, optional
            Number of splits for cross-validation, by default 4.

        Returns
        -------
        tuple of np.ndarray
            Arrays of computed coefficients and optimal alpha values.
        """
        clm_prime = []
        opt_alpha = []
        # Get spherical data for the given radius
        S_prime_list = self.gwf.get_spherical_data(r, self.num_t, self.num_p)
        for i, S_prime in enumerate(S_prime_list):
            # Flatten the spherical data for regression
            Q = S_prime.flatten()
            # Determine the optimal alpha using cross-validation
            opt_alpha_cv = self._decide_param(Q, num_split)
            # Perform Lasso regression with the optimal alpha
            clf = Lasso(alpha=opt_alpha_cv, max_iter=10000)
            clf.fit(X=self.S_mat, y=Q)
            # Normalize the coefficients and store them
            clm_prime.append(clf.coef_ / np.linalg.norm(clf.coef_))
            opt_alpha.append(opt_alpha_cv)
        return np.array(clm_prime), np.array(opt_alpha)

    def run(self, rstep=0.1, num_split=4):
        """
        Run the Lasso regression over a range of radii to find the optimal radius.

        Parameters
        ----------
        rstep : float, optional
            Step size for radius calculation, by default 0.1.
        num_split : int, optional
            Number of splits for cross-validation, by default 4.

        Returns
        -------
        tuple
            Optimal radii, alpha values, and coefficients.
        """
        # Calculate radii to evaluate
        r_calc = np.arange(0.1, 0.4, rstep)

        nr = len(r_calc)
        nlm = len(self.lmset)
        npsi = len(self.gwf.psi_org)

        # Initialize arrays to store results
        nzero = np.zeros((nr, npsi), dtype=int)
        alpha_r = np.zeros((nr, npsi), dtype=float)
        clm_r = np.zeros((nr, npsi, nlm), dtype=float)
        for ir, r in enumerate(r_calc):
            # Compute coefficients and alpha for each radius
            clm, alpha = self.run_at_r(r)
            clm_r[ir] = clm
            nzero[ir] = np.count_nonzero(clm, axis=1)
            alpha_r[ir] = alpha
        # Determine the optimal radius index for each wavefunction
        opt_r_idx = np.argmin(nzero, axis=0)

        # Collect final results for optimal radii
        r_opt = []
        clm_opt = []
        alpha_opt = []
        for i in range(npsi):
            r_opt.append(r_calc[opt_r_idx[i]])
            clm_opt.append(clm_r[opt_r_idx[i], i])
            alpha_opt.append(alpha_r[opt_r_idx[i], i])
        # Output the results to a file
        self._output_info(r_opt, clm_opt, alpha_opt)

        return (r_opt, alpha_opt, clm_opt)

    def _decide_param(self, Q, num_split):
        """
        Determine the best alpha parameter using cross-validation.

        Parameters
        ----------
        Q : np.ndarray
            Flattened spherical data for regression.
        num_split : int
            Number of splits for cross-validation.

        Returns
        -------
        float
            Optimal alpha value.
        """
        # Define a range of alpha parameters to test
        params = 10**(np.arange(-4, 1, 0.05))
        # Perform cross-validation to find the best alpha
        clf = LassoCV(alphas=params, cv=num_split)
        clf.fit(X=self.S_mat, y=Q)
        return clf.alpha_

    def _output_info(self, r_opt, clm_opt, alpha_opt):
        """
        Output the computed coefficients and optimal parameters to a file.

        Parameters
        ----------
        r_opt : list of float
            Optimal radii for each wavefunction.
        clm_opt : list of np.ndarray
            Coefficients of real spherical harmonics for each wavefunction.
        alpha_opt : list of float
            Optimal alpha values for each wavefunction.
        """
        # Prepare the output string with detailed information
        o = f'Target data : {self.psi_name}\n\n'
        for info, r, clm, alpha in zip(self.psi_info, r_opt, clm_opt, alpha_opt):
            o += f'molecule {int(info[0]):>4d}, band {int(info[1]):>4d}\n\n'
            o += f'r = {r:>8.5f}, alpha = {alpha:>8.5f}\n\n'
            o += 'predicted coefficients of real spherical harmonics:\n\n'
            for c, lm in zip(clm, self.lmset):
                o += f'l = {lm[0]:>2}, m = {lm[1]:>2} ({sph.orb_symbol[(lm[0], lm[1])]:>12}): {c:>15.7E}\n'
            o += '\n'
        # Write the output to a file
        with open(f'lasso.out', 'w') as fw:
            fw.write(o)

if __name__ == '__main__':
    # Example usage of the LassoRegression class
    n = 3
    num_t = 64
    num_p = 64
    nx = ny = nz = 32
    psi_name = 'image32x32x32.npy'
    psi_info = 'image_info.npy'

    # Create a dictionary with wavefunction information
    psi_dict = {'name': psi_name, 'info': psi_info, 'resolution': [nx, ny, nz]}
    # Initialize the LassoRegression class
    lr = LassoRegression(num_t, num_p, n, **psi_dict)
    # Run the regression to obtain optimal parameters and coefficients
    rs, alphas, clms = lr.run()

    # Output the results to a CSV file
    from predict import output_csv
    output_csv(lr.psi_info, lr.lmset, clms, sph.orb_symbol)
