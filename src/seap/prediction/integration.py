import numpy as np

from . import wavefunction as wf
from . import sph_harm as sph

# This module provides a class for performing integration of spherical harmonics over a given wavefunction.
# The class computes the coefficients of real spherical harmonics for a given wavefunction data.
# The coefficients are calculated using the formula:
# S^{\\prime}(\\theta,\\phi) = \\sum_{l,m}c^{\\prime}_{lm}S_{lm}(\\theta, \\phi)
# where c^{\\prime}_{lm} = <S_{lm}|S^{\\prime}>_{\\theta, \\phi}
# The class also provides methods to output the computed coefficients and optimal radius to a file.

class Integration:
    r"""
    A class to perform integration of spherical harmonics over a given wavefunction.

    The integration is performed to compute the coefficients of real spherical harmonics
    for a given wavefunction data. The coefficients are calculated using the formula:

    S^{\prime}(\theta,\phi) = \sum_{l,m}c^{\prime}_{lm}S_{lm}(\theta, \phi)

    where c^{\prime}_{lm} = <S_{lm}|S^{\prime}>_{\theta, \phi}

    Attributes
    ----------
    num_t : int
        Number of divisions along the theta axis.
    num_p : int
        Number of divisions along the phi axis.
    lmset : list of tuple
        List of quantum numbers (l, m) for spherical harmonics.
    tline : np.ndarray
        Array of theta values.
    pline : np.ndarray
        Array of phi values.
    theta : np.ndarray
        Meshgrid of theta values.
    phi : np.ndarray
        Meshgrid of phi values.
    psi_name : str
        Name of the wavefunction file.
    gwf : GammaWaveFunction
        Instance of GammaWaveFunction containing wavefunction data.
    psi_info : np.ndarray
        Array containing additional information about the wavefunction.

    Methods
    -------
    run_at_r(r)
        Compute the coefficients of spherical harmonics at a given radius.
    _sph_int(f, g)
        Compute the inner product of two functions over theta and phi.
    run(rstep=0.1, threshold=0.1)
        Find the optimal radius and compute the coefficients of spherical harmonics.
    output_info(r_opt, clm_opt)
        Output the computed coefficients and optimal radius to a file.
    """

    def __init__(self, num_t: int, num_p: int, n: int, **psi_dict):
        """
        Initialize the Integration class with the given parameters.

        Parameters
        ----------
        num_t : int
            Number of divisions along the theta axis.
        num_p : int
            Number of divisions along the phi axis.
        n : int
            Quantum number for determining the set of spherical harmonics.
        psi_dict : dict
            Dictionary containing wavefunction data and resolution.
        """
        self.num_t = num_t
        self.num_p = num_p
        self.lmset = sph.quantum_number(n)

        # Create theta and phi lines for integration
        self.tline = np.linspace(0, np.pi, self.num_t)
        self.pline = np.linspace(0, 2 * np.pi, self.num_p)
        self.theta, self.phi = np.meshgrid(self.tline, self.pline, indexing='ij')

        # Load wavefunction data
        nx, ny, nz = psi_dict.get('resolution')
        self.psi_name = psi_dict.get('name')
        self.gwf = wf.GammaWaveFunction(np.load(self.psi_name), nx, ny, nz)
        self.psi_info = np.load(psi_dict.get('info'))

    def run_at_r(self, r: float) -> np.ndarray:
        """
        Compute the coefficients of spherical harmonics at a given radius.

        Parameters
        ----------
        r : float
            Radius at which to compute the coefficients.

        Returns
        -------
        np.ndarray
            Array of computed coefficients for each wavefunction.
        """
        clm_list = []
        S_prime_list = self.gwf.get_spherical_data(r, self.num_t, self.num_p)
        for i, S_prime in enumerate(S_prime_list):
            clm = []
            for lm in self.lmset:
                l, m = lm
                S_lm = sph.spherical_harmonics(l, m, self.theta, self.phi)
                c_lm = self._sph_int(S_lm, S_prime)
                clm.append(c_lm)
            clm_list.append(np.array(clm) / np.linalg.norm(clm))
        return np.array(clm_list)

    def _sph_int(self, f: np.ndarray, g: np.ndarray) -> float:
        """
        Compute the inner product of two functions over theta and phi.

        Parameters
        ----------
        f : np.ndarray
            First function (spherical harmonics).
        g : np.ndarray
            Second function (wavefunction data).

        Returns
        -------
        float
            Inner product of the two functions.
        """
        jacob = np.sin(self.tline)
        fg_p = np.zeros(len(self.pline), dtype=float)
        for ip, p in enumerate(self.pline):
            fg = f[:, ip] * g[:, ip]
            fg_p[ip] = np.trapz(fg * jacob, x=self.tline)
        return np.trapz(fg_p, x=self.pline)

    def run(self, rstep: float = 0.1, threshold: float = 0.1) -> tuple:
        """
        Find the optimal radius and compute the coefficients of spherical harmonics.

        Parameters
        ----------
        rstep : float, optional
            Step size for radius calculation, by default 0.1.
        threshold : float, optional
            Threshold for coefficient significance, by default 0.1.

        Returns
        -------
        tuple
            Arrays of optimal radii and corresponding coefficients.
        """
        r_calc = np.arange(0.1, 0.4, rstep)

        nr = len(r_calc)
        nlm = len(self.lmset)
        npsi = len(self.gwf.psi_org)

        # Find optimal radius
        nzero = np.zeros((nr, npsi), dtype=int)
        clm_r = np.zeros((nr, npsi, nlm), dtype=float)
        for ir, r in enumerate(r_calc):
            clm = self.run_at_r(r)
            clm_r[ir] = clm
            bool_array = clm >= threshold
            nzero[ir] = np.sum(bool_array, axis=1)
        opt_r_idx = np.argmin(nzero, axis=0)

        # Get final results
        r_opt = []
        clm_opt = []
        for i in range(npsi):
            r_opt.append(r_calc[opt_r_idx[i]])
            clm_opt.append(clm_r[opt_r_idx[i], i])
        self.output_info(r_opt, clm_opt)

        return np.array(r_opt), np.array(clm_opt)

    def output_info(self, r_opt: np.ndarray, clm_opt: np.ndarray) -> None:
        """
        Output the computed coefficients and optimal radius to a file.

        Parameters
        ----------
        r_opt : np.ndarray
            Array of optimal radii for each wavefunction.
        clm_opt : np.ndarray
            Array of computed coefficients for each wavefunction.

        Returns
        -------
        None
        """
        output = f'Target data : {self.psi_name}\n\n'
        for info, r, clm in zip(self.psi_info, r_opt, clm_opt):
            output += f'molecule {int(info[0]):>4d}, band {int(info[1]):>4d}\n\n'
            output += f'r = {r}\n\n'
            output += 'predicted coefficients of real spherical harmonics:\n\n'
            for c, lm in zip(clm, self.lmset):
                output += f'l = {lm[0]:>2}, m = {lm[1]:>2} ({sph.orb_symbol[(lm[0], lm[1])]:>12}): {c:>15.7E}\n'
            output += '\n'
        with open('itg.out', 'w') as fw:
            fw.write(output)

if __name__ == '__main__':
    # Example usage of the Integration class
    n = 3
    num_t = 64
    num_p = 64
    nx = ny = nz = 32
    psi_name = 'image32x32x32.npy'
    psi_info = 'image_info.npy'

    psi_dict = {'name': psi_name, 'info': psi_info, 'resolution': [nx, ny, nz]}
    itg = Integration(num_t, num_p, n, **psi_dict)
    rs, clms = itg.run()

    from predict import output_csv
    output_csv(itg.psi_info, itg.lmset, clms, sph.orb_symbol)
