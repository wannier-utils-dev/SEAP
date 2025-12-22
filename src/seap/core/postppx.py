import sys
import glob
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy import interpolate as itp

from pymatgen.core import Structure, Lattice
from doped.generation import get_interstitial_sites


# Handle both relative imports (when used as module) and absolute imports (when run directly)
#try:
#    from seap.common import xsf
#    from seap.common.molecule import atoms2molecules
#    from seap.common.lattice import crys2cart, cart2crys
#except ImportError:

# Add the src directory to the path for direct execution
if __package__ in (None, ""):
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from seap.common import xsf
from seap.common.molecule import atoms2molecules
from seap.common.lattice import crys2cart, cart2crys

# This script is used to postprocess the Wannier function data.
# It takes an XSF file and a bond length threshold, and it will cluster the atoms based on the bond length.
# It will then generate a cube file for each cluster, which can be used to visualize the Wannier function.
# The cube files will be named according to the band index and molecule index.

class PostPPX:
    def __init__(self, xsf_file, bond_length=1.8, interstitial_length=None, m1=3, m2=3, m3=3):
        """
        Initialize the PostPPX class with the given parameters.

        Parameters
        ----------
        xsf_file : str
            Path to the XSF file.
        bond_length : float, optional
            Bond length threshold for clustering, by default 1.8.
        interstitial_length : float
            Length for searching voids, by default None
        m1 : int, optional
            Supercell multiplier in the first direction, by default 3.
        m2 : int, optional
            Supercell multiplier in the second direction, by default 3.
        m3 : int, optional
            Supercell multiplier in the third direction, by default 3.
        """
        params = xsf.get_params_from_xsf(xsf_file)
        self._primvecs = np.array(params["prim_vec"])
        self._spanvecs = np.array(params["span_vec"])
        self._gridvecs = np.array(params["grid_info"])
        self._origin = np.array(params["pos_org"])
        self._primcoord = params["prim_coord"]
        self._num_atom = params["num_atom"]

        if interstitial_length:
            self._void_centers = self.find_voids(interstitial_length)
            print(f"# of void_center:{len(self._void_centers)}")
            if self._void_centers:
                print("add interstitial sites X")
                for v in self._void_centers:
                    self._primcoord.append(["X", f"{v[0]:.8f}", f"{v[1]:.8f}", f"{v[2]:.8f}"])
                self._num_atom += len(self._void_centers)
        else:
            print("not add voids")
            self._void_centers=[]
        
        # Perform clustering to identify molecular clusters
        self.cluster_indices, self.cluster_positions = self._clustering(bond_length)

        self._m1 = m1
        self._m2 = m2
        self._m3 = m3

        # Search for nearest neighbors in the supercell
        self._nn_rvecs, self._nn_atoms = self._search_nearest_neighbors()

    def _supervecs(self, m1, m2, m3):
        """
        Return the lattice vectors of the supercell.

        Parameters
        ----------
        m1 : int
            Supercell multiplier in the first direction.
        m2 : int
            Supercell multiplier in the second direction.
        m3 : int
            Supercell multiplier in the third direction.

        Returns
        -------
        np.ndarray
            Lattice vectors of the supercell.
        """
        return np.dot(np.diag([m1, m2, m3]), self._primvecs)

    def _atomic_positions(self, crys=True):
        """
        Get atomic positions in either Cartesian or crystal coordinates.

        Parameters
        ----------
        crys : bool, optional
            If True, return positions in crystal coordinates, otherwise in Cartesian coordinates, by default True.

        Returns
        -------
        np.ndarray
            Atomic positions.
        """
        pos = np.array([[float(s[1]), float(s[2]), float(s[3])] for s in self._primcoord])
        if crys:
            return cart2crys(pos, self._primvecs)
        else:
            return pos

    def _element_symbols(self):
        """
        Get element symbols from the primitive coordinates.

        Returns
        -------
        np.ndarray
            Element symbols.
        """
        return np.array([s[0] for s in self._primcoord])

    def _new_gridvecs(self, m1, m2, m3):
        """
        Calculate the new grid vectors for the supercell.

        Parameters
        ----------
        m1 : int
            Supercell multiplier in the first direction.
        m2 : int
            Supercell multiplier in the second direction.
        m3 : int
            Supercell multiplier in the third direction.

        Returns
        -------
        np.ndarray
            New grid vectors.
        """
        n1, n2, n3 = self._gridvecs
        return np.array([n1 * m1, n2 * m2, n3 * m3])

    def _clustering(self, bond_length):
        """
        Perform clustering to identify molecular clusters based on bond length.

        Parameters
        ----------
        bond_length : float
            Bond length threshold for clustering.

        Returns
        -------
        tuple
            Cluster indices and cluster positions.
        """
        pos_crys = self._atomic_positions(crys=True)
        cluster_indices, cluster_positions = atoms2molecules(self._primvecs, pos_crys, threshold=bond_length)
        return cluster_indices, cluster_positions

    def _data_in_primitive(self, data, pbc=False):
        """
        Reshape data to fit within the primitive cell.

        Parameters
        ----------
        data : np.ndarray
            Data to be reshaped.
        pbc : bool, optional
            If True, apply periodic boundary conditions, by default False.

        Returns
        -------
        np.ndarray
            Reshaped data.
        """
        n1, n2, n3 = self._gridvecs
        if pbc:
            i, j, k = np.meshgrid(range(n1 + 1), range(n2 + 1), range(n3 + 1), indexing="ij")
        else:
            i, j, k = np.meshgrid(range(n1), range(n2), range(n3), indexing="ij")
        return data[(i % n1) + (j % n2) * n1 + (k % n3) * n1 * n2]

    def _data_in_super(self, data, m1, m2, m3, pbc=False):
        """
        Reshape data to fit within the supercell.

        Parameters
        ----------
        data : np.ndarray
            Data to be reshaped.
        m1 : int
            Supercell multiplier in the first direction.
        m2 : int
            Supercell multiplier in the second direction.
        m3 : int
            Supercell multiplier in the third direction.
        pbc : bool, optional
            If True, apply periodic boundary conditions, by default False.

        Returns
        -------
        np.ndarray
            Reshaped data.
        """
        np1, np2, np3 = self._gridvecs
        ns1, ns2, ns3 = self._new_gridvecs(m1, m2, m3)
        if pbc:
            i, j, k = np.meshgrid(range(ns1 + 1), range(ns2 + 1), range(ns3 + 1), indexing="ij")
        else:
            i, j, k = np.meshgrid(range(ns1), range(ns2), range(ns3), indexing="ij")
        return data[(i % np1) + (j % np2) * np1 + (k % np3) * np1 * np2]

    def _interpolator_in_primitive(self, data):
        """
        Create an interpolator for data within the primitive cell.

        Parameters
        ----------
        data : np.ndarray
            Data to be interpolated.

        Returns
        -------
        scipy.interpolate.RegularGridInterpolator
            Interpolator for the data.
        """
        n1, n2, n3 = self._gridvecs
        x_1 = np.linspace(0, 1, n1 + 1, endpoint=True)
        x_2 = np.linspace(0, 1, n2 + 1, endpoint=True)
        x_3 = np.linspace(0, 1, n3 + 1, endpoint=True)
        data_reshape = self._data_in_primitive(data, pbc=True)
        return itp.RegularGridInterpolator((x_1, x_2, x_3), data_reshape)

    def _interpolator_in_super(self, data, m1, m2, m3):
        """
        Create an interpolator for data within the supercell.

        Parameters
        ----------
        data : np.ndarray
            Data to be interpolated.
        m1 : int
            Supercell multiplier in the first direction.
        m2 : int
            Supercell multiplier in the second direction.
        m3 : int
            Supercell multiplier in the third direction.

        Returns
        -------
        scipy.interpolate.RegularGridInterpolator
            Interpolator for the data.
        """
        ns1, ns2, ns3 = self._new_gridvecs(m1, m2, m3)
        x_1 = np.linspace(0, 1, ns1 + 1, endpoint=True)
        x_2 = np.linspace(0, 1, ns2 + 1, endpoint=True)
        x_3 = np.linspace(0, 1, ns3 + 1, endpoint=True)
        data_reshape = self._data_in_super(data, m1, m2, m3, pbc=True)
        return itp.RegularGridInterpolator((x_1, x_2, x_3), data_reshape)

    def _search_nearest_neighbors(self):
        """
        Search for the nearest neighbors in the supercell.

        Returns
        -------
        tuple
            Nearest neighbor relative vectors and atom indices.
        """
        m1, m2, m3 = self._m1, self._m2, self._m3
        ns1, ns2, ns3 = self._new_gridvecs(m1, m2, m3)
        ncell = np.prod([ns1, ns2, ns3])

        # Generate candidate grid points in Cartesian coordinates
        all_grid = np.array(list(itertools.product(range(ns1), range(ns2), range(ns3))))
        grid_cart = np.dot(all_grid, self._spanvecs / np.array([ns1, ns2, ns3])) + self._origin
        transvecs = crys2cart(np.array(list(itertools.product(range(m1), range(m2), range(m3)))), self._primvecs)
        candidate = grid_cart[:, None, :] + transvecs[None, :, :] - np.sum(self._primvecs, axis=0)

        # Search for nearest neighbors using KDTree
        distances, indices = KDTree(crys2cart(self.cluster_positions, self._primvecs)).query(candidate)
        nn_index = np.argmin(distances, axis=1)
        nn_rvecs = candidate[np.arange(ncell), nn_index]
        nn_atoms = indices[np.arange(ncell), nn_index]
        return nn_rvecs, nn_atoms

    def get_centers(self, data):
        """
        Calculate the centers of charge for each cluster.

        Parameters
        ----------
        data : np.ndarray
            Data representing the charge density.

        Returns
        -------
        tuple
            Centers of charge and charge ratios for each cluster.
        """
        ns1, ns2, ns3 = self._new_gridvecs(self._m1, self._m2, self._m3)
        all_grid = np.array(list(itertools.product(range(ns1), range(ns2), range(ns3))))

        # Interpolate data within the primitive cell
        func_pcell = self._interpolator_in_primitive(data)
        x_1, x_2, x_3 = [np.linspace(0, 1, n, endpoint=False) for n in [ns1, ns2, ns3]]
        X_1, X_2, X_3 = np.meshgrid(x_1, x_2, x_3, indexing="ij")
        data_new = np.ravel(func_pcell((X_1, X_2, X_3)), order="F")

        num_cluster = len(self.cluster_indices)
        rho_total = np.sum(abs(data_new))
        rho_ratios = np.zeros((num_cluster), dtype=float)
        centers = np.zeros((num_cluster, 3), dtype=float)
        for ic, cluster in enumerate(self.cluster_indices):
            r_indices = np.isin(self._nn_atoms, np.array(list(cluster)))
            rvecs = self._nn_rvecs[r_indices]

            i, j, k = [all_grid[r_indices, l] for l in [0, 1, 2]]
            rho = abs(data_new[(i % ns1) + (j % ns2) * ns1 + (k % ns3) * ns1 * ns2])
            rho_per_cluster = np.sum(rho)
            centers[ic] = np.einsum('ij,i->j', rvecs, rho) / rho_per_cluster
            rho_ratios[ic] = rho_per_cluster / rho_total
        return centers, rho_ratios

    def generate_cube_data(self, data, centers, cube_length=4, num_div=32, xsf_name=None):
        """
        Generate cube data for visualization.

        Parameters
        ----------
        data : np.ndarray
            Data representing the charge density.
        centers : np.ndarray
            Centers of charge for each cluster.
        cube_length : float, optional
            Length of the cube, by default 4.
        num_div : int, optional
            Number of divisions in each direction, by default 32.
        xsf_name : str, optional
            Base name for the XSF files, by default None.

        Returns
        -------
        np.ndarray
            Generated cube data.
        """
        # Set supercell multipliers
        m1 = m2 = m3 = 5
        func_scell = self._interpolator_in_super(data, m1, m2, m3)

        # Generate grid points within the cube
        gridvecs = np.repeat(num_div, 3)
        grid_tmp = np.indices(gridvecs).transpose((1, 2, 3, 0)).reshape((-1, 3)) / gridvecs
        r = (grid_tmp + 1.0 / 2.0 / gridvecs) * cube_length

        data_total = []
        supervecs = self._supervecs(m1, m2, m3)
        print(supervecs)
        for i, center in enumerate(centers):
            cube_origin = center - np.repeat(cube_length / 2.0, 3)
            r_shift = r + cube_origin + np.sum(2 * np.array(self._primvecs), axis=0)
            r_shift_crys = cart2crys(r_shift, supervecs)
            print(np.min(r_shift_crys), np.max(r_shift_crys))
            rho_cube = func_scell((r_shift_crys[:, 0], r_shift_crys[:, 1], r_shift_crys[:, 2]))
            data_total.append(np.sign(rho_cube) * np.sqrt(np.abs(rho_cube)))
            if xsf_name:
                filename = xsf_name + f"_{i}.xsf"
                xsf_data = np.ravel(rho_cube.reshape(gridvecs), order="F")
                self._output_cube_data_xsf(filename, gridvecs, cube_origin, cube_length, xsf_data)
        return np.array(data_total)

    def _output_cube_data_xsf(self, xsf_name, gridvecs, origin, cube_length, data):
        """
        Output cube data in XSF format.

        Parameters
        ----------
        xsf_name : str
            Name of the XSF file.
        gridvecs : np.ndarray
            Grid vectors.
        origin : np.ndarray
            Origin of the cube.
        cube_length : float
            Length of the cube.
        data : np.ndarray
            Data to be output.

        Note
        ----
        This data is on a grid that does not include the endpoints of the spanvecs,
        so when output in XSF format, the data may appear slightly misaligned. 
        Specifically, the wave function may seem offset from the atomic positions. 
        For example, when displaying it in VESTA, you can verify the correctness of 
        the data shape by removing the atomic symbols.
        """
        params = {
            "xsf_name": xsf_name,
            "prim_vec": self._primvecs,
            "num_atom": self._num_atom,
            "prim_coord": self._primcoord,
            "grid_info": gridvecs,
            "pos_org": origin,
            "span_vec": np.diag([cube_length, cube_length, cube_length]),
            "grid_data": data,
        }
        xsf.output_xsf(params)

    def find_voids(self, interstitial_length):
        """
        search voids using doped package
        Returns
        -------
        void_centers : list of np.ndarray
            cartesian coordinates of void centers
        """

        void_centers=[]
        try:
            primvecs = self._primvecs
            species = [atom[0] for atom in self._primcoord]
            coords = np.array([[float(a[1]), float(a[2]), float(a[3])] for a in self._primcoord])
            lattice = Lattice(primvecs)
            fcoords = lattice.get_fractional_coords(coords)
            fcoords = np.round(fcoords,8)
            structure = Structure(lattice, species, fcoords, coords_are_cartesian=False)

            interstitial_sites = get_interstitial_sites(structure,interstitial_gen_kwargs={"min_dist":interstitial_length})
            
            for _ideal_sites, _multiplicity, equiv_frac_coords in interstitial_sites:
                for frac_coord in equiv_frac_coords:
                    cart_coord = structure.lattice.get_cartesian_coords(frac_coord)
                    void_centers.append(cart_coord)
            return void_centers
        except Exception as e:
            print(f"Failed to find voids: {e}", file=sys.stderr)
            return []


def output_info(kband, imol, rho_ratio, center):
    """
    Output information about the calculation to a file.

    Parameters
    ----------
    kband : int
        Band index.
    imol : int
        Molecule index.
    rho_ratio : float
        Ratio of charge density.
    center : np.ndarray
        Center of charge.

    Returns
    -------
    None
    """
    with open('output_ppin.out', 'a') as fw:
        fw.write(f'band     : {kband}\n')
        fw.write(f'molecure : {imol}\n')
        fw.write('cart. coord.\n')
        fw.write('center = {0[0]:>10.6f}{0[1]:>10.6f}{0[2]:>10.6f}\n'.format(center))
        fw.write('rho/total = {0:>10.6f}\n'.format(rho_ratio))
        fw.write('\n')


def output_csv(calc_info):
    """
    Output calculation information to a CSV file.

    Parameters
    ----------
    calc_info : list
        List of calculation information.

    Returns
    -------
    None
    """
    data = {'band_index': [], 'molecule_index': [], 'center_x': [], 'center_y': [], 'center_z': []}
    for info in calc_info:
        data['band_index'].append(int(info[1]))
        data['molecule_index'].append(int(info[0]))
        data['center_x'].append(info[2][0])
        data['center_y'].append(info[2][1])
        data['center_z'].append(info[2][2])
    df = pd.DataFrame(data)
    df.to_csv('center.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='output_ppin.py')
    parser.add_argument('--bl', dest='bl', default=1.8, type=float, help='Bond length threshold for clustering')
    parser.add_argument('--cl', dest='cl', default=4.0, type=float, help='Cube length for visualization')
    parser.add_argument('--il', dest='il', default=None, type=float, help='Length for getting interstitial sites')
    args = parser.parse_args()

    calc_info = []
    total_data = []
    total_idx = []

    # Get list of XSF files and sort them by band index
    xsf_files = glob.glob('pwscf_*.xsf')
    xsfidx = np.argsort([int(s.split('_')[3][1:4]) for s in xsf_files])

    for i, ixsf in enumerate(xsfidx):
        xsf_file = xsf_files[ixsf]
        kband = xsf_file.split('_')[3][1:4]
        if i == 0:
            pppx = PostPPX(xsf_file, args.bl, args.il)
        params = xsf.get_params_from_xsf(xsf_file)

        # Get centers of charge and charge ratios
        centers, rho_ratios = pppx.get_centers(params["grid_data"])

        # Select clusters with significant charge density
        target_idx = np.where(rho_ratios > 0.1)[0]

        # Generate cube data for visualization
        psi_per_band = pppx.generate_cube_data(params["grid_data"], centers[target_idx], xsf_name=f'{kband}', cube_length=args.cl)

        for j, center in enumerate(centers[target_idx]):
            total_data.append(psi_per_band[j])
            calc_info.append([target_idx[j], kband, center])
            total_idx.append((target_idx[j], kband))

    # Save data and calculation information
    np.save('image32x32x32', np.array(total_data))
    np.save('image_info.npy', np.array(total_idx))
    output_csv(calc_info)

    with open('calc.dat', 'w') as fw:
        fw.write(f'{len(calc_info)}\n')
        for info in calc_info:
            fw.write(f'{info[0]}  {info[1]}\n')
