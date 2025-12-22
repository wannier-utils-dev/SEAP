import sys
import bisect
import argparse
import os

import numpy as np


from pymatgen.core import Structure, Lattice
from doped.generation import get_interstitial_sites

# Handle both relative imports (when used as module) and absolute imports (when run directly)
#try:
#    from seap.interface.qe import PWIn, PPIn
#    from seap.common.molecule import atoms2molecules
#except ImportError:
    # Add the src directory to the path for direct execution
if __package__ in (None, ""):
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from seap.interface.qe import PWIn, PPIn
from seap.common.molecule import atoms2molecules

# This script is used to select the bands for the Wannier90 calculation.
# It reads the SCF input and output files, and the band data file.
# It then counts the number of carbon-containing molecules, and finds the nearest bands to the Fermi energy.
# It then generates initial PP input files for each band, and writes the selected band indices to 'band.dat' file.
# The script also allows for optional arguments to specify the band data file and the bond length threshold.
# The script outputs the number of bands to 'nbnd' file and the molecular information to 'mol.info' file.

def main(path, scf_in, scf_out, bond_length, interstitial_length=None, band_dat=None):
    """
    Main function to process the input files and select bands.

    Parameters
    ----------
    path : str
        Path to the work directory for DFT calculation.
    scf_in : str
        Path to the SCF input file.
    scf_out : str
        Path to the SCF output file.
    bond_length : float
        Bond length threshold for molecule detection.
    interstitial_length : float
        Length for searching voids
    band_dat : str, optional
        Path to the file specifying the bands.

    Returns
    -------
    None
    """
    # Read SCF input file
    scf = PWIn(scf_in)
    elements = [atom[0] for atom in scf.atoms]
    primvecs = scf.cell_parameters
    pos_crys = scf.get_atomic_positions(ounit='crystal')
    primvecs = np.round(primvecs, 6)
    pos_crys = np.round(pos_crys, 6)

    if interstitial_length:

        lattice = Lattice(primvecs)
        species = [atom[0] for atom in scf.atoms]
        coords = pos_crys
        structure = Structure(lattice, species, coords, coords_are_cartesian=False)
       
        interstitial_sites = get_interstitial_sites(structure,interstitial_gen_kwargs={"min_dist":interstitial_length})
    
        for _ideal_sites, _multiplicity, equiv_frac_coords in interstitial_sites:
            for frac_coord in equiv_frac_coords:
                pos_crys = np.concatenate([pos_crys,frac_coord.reshape(1,-1)])
                elements.append("X")
    
    # Detect molecules based on bond length threshold
    molecules, positions = atoms2molecules(primvecs, pos_crys, threshold=bond_length)
    output_mol_info(elements, molecules, positions)

    # Count the number of carbon-containing molecules
    ncmol = count_cmol(elements, molecules)
    
    # Extract eigenvalues, Fermi energy, and k-points from SCF output file
    eigvals, ef, kpoints = get_data_from_pwout(scf_out)
    
    # Find the index of the k-point at Gamma (0,0,0)
    ik0 = 0
    for i, k in enumerate(kpoints):
        if np.all(k == 0.):
            ik0 = i
            break

    # Process band data if provided
    if band_dat:
        ibands = []
        with open(band_dat, 'r') as fr:
            nrow = int(fr.readline())
            for i in range(nrow):
                ib = int(fr.readline())
                ibands.append(ib-1)
        ibands.sort()
        if ncmol != 0:
            ref_ibands = find_nearest_bands(eigvals[ik0], ef, ncmol)
            max_band_idx = max(ref_ibands)
        else:
            nscf = PWIn(scf_in.replace('scf', 'nscf'))
            max_band_idx = int(nscf.system['nbnd']) - 1
    else:
        if ncmol != 0:
            ibands = find_nearest_bands(eigvals[ik0], ef, ncmol)
            ibands.sort()
            max_band_idx = max(ibands)
        else:
            sys.exit('the automatic band selection for inorganic crystals has not been implemented yet')

    # Write the number of bands to 'nbnd' file
    with open('nbnd', 'w') as fw:
        fw.write(f'{max_band_idx + 1}')
 
    # Generate initial PP input files for each band
    for ib in ibands:
        get_initial_ppin(path, ib, primvecs)

    # Write the selected band indices to 'band.dat' file
    write_band_dat(ibands)

def count_cmol(elements, molecules):
    """
    Count the number of carbon-containing molecules.

    Parameters
    ----------
    elements : list of str
        List of element symbols for each atom.
    molecules : list of list of int
        List of molecules, each represented by a list of atom indices.

    Returns
    -------
    int
        Number of carbon-containing molecules.
    """
    ncmol = 0
    for molecule in molecules:
        elements_in_mol = [elements[i] for i in molecule]
        if 'C' in elements_in_mol:
            ncmol += 1
    return ncmol

def output_mol_info(elements, molecules, positions):
    """
    Output molecular information to 'mol.info' file.

    Parameters
    ----------
    elements : list of str
        List of element symbols for each atom.
    molecules : list of list of int
        List of molecules, each represented by a list of atom indices.
    positions : list of list of float
        List of atomic positions.
    Returns
    -------
    None
    """
    with open('mol.info', 'w') as fw:
        lines = []
        for i, molecule in enumerate(molecules):
            lines.append(f'molecule : {i}\n')
            for j in molecule:
                elm = elements[j]
                pos = positions[j]
                lines.append(f'{elm:>3}:{pos[0]:>10.6f}{pos[1]:>10.6f}{pos[2]:>10.6f}\n')

        fw.writelines(lines)

def get_data_from_pwout(nscf_out):
    """
    Extract eigenvalues, Fermi energy, and k-points from SCF output file.

    Parameters
    ----------
    nscf_out : str
        Path to the SCF output file.

    Returns
    -------
    tuple of (np.ndarray, float, np.ndarray)
        Eigenvalues, Fermi energy, and k-points.
    """
    with open(nscf_out, "r") as fr:
        lines = fr.readlines()
        eigvals = []
        kpoints = []
        for i, line in enumerate(lines):
            if "Kohn-Sham" in line:
                num_band = int(line.split()[-1])
            if "number of k points" in line:
                num_k = int(line.split()[4])
                for j in range(i+2, i+2+num_k):
                    kstr = lines[j].split("=")[1].split("(")[1].split(")")[0]
                    kpoints.append(np.array([float(s) for s in kstr.split()]))
            if " k =" in line:
                nl = (num_band // 8) + 1
                enk_str = "".join(lines[i+2:i+2+nl]) 
                eigvals.append([float(e) for e in enk_str.split()])
            if "the Fermi" in line:
                ef = float(line.split()[-2])
    return np.array(eigvals), ef, np.array(kpoints)

def find_nearest_bands(eigvals, ef, nmol):
    """
    Find the nearest bands to the Fermi energy.

    Parameters
    ----------
    eigvals : np.ndarray
        Array of eigenvalues.
    ef : float
        Fermi energy.
    nmol : int
        Number of molecules.

    Returns
    -------
    list of int
        Indices of the nearest bands.
    """
    neig = len(eigvals)
    closest_idx = bisect.bisect_left(eigvals, ef)
    if (closest_idx == 0):
        nearest_indices = list(range(nmol))
    elif (closest_idx == neig):
        nearest_indices = list(range(neig - nmol, neig))
    else:
        left_idx = closest_idx - 1
        right_idx = closest_idx
        
        nearest_indices = []
        while len(nearest_indices) < nmol:
            cond_1 = right_idx >= neig
            cond_2 = left_idx >= 0
            cond_3 = abs(ef - eigvals[left_idx]) <= abs(ef - eigvals[right_idx])
            if cond_1 or (cond_2 and cond_3):
                nearest_indices.append(left_idx)
                left_idx -= 1
            else:
                nearest_indices.append(right_idx)
                right_idx += 1
    return nearest_indices

def get_initial_ppin(path, iband, primvecs):
    """
    Generate initial PP input file for a given band.

    Parameters
    ----------
    path : str
        Path to the work directory.
    iband : int
        Band index.
    primvecs : list of list of float
        Primitive vectors.

    Returns
    -------
    None
    """
    pp = PPIn(set_region=False)
    pp.set_kband(iband+1)
    pp.set_prefix('\'pwscf\'')
    pp.set_outdir(f'\'{path}/work\'')
    pp.set_fileout(f'pwscf_wfn_K001_B{str(iband+1).zfill(3)}')
    pp.write(f'pwscf.pp_{str(iband+1).zfill(3)}.in')

def write_band_dat(band_indices):
    """
    Write the selected band indices to 'band.dat' file.

    Parameters
    ----------
    band_indices : list of int
        List of selected band indices.

    Returns
    -------
    None
    """
    with open('band.dat', 'w') as fw:
        fw.write(f'{len(band_indices)}\n')
        for ib in band_indices:
            fw.write(f'{str(ib+1).zfill(3)}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='select_bands.py')
    parser.add_argument('path', type=str, help='path to the work directory for DFT calculation')
    parser.add_argument('--man', dest='fband', default=None, type=str, help='file specifying the bands')
    parser.add_argument('--bl', dest='bl', default=1.8, type=float, help='bond length threshold')
    parser.add_argument('--il', dest='il', default=None, type=float, help='length for getting interstitial sites')
    args = parser.parse_args()

    scf_in = args.path + '/scf.in'
    scf_out = args.path + '/scf.out'
    if args.fband:
        band_dat = args.path + '/' + args.fband
        main(args.path, scf_in, scf_out, args.bl, interstitial_length=args.il, band_dat=band_dat)
    else:
        main(args.path, scf_in, scf_out, args.bl, interstitial_length=args.il)
