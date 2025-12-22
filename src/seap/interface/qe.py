import re
import os
import sys
import subprocess

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.units import AU2ANG
from common.lattice import crys2cart, cart2crys

# This module provides classes for reading and processing input files for the PWscf code.
# The classes allow for reading and processing the input file, and retrieving atomic positions in different units.   


class PWIn:
    def __init__(self, filename):
        """
        Initialize the PWIn object and read the input file.

        Parameters
        ----------
        filename : str
            The name of the input file to be read.
        """
        try:
            if os.path.exists(filename):
                self._read_all(filename)
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            print(filename + " does not exist.")

    def _read_all(self, filename):
        """
        Read the entire input file and process its contents.

        Parameters
        ----------
        filename : str
            The name of the input file to be read.
        """
        with open(filename, "r") as fr:
            lines = fr.readlines()
        self._read_namelist(lines)
        self.ibrav = int(self.system["ibrav"])
        # If ibrav is not 0, exit the program as it is not implemented.
        # if (ibrav != 0):
        #     print(f"ibrav = {ibrav} is not implemented.")
        #     sys.exit(1) 
        idx = np.where([("/\n" in s.replace(" ", "")) for s in lines])[0][-1]
        if self.ibrav == 0:
            self.cell_parameters, self.alat = self._read_cell_parameters(lines[idx+1:])
        else:
            self.cell_parameters, self.alat = self._get_bravais_lattice()
        self.atypes = self._read_atomic_species(lines[idx+1:])
        self.atoms = self._read_atoms(lines[idx+1:])

    def _read_namelist(self, lines):
        """
        Read and process the namelist section of the input file.

        Parameters
        ----------
        lines : list of str
            List of lines from the input file.
        """
        # Check namelist format
        num_start = sum(("&" in l) for l in lines)
        num_end = sum(("/\n" in l.replace(" ", "")) for l in lines)
        if num_start != num_end:
            print("wrong namelist file.")
            sys.exit(1)
        lines_iter = iter(lines)
        nml = {"control": {}, "system": {}, "electrons": {}}
        for line in lines_iter:
            if "&" in line:
                group = line.replace("\n", "").split("&")[-1]
                lnext = next(lines_iter).replace(" ", "")
                while "/\n" not in lnext:
                    key, value = lnext.replace("\n", "").split("=")
                    nml[group][key] = value
                    lnext = next(lines_iter).replace(" ", "")
        self.control = nml["control"]
        self.system = nml["system"]
        self.electrons = nml["electrons"]
    
    def _get_bravais_lattice(self):
        """
        Get the Bravais lattice parameters based on the ibrav value.

        Returns
        -------
        tuple
            A tuple containing the lattice parameters and the lattice constant.
        """
        if self.ibrav == -3: 
            a = float(self.system["a"])
            lattice = [[-a/2, a/2, a/2], [a/2, -a/2, a/2], [a/2, a/2, -a/2]]
            a *= np.sqrt(3)/2
        elif self.ibrav == 5:
            a = float(self.system["a"])
            cosg = float(self.system["cosab"])
            tx = np.sqrt((1 - cosg)/2)
            ty = np.sqrt((1 - cosg)/6)
            tz = np.sqrt((1 + 2*cosg)/3)
            lattice = np.array([[tx, -ty, tz], [0, 2*ty, tz], [-tx, -ty, tz]])*a
        elif self.ibrav == 6:
            a = float(self.system["a"])
            c = float(self.system["c"])
            lattice = [[a, 0, 0], [0, a, 0], [0, 0, c]]
        else:
            sys.exit(f"ibrav == {self.ibrav} is not implemented.")
        return np.array(lattice), a

    def _read_cell_parameters(self, lines):
        """
        Read the cell parameters from the input file.

        Parameters
        ----------
        lines : list of str
            List of lines from the input file.

        Returns
        -------
        tuple
            A tuple containing the cell parameters and the scale factor.
        """
        cell = np.zeros((3, 3), dtype=float)
        # cell = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        pattern = r"\s*CELL_PARAMETERS\s*\{?\s*(\w*)\s*\}?"
        for i, line in enumerate(lines):
            match = re.match(pattern, line)
            if match is not None:
                cell_unit = match.group(1)
                # if (cell_unit != "alat"):
                #     print(f"cell unit should be alat.")
                #     sys.exit(1)
                for j in range(3):
                    # cell[j] = [float(x) for x in lines[i+j+1].split()]
                    cell[j] = np.array([float(x) for x in lines[i+j+1].split()])
                if cell_unit == "alat":
                    scale = float(self.system["A"])
                    cell *= scale
                elif cell_unit == "angstrom":
                    scale = np.linalg.norm(np.array(cell)[0])
        return cell, scale

    def _read_atomic_species(self, lines):
        """
        Read the atomic species information from the input file.

        Parameters
        ----------
        lines : list of str
            List of lines from the input file.

        Returns
        -------
        dict
            A dictionary containing the atomic species information.
        """
        atypes = {}
        for i, line in enumerate(lines):
            if "ATOMIC_SPECIES" in line:
                for j in range(int(self.system["ntyp"])):
                    atype, mass, psp = lines[i+j+1].split()
                    atypes[atype] = [mass, psp]
        return atypes

    def _read_atoms(self, lines):
        """
        Read the atomic positions from the input file.

        Parameters
        ----------
        lines : list of str
            List of lines from the input file.

        Returns
        -------
        list
            A list containing the atomic positions.
        """
        atoms = []
        pattern = r"\s*ATOMIC_POSITIONS\s*\{?\s*(\w*)\s*\}?"
        for i, line in enumerate(lines):
            match = re.match(pattern, line) 
            if match is not None:
                self.atomic_pos_type = match.group(1).lower()
                for j in range(int(self.system["nat"])):
                    atype, *r = lines[i+j+1].split()
                    atoms.append([atype, [float(x) for x in r]])
        return atoms 
   
    def get_atomic_positions(self, ounit=None):
        """
        Get the atomic positions in the specified units.

        Parameters
        ----------
        ounit : str, optional
            The desired units for the atomic positions. If not specified, the original units are used.

        Returns
        -------
        np.ndarray
            The atomic positions in the specified units.
        """
        positions = np.array([atom[1] for atom in self.atoms])
        ounit = ounit if ounit is not None else self.atomic_pos_type
        if ounit == self.atomic_pos_type:
            return positions
        # Step 1: convert the units of atomic positions to angstrom
        scale_in = 1.0
        if self.atomic_pos_type == "bohr":
            scale_in = AU2ANG
        elif self.atomic_pos_type == "alat":
            scale_in = float(self.system["A"])
        elif self.atomic_pos_type == "crystal":
            positions = crys2cart(positions, self.cell_parameters)
        positions *= scale_in
        # Step 2: convert the units of atomic positions from angstroms to ounit
        scale_out = 1.0
        if ounit == "crystal":
            positions = cart2crys(positions, self.cell_parameters)
        elif ounit == "alat":
            scale_out = 1.0 / float(self.system["A"])
        elif ounit == "bohr":
            scale_out = ang2au
        positions *= scale_out
        return positions

class PPIn:
    def __init__(self, set_region=True):
        """
        Initialize the PPIn object and set default namelist values.

        Parameters
        ----------
        set_region : bool, optional
            Whether to set the region for plotting, by default True.
        """
        self.set_region = set_region
        self._set_default_namelist()

    def _set_default_namelist(self):
        """
        Set the default namelist values for inputpp and plot.
        """
        self.inputpp = {
            'prefix': "'pwscf'",
            'outdir': "'../work/'",
            'plot_num': 7,
            'kpoint(1)': 1,
            'kband(1)': 1,
            'lsign': '.true.',
        } 
        if self.set_region:
            self.plot = {
                'iflag': 3,
                'output_format': 3,
                'fileout': 'wfc.xsf',
                'e1(1)': 1.0,
                'e1(2)': 0.0,
                'e1(3)': 0.0,
                'e2(1)': 0.0,
                'e2(2)': 1.0,
                'e2(3)': 0.0,
                'e3(1)': 0.0,
                'e3(2)': 0.0,
                'e3(3)': 1.0,
                'x0(1)': 0.0,
                'x0(2)': 0.0,
                'x0(3)': 0.0,
                'nx': 64,
                'ny': 64,
                'nz': 64,
            }
        else:
            self.plot = {
                'iflag': 3,
                'output_format': 5,
                'fileout': 'wfc.xsf',
                'nx': 64,
                'ny': 64,
                'nz': 64,
            }

    def set_prefix(self, prefix):
        """
        Set the prefix for the inputpp namelist.

        Parameters
        ----------
        prefix : str
            The prefix to be set.
        """
        self.inputpp['prefix'] = prefix

    def set_outdir(self, dirname): 
        """
        Set the output directory for the inputpp namelist.

        Parameters
        ----------
        dirname : str
            The output directory to be set.
        """
        self.inputpp['outdir'] = dirname

    def set_kband(self, idx_k1, idx_k2=None):
        """
        Set the kband indices for the inputpp namelist.

        Parameters
        ----------
        idx_k1 : int
            The first kband index.
        idx_k2 : int, optional
            The second kband index, by default None.
        """
        self.inputpp['kband(1)'] = idx_k1
        if idx_k2 is not None:
            self.inputpp['kband(2)'] = idx_k2
    
    def set_fileout(self, oprefix):
        """
        Set the output file name for the plot namelist.

        Parameters
        ----------
        oprefix : str
            The prefix for the output file name.
        """
        if not self.set_region:
            self.plot['fileout'] = f'{oprefix}.xsf'
        else:
            self.plot['fileout'] = f'{oprefix}.wfc.xsf'

    def set_cubic_e(self, length_alat):
        """
        Set the cubic edge length for the plot namelist.

        Parameters
        ----------
        length_alat : float
            The length of the cubic edge in alat units.
        """
        if not self.set_region:
            sys.exit('Please set set_region=True.')
        self.plot['e1(1)'] = length_alat
        self.plot['e2(2)'] = length_alat
        self.plot['e3(3)'] = length_alat

    def set_origin(self, pos_alat):
        """
        Set the origin for the plot namelist.

        Parameters
        ----------
        pos_alat : list of float
            The origin position in alat units.
        """
        if not self.set_region:
            sys.exit('Please set set_region=True.')
        self.plot['x0(1)'] = pos_alat[0]
        self.plot['x0(2)'] = pos_alat[1]
        self.plot['x0(3)'] = pos_alat[2]

    def set_mesh(self, mesh):
        """
        Set the mesh size for the plot namelist.

        Parameters
        ----------
        mesh : list of int
            The mesh size in the format [nx, ny, nz].
        """
        self.plot['nx'] = mesh[0]
        self.plot['ny'] = mesh[1]
        self.plot['nz'] = mesh[2]

    def write(self, o_file, folder='.'):
        """
        Write the namelist to a file.

        Parameters
        ----------
        o_file : str
            The name of the output file.
        folder : str, optional
            The folder to save the output file, by default '.'.
        """
        if folder != '.':
            subprocess.call(f'mkdir -p {folder}', shell=True, cwd='./')
        pp = ''
        pp += self._stringify_group('inputpp', self.inputpp)
        pp += self._stringify_group('plot', self.plot)
        open(f'{folder}/{o_file}', 'w').write(pp)
    
    def _stringify_group(self, group_name, group):
        """
        Convert a namelist group to a string format.

        Parameters
        ----------
        group_name : str
            The name of the namelist group.
        group : dict
            The namelist group dictionary.

        Returns
        -------
        str
            The string representation of the namelist group.
        """
        s = f'&{group_name}\n'
        for keyword in group:
            if group[keyword] is not None:
                s += f'  {keyword:<14s} = {group[keyword]}\n'
        s += '/\n'
        return s


if __name__ == "__main__":
    # Create an instance of PWIn with the input file "scf.in"
    qe_input = PWIn("./scf.in")
    # Get the atomic positions in angstrom units
    positions = qe_input.get_atomic_positions(ounit='angstrom')
    # Print the atomic positions
    print(positions)
