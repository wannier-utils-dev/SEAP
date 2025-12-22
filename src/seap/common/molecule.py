import itertools

import numpy as np

# This module provides functions for dividing atoms into molecules based on a threshold distance.
# The functions are used for grouping atoms in a molecule and for identifying periodic images.

def atoms2molecules(lattice, positions, threshold=1.8):
    """
    Divide groups of atoms into molecules using a threshold distance (in Angstrom) 
    between atoms. Lattice means direct lattice vectors given in Cartesian axis:
        a1(x) a1(y) a1(z) -- 1st vector
        a2(x) a2(y) a2(z) -- 2nd vector
        a3(x) a3(y) a3(z) -- 3rd vector
    Positions are atomic positions in fractional coordinates.

    Returns
    -------
    - molecules is a list where the indices of atoms are grouped together as sets
      for each molecule
    - periodic_images is a ndarray of the atomic positions in fractional
      coordinates used for distance calculation
    """
    molecules = []  # List to store sets of atom indices for each molecule
    periodic_images = positions.copy()  # Copy of positions for periodic image calculations
    # Generate all possible periodic boundary condition (PBC) directions
    pbc_directions = list(itertools.product([-2, -1, 0, 1, 2], repeat=3))
    
    # Iterate over each atom in the positions list
    for i, atom_i in enumerate(positions):
        # Skip atoms that are already part of a molecule
        if any(i in molecule for molecule in molecules):
            continue
        molecule = set()  # Set to store indices of atoms in the current molecule
        queue = [i]  # Queue for breadth-first search (BFS) starting with the current atom
        
        # Perform BFS to find all atoms in the current molecule
        while queue:
            curt_atom = queue.pop(0)  # Get the current atom index from the queue
            if curt_atom in molecule:
                continue  # Skip if the atom is already in the molecule
            molecule.add(curt_atom)  # Add the current atom to the molecule
            
            # Check all other atoms to see if they belong to the current molecule
            for j, atom_j in enumerate(periodic_images):
                if j != curt_atom and j not in molecule:
                    # Check all periodic images of atom_j
                    for t in pbc_directions:
                        atom_j_pim = atom_j + t  # Calculate the periodic image of atom_j
                        # Calculate the distance between the current atom and the periodic image
                        diff = (periodic_images[curt_atom] - atom_j_pim) @ lattice
                        distance = np.linalg.norm(diff)
                        # If the distance is within the threshold, add atom_j to the queue
                        if distance <= threshold and j not in queue:
                            queue.append(j)
                            periodic_images[j] = atom_j_pim  # Update the position of atom_j
                            break
        molecules.append(molecule)  # Add the found molecule to the list of molecules
    return molecules, periodic_images  # Return the list of molecules and updated positions


if __name__ == '__main__':
    import xsf

    xsf_file = 'pp.wfc.xsf'  # Path to the XSF file containing atomic data
    params = xsf.get_params_from_xsf(xsf_file)  # Extract parameters from the XSF file
    grid = params['grid_info']  # Grid information
    data = params['grid_data']  # Grid data
    rorg = params['pos_org']  # Origin of the positions
    natom = params['num_atom']  # Number of atoms
    spanning = params['span_vec']  # Spanning vectors
    prim_vecs = params['prim_vec']  # Primitive vectors
    atomic_info = params['prim_coord']  # Atomic coordinates and element information

    elements = []  # List to store element symbols
    frac_coords = np.zeros((natom, 3), dtype=float)  # Array for fractional coordinates
    ainv = np.linalg.inv(prim_vecs)  # Inverse of the primitive vectors for conversion
    
    # Convert atomic coordinates from Cartesian to fractional
    for i, ai in enumerate(atomic_info):
        elements.append(ai[0])  # Append the element symbol
        cart_coord = np.array([float(ai[1]), float(ai[2]), float(ai[3])])  # Cartesian coordinates
        frac_coords[i] = cart_coord @ ainv  # Convert to fractional coordinates

    # Divide atoms into molecules based on the threshold distance
    molecules, positions_ref = atoms2molecules(prim_vecs, frac_coords)

    # Print information about each molecule
    for i, molecule in enumerate(molecules):
        elements_mol = [elements[j] for j in molecule]  # Get element symbols for the molecule
        print(f'Molecule {i+1}: {elements_mol}')  # Print the molecule number and elements
        positions = np.array([positions_ref[j] for j in molecule])  # Get positions of the molecule
        center = np.sum(positions, axis=0) / len(positions)  # Calculate the center of the molecule
        print(f'- center = {center}')  # Print the center of the molecule
