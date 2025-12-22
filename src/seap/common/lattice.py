import numpy as np

# This module provides functions for converting between crystal and Cartesian coordinates.

def crys2cart(pos_crys, lattice):
    """
    Convert crystal coordinates to Cartesian coordinates.

    Parameters
    ----------
    pos_crys : np.ndarray
        Array of positions in crystal coordinates.
    lattice : np.ndarray
        Lattice vectors defining the unit cell.

    Returns
    -------
    np.ndarray
        Array of positions in Cartesian coordinates.
    """
    # Perform matrix multiplication to convert crystal coordinates to Cartesian coordinates
    return np.einsum('ij,jk->ik', pos_crys, lattice)

def cart2crys(pos_cart, lattice):
    """
    Convert Cartesian coordinates to crystal coordinates.

    Parameters
    ----------
    pos_cart : np.ndarray
        Array of positions in Cartesian coordinates.
    lattice : np.ndarray
        Lattice vectors defining the unit cell.

    Returns
    -------
    np.ndarray
        Array of positions in crystal coordinates.
    """
    # Calculate the inverse of the lattice matrix
    lattice_inv = np.linalg.inv(lattice)
    # Perform matrix multiplication to convert Cartesian coordinates to crystal coordinates
    return np.einsum('ij,jk->ik', pos_cart, lattice_inv)
