import math
import numpy as np
from itertools import product

# This module provides functions for generating grids of points in Cartesian and polar coordinates.
# The functions are used for scatter plotting and for generating data for machine learning models.

def grid_for_scatter(nlist):
    """
    Generate a grid for scatter plotting.

    Parameters
    ----------
    nlist : list of int
        List containing the number of divisions along each axis.

    Returns
    -------
    tuple of np.ndarray
        Flattened arrays of x, y, and z coordinates for scatter plotting.

    Examples
    --------
    >>> grid_for_scatter([2, 2, 2])
    (array([0, 0, 0, 0, 1, 1, 1, 1]), array([0, 0, 1, 1, 0, 0, 1, 1]), array([0, 1, 0, 1, 0, 1, 0, 1]))
    """
    # Create a list of arrays for each axis based on the number of divisions
    xyz = [np.arange(n) for n in nlist]
    # Generate a meshgrid and flatten the arrays
    x, y, z = np.meshgrid(*xyz, indexing="ij")
    return x.ravel(), y.ravel(), z.ravel()

def cartesian_grid(n_div_list, size):
    """
    Generate a Cartesian grid within a specified size.

    Parameters
    ----------
    n_div_list : list of int
        List containing the number of divisions along each axis.
    size : float
        Size of the grid along each axis.

    Returns
    -------
    np.ndarray
        Array of Cartesian coordinates.

    Examples
    --------
    >>> cartesian_grid([2, 2, 2], 2.0)
    array([[-0.5, -0.5, -0.5],
           [-0.5, -0.5,  0.5],
           [-0.5,  0.5, -0.5],
           [-0.5,  0.5,  0.5],
           [ 0.5, -0.5, -0.5],
           [ 0.5, -0.5,  0.5],
           [ 0.5,  0.5, -0.5],
           [ 0.5,  0.5,  0.5]])
    """
    # Calculate step size for each axis
    steps = [float(size / n_div) for n_div in n_div_list]
    # Generate grid points for each axis
    xyz = [np.arange(-size/2.0 + step/2.0, size/2.0, step) for step in steps]
    # Create a Cartesian product of the grid points
    mesh = np.array(list(product(xyz[0], xyz[1], xyz[2])))
    return mesh

def extract_inscribed_ball(pos_cart, size):
    """
    Extract points within an inscribed sphere from a Cartesian grid.

    Parameters
    ----------
    pos_cart : np.ndarray
        Array of Cartesian coordinates.
    size : float
        Size of the grid along each axis.

    Returns
    -------
    tuple of (np.ndarray, list of int)
        Polar coordinates of points within the inscribed sphere and their indices.

    Examples
    --------
    >>> extract_inscribed_ball(np.array([[0, 0, 0], [1, 1, 1]]), 2.0)
    (array([[1.73205081, 0.95531662, 0.78539816]]), [1])
    """
    rmax = size / 2.0
    pos_cart_in_ball = []
    map_idx = []
    # Iterate over each position and check if it lies within the sphere
    for ir, pos in enumerate(pos_cart):
        if np.all(pos == 0.0):
            continue
        norm = np.linalg.norm(pos)
        if norm <= rmax:
            pos_cart_in_ball.append(pos)
            map_idx.append(ir)
    # Convert Cartesian coordinates to polar coordinates
    pos_pol_in_ball = cartesian2polar(np.array(pos_cart_in_ball))
    return pos_pol_in_ball, map_idx

def cartesian2polar(pos_cart):
    """
    Convert Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    pos_cart : np.ndarray
        Array of Cartesian coordinates.

    Returns
    -------
    np.ndarray
        Array of polar coordinates.

    Examples
    --------
    >>> cartesian2polar(np.array([[1, 1, 1]]))
    array([[1.73205081, 0.95531662, 0.78539816]])
    """
    num_pos = pos_cart.shape[0]
    pos_pol = np.empty([num_pos, 3], dtype=np.float64)
    # Calculate radial distance
    pos_pol[:, 0] = np.linalg.norm(pos_cart, axis=1)
    # Calculate polar angle (handle division by zero for points at origin)
    with np.errstate(divide='ignore', invalid='ignore'):
        pos_pol[:, 1] = np.arccos(pos_cart[:, 2] / pos_pol[:, 0])
    # Calculate azimuthal angle
    phi_tmp = np.arctan2(pos_cart[:, 1], pos_cart[:, 0])
    # Convert azimuthal angle range from [-pi, pi] to [0, 2pi]
    for i, phi in enumerate(phi_tmp):
        pos_pol[i, 2] = math.fmod(phi + 2 * np.pi, 2 * np.pi)
    # Handle NaN values in polar angle
    nan_index = np.isnan(pos_pol[:, 1])
    pos_pol[nan_index, 1] = 0
    return pos_pol

if __name__ == "__main__":
    # Example usage of cartesian_grid function
    nlist = [4, 4, 4]
    grid_cart = cartesian_grid(nlist, 2.0)
    print(grid_cart)
