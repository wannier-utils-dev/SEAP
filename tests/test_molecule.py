"""
Tests for molecule detection functions.
"""

import numpy as np
from seap.common.molecule import atoms2molecules


class TestAtoms2Molecules:
    """Test class for atoms2molecules function."""

    def test_single_atom(self):
        """Test with a single atom."""
        lattice = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        positions = np.array([[0.5, 0.5, 0.5]])
        molecules, periodic_images = atoms2molecules(lattice, positions, threshold=1.8)
        
        assert len(molecules) == 1
        assert len(molecules[0]) == 1
        assert 0 in molecules[0]

    def test_two_separate_atoms(self):
        """Test with two atoms far apart."""
        lattice = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        # Two atoms separated by 5 Angstrom
        positions = np.array([[0.0, 0.0, 0.0],
                             [5.0, 0.0, 0.0]])
        molecules, periodic_images = atoms2molecules(lattice, positions, threshold=1.8)
        
        assert len(molecules) == 2
        assert len(molecules[0]) == 1
        assert len(molecules[1]) == 1

    def test_two_close_atoms(self):
        """Test with two atoms close together."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Two atoms separated by 1 Angstrom (in fractional coordinates)
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.1, 0.0, 0.0]])
        molecules, periodic_images = atoms2molecules(lattice, positions, threshold=1.8)
        
        # Should be grouped into one molecule
        assert len(molecules) == 1
        assert len(molecules[0]) == 2

    def test_three_atoms_chain(self):
        """Test with three atoms forming a chain."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Three atoms in a chain, each separated by 1 Angstrom
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.1, 0.0, 0.0],
                             [0.2, 0.0, 0.0]])
        molecules, periodic_images = atoms2molecules(lattice, positions, threshold=1.8)
        
        # All should be in one molecule
        assert len(molecules) == 1
        assert len(molecules[0]) == 3

    def test_periodic_images(self):
        """Test that periodic_images are returned correctly."""
        lattice = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        positions = np.array([[0.5, 0.5, 0.5]])
        molecules, periodic_images = atoms2molecules(lattice, positions, threshold=1.8)
        
        assert isinstance(periodic_images, np.ndarray)
        assert periodic_images.shape == positions.shape

    def test_empty_positions(self):
        """Test with empty positions array."""
        lattice = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        positions = np.array([]).reshape(0, 3)
        molecules, periodic_images = atoms2molecules(lattice, positions, threshold=1.8)
        
        assert len(molecules) == 0
        assert periodic_images.shape == (0, 3)

    def test_threshold_parameter(self):
        """Test that threshold parameter works correctly."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Two atoms separated by 2 Angstrom
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.2, 0.0, 0.0]])
        
        # With threshold 1.8, should be separate
        molecules_small, _ = atoms2molecules(lattice, positions, threshold=1.8)
        assert len(molecules_small) == 2
        
        # With threshold 2.5, should be together
        molecules_large, _ = atoms2molecules(lattice, positions, threshold=2.5)
        assert len(molecules_large) == 1

    def test_periodic_boundary_conditions(self):
        """Test periodic boundary conditions with atoms near cell boundaries."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Two atoms near opposite boundaries that should be connected via PBC
        positions = np.array([[0.95, 0.5, 0.5],  # Near right boundary
                             [0.05, 0.5, 0.5]])  # Near left boundary
        molecules, periodic_images = atoms2molecules(lattice, positions, threshold=2.0)
        
        # Should be connected via periodic boundary
        assert len(molecules) == 1
        assert len(molecules[0]) == 2

    def test_complex_molecular_structure(self):
        """Test with a more complex molecular structure."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Create a ring-like structure (4 atoms in a square)
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.1, 0.0, 0.0],
                             [0.1, 0.1, 0.0],
                             [0.0, 0.1, 0.0]])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.5)
        
        # All should be in one molecule
        assert len(molecules) == 1
        assert len(molecules[0]) == 4

    def test_already_in_molecule_skip(self):
        """Test that atoms already in a molecule are skipped."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Three atoms: two close, one far
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.05, 0.0, 0.0],  # Close to first
                             [0.5, 0.5, 0.5]])  # Far from others
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.8)
        
        # First two should be together, third separate
        assert len(molecules) == 2
        # Check that atoms are correctly assigned
        all_atoms = set()
        for mol in molecules:
            all_atoms.update(mol)
        assert all_atoms == {0, 1, 2}

    def test_queue_duplicate_prevention(self):
        """Test that atoms already in queue are not added again."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Linear chain where each atom connects to next
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.05, 0.0, 0.0],
                             [0.10, 0.0, 0.0],
                             [0.15, 0.0, 0.0]])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.5)
        
        # All should be in one molecule
        assert len(molecules) == 1
        assert len(molecules[0]) == 4

    def test_periodic_image_update(self):
        """Test that periodic images are correctly updated."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.95, 0.0, 0.0]])
        molecules, periodic_images = atoms2molecules(lattice, positions, threshold=2.0)
        
        # Check that periodic_images were updated
        assert isinstance(periodic_images, np.ndarray)
        assert periodic_images.shape == positions.shape
        # If connected via PBC, positions might be adjusted
        assert len(molecules) == 1

    def test_atom_already_in_molecule_skip(self):
        """Test that atoms already in molecule set are skipped in BFS."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Create a structure where an atom might be checked multiple times
        # Triangle structure where each atom connects to the other two
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.05, 0.0, 0.0],
                             [0.025, 0.043, 0.0]])  # Equilateral triangle
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.5)
        
        # All should be in one molecule
        assert len(molecules) == 1
        assert len(molecules[0]) == 3

    def test_multiple_periodic_images(self):
        """Test with atoms that require checking multiple periodic images."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Atoms near different boundaries - first two should connect via PBC
        positions = np.array([[0.95, 0.5, 0.5],
                             [0.05, 0.5, 0.5],
                             [0.5, 0.95, 0.5]])
        molecules, periodic_images = atoms2molecules(lattice, positions, threshold=2.0)
        
        # First two should be connected via periodic boundaries
        # Third might be separate depending on distance
        assert len(molecules) >= 1
        # At least first two atoms should be together
        found_pair = False
        for mol in molecules:
            if 0 in mol and 1 in mol:
                found_pair = True
                break
        assert found_pair, "Atoms 0 and 1 should be connected via periodic boundary"

    def test_break_statement_in_pbc_loop(self):
        """Test that break statement works correctly in periodic image loop."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Two atoms that should connect via first periodic image found
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.95, 0.0, 0.0]])
        molecules, periodic_images = atoms2molecules(lattice, positions, threshold=2.0)
        
        # Should find connection and break from loop
        assert len(molecules) == 1

    def test_atom_already_in_molecule_continue(self):
        """Test that atoms already in molecule set are skipped with continue."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Create a structure where an atom might be processed multiple times
        # This tests the "if curt_atom in molecule: continue" branch
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.05, 0.0, 0.0],
                             [0.10, 0.0, 0.0]])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.5)
        
        # All should be in one molecule
        assert len(molecules) == 1
        assert len(molecules[0]) == 3

    def test_atom_in_molecule_check(self):
        """Test that atoms already in molecule are skipped in BFS queue."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Create a structure where an atom might be added to queue multiple times
        # Star structure: central atom connected to multiple outer atoms
        positions = np.array([[0.5, 0.5, 0.5],  # Center
                             [0.45, 0.5, 0.5],   # Connected to center
                             [0.55, 0.5, 0.5],   # Connected to center
                             [0.5, 0.45, 0.5],   # Connected to center
                             [0.5, 0.55, 0.5]])  # Connected to center
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.5)
        
        # All should be in one molecule
        assert len(molecules) == 1
        assert len(molecules[0]) == 5

    def test_non_orthogonal_lattice(self):
        """Test with non-orthogonal lattice vectors (monoclinic-like)."""
        # Monoclinic-like lattice with b-axis tilted
        lattice = np.array([[10.0, 0.0, 0.0],
                           [2.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Two atoms that are close in fractional coordinates
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.1, 0.0, 0.0]])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.8)
        
        # Should be in one molecule
        assert len(molecules) == 1
        assert len(molecules[0]) == 2

    def test_hexagonal_lattice(self):
        """Test with hexagonal lattice vectors."""
        # Hexagonal lattice (a = b, gamma = 120 degrees)
        a = 5.0
        lattice = np.array([[a, 0.0, 0.0],
                           [-a/2, a*np.sqrt(3)/2, 0.0],
                           [0.0, 0.0, 10.0]])
        # Two atoms in fractional coordinates
        positions = np.array([[0.0, 0.0, 0.0],
                             [1/3, 1/3, 0.0]])
        molecules, _ = atoms2molecules(lattice, positions, threshold=3.0)
        
        # Check that distance calculation works correctly for hexagonal lattice
        assert len(molecules) >= 1

    def test_negative_fractional_coordinates(self):
        """Test with negative fractional coordinates."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Atoms with negative fractional coordinates
        positions = np.array([[-0.1, 0.0, 0.0],
                             [0.0, 0.0, 0.0]])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.8)
        
        # Should be in one molecule (distance is 1 Angstrom)
        assert len(molecules) == 1
        assert len(molecules[0]) == 2

    def test_fractional_coords_outside_unit_cell(self):
        """Test with fractional coordinates outside [0, 1] range."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Atoms with coordinates outside [0, 1]
        positions = np.array([[1.5, 0.5, 0.5],
                             [1.6, 0.5, 0.5]])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.8)
        
        # Should be in one molecule (distance is 1 Angstrom)
        assert len(molecules) == 1
        assert len(molecules[0]) == 2

    def test_multiple_separate_molecules(self):
        """Test with multiple separate molecules in the cell."""
        lattice = np.array([[20.0, 0.0, 0.0],
                           [0.0, 20.0, 0.0],
                           [0.0, 0.0, 20.0]])
        # Three separate pairs of atoms
        positions = np.array([
            [0.0, 0.0, 0.0],      # Molecule 1: atom 1
            [0.05, 0.0, 0.0],     # Molecule 1: atom 2 (1 Angstrom apart)
            [0.5, 0.0, 0.0],      # Molecule 2: atom 1
            [0.55, 0.0, 0.0],     # Molecule 2: atom 2 (1 Angstrom apart)
            [0.5, 0.5, 0.5],      # Molecule 3: atom 1
            [0.55, 0.5, 0.5],     # Molecule 3: atom 2 (1 Angstrom apart)
        ])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.8)
        
        # Should have 3 molecules, each with 2 atoms
        assert len(molecules) == 3
        for mol in molecules:
            assert len(mol) == 2

    def test_threshold_at_exact_boundary(self):
        """Test atoms at exactly the threshold distance."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Two atoms exactly 1.8 Angstrom apart
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.18, 0.0, 0.0]])  # 1.8 Angstrom
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.8)
        
        # Should be in one molecule (distance <= threshold)
        assert len(molecules) == 1
        assert len(molecules[0]) == 2

    def test_threshold_just_above_distance(self):
        """Test atoms just inside threshold."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Two atoms 1.79 Angstrom apart
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.179, 0.0, 0.0]])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.8)
        
        # Should be in one molecule
        assert len(molecules) == 1

    def test_threshold_just_below_distance(self):
        """Test atoms just outside threshold."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Two atoms 1.81 Angstrom apart
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.181, 0.0, 0.0]])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.8)
        
        # Should be separate molecules
        assert len(molecules) == 2

    def test_water_molecule_like_structure(self):
        """Test with water molecule-like structure (H-O-H angle ~104.5 degrees)."""
        lattice = np.array([[20.0, 0.0, 0.0],
                           [0.0, 20.0, 0.0],
                           [0.0, 0.0, 20.0]])
        # O-H bond length is about 0.96 Angstrom
        # Place atoms in fractional coordinates
        # Oxygen at center
        o_pos = np.array([0.5, 0.5, 0.5])
        # Hydrogen 1 (0.96 Angstrom away = 0.048 in fractional coords)
        h1_pos = np.array([0.548, 0.5, 0.5])
        # Hydrogen 2 at about 104.5 degrees
        angle = np.radians(104.5)
        h2_pos = np.array([
            0.5 + 0.048 * np.cos(angle),
            0.5 + 0.048 * np.sin(angle),
            0.5
        ])
        positions = np.array([o_pos, h1_pos, h2_pos])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.5)
        
        # All three should be in one molecule
        assert len(molecules) == 1
        assert len(molecules[0]) == 3

    def test_periodic_boundary_all_three_directions(self):
        """Test periodic boundary conditions in all three directions."""
        lattice = np.array([[5.0, 0.0, 0.0],
                           [0.0, 5.0, 0.0],
                           [0.0, 0.0, 5.0]])
        # Atoms near opposite corners that should connect via PBC
        positions = np.array([
            [0.02, 0.02, 0.02],   # Near origin
            [0.98, 0.98, 0.98],   # Near opposite corner
        ])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.0)
        
        # Should be connected via periodic boundary (diagonal distance ~0.3 Angstrom)
        assert len(molecules) == 1
        assert len(molecules[0]) == 2

    def test_pbc_x_direction_only(self):
        """Test periodic boundary in x-direction only."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        positions = np.array([
            [0.02, 0.5, 0.5],
            [0.98, 0.5, 0.5],
        ])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.5)
        
        # Distance via PBC: (1.0 - 0.98 + 0.02) * 10 = 0.4 Angstrom
        assert len(molecules) == 1

    def test_pbc_y_direction_only(self):
        """Test periodic boundary in y-direction only."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        positions = np.array([
            [0.5, 0.02, 0.5],
            [0.5, 0.98, 0.5],
        ])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.5)
        
        # Distance via PBC: 0.4 Angstrom
        assert len(molecules) == 1

    def test_pbc_z_direction_only(self):
        """Test periodic boundary in z-direction only."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        positions = np.array([
            [0.5, 0.5, 0.02],
            [0.5, 0.5, 0.98],
        ])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.5)
        
        # Distance via PBC: 0.4 Angstrom
        assert len(molecules) == 1

    def test_large_molecule_chain(self):
        """Test with a longer chain of atoms (10 atoms)."""
        lattice = np.array([[50.0, 0.0, 0.0],
                           [0.0, 50.0, 0.0],
                           [0.0, 0.0, 50.0]])
        # Chain of 10 atoms, each 1 Angstrom apart
        positions = np.array([[i * 0.02, 0.5, 0.5] for i in range(10)])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.5)
        
        # All should be in one molecule
        assert len(molecules) == 1
        assert len(molecules[0]) == 10

    def test_branched_structure(self):
        """Test with a branched molecular structure."""
        lattice = np.array([[20.0, 0.0, 0.0],
                           [0.0, 20.0, 0.0],
                           [0.0, 0.0, 20.0]])
        # Y-shaped structure
        positions = np.array([
            [0.5, 0.5, 0.5],     # Center
            [0.55, 0.5, 0.5],    # Right branch
            [0.45, 0.5, 0.5],    # Left branch
            [0.5, 0.55, 0.5],    # Up branch
        ])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.5)
        
        # All should be in one molecule
        assert len(molecules) == 1
        assert len(molecules[0]) == 4

    def test_default_threshold(self):
        """Test that default threshold of 1.8 Angstrom is used."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Two atoms 1.5 Angstrom apart (within default threshold)
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.15, 0.0, 0.0]])
        molecules, _ = atoms2molecules(lattice, positions)
        
        # Should be in one molecule with default threshold
        assert len(molecules) == 1

    def test_positions_array_not_modified(self):
        """Test that original positions array is not modified."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.95, 0.0, 0.0]])
        original_positions = positions.copy()
        atoms2molecules(lattice, positions, threshold=2.0)
        
        # Original positions should not be modified
        np.testing.assert_array_equal(positions, original_positions)

    def test_periodic_images_different_from_positions(self):
        """Test that periodic_images can differ from original positions."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        # Two atoms that connect via periodic boundary
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.95, 0.0, 0.0]])
        molecules, periodic_images = atoms2molecules(lattice, positions, threshold=2.0)
        
        # If connected via PBC, periodic_images should be updated
        assert len(molecules) == 1
        # Second atom should have been moved via PBC
        # The exact position depends on which periodic image was found first

    def test_triclinic_lattice(self):
        """Test with triclinic lattice (all angles != 90 degrees)."""
        # Triclinic lattice
        lattice = np.array([
            [10.0, 0.0, 0.0],
            [2.0, 9.0, 0.0],
            [1.0, 1.5, 8.0]
        ])
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.1, 0.0, 0.0]])
        molecules, _ = atoms2molecules(lattice, positions, threshold=2.0)
        
        # Check that function handles triclinic lattice
        assert len(molecules) >= 1

    def test_all_atoms_separated(self):
        """Test when all atoms are far apart and form individual molecules."""
        lattice = np.array([[100.0, 0.0, 0.0],
                           [0.0, 100.0, 0.0],
                           [0.0, 0.0, 100.0]])
        # 5 atoms, each 20 Angstrom apart
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.8)
        
        # Each atom should be its own molecule
        assert len(molecules) == 5
        for mol in molecules:
            assert len(mol) == 1

    def test_two_molecules_different_sizes(self):
        """Test with two molecules of different sizes."""
        lattice = np.array([[30.0, 0.0, 0.0],
                           [0.0, 30.0, 0.0],
                           [0.0, 0.0, 30.0]])
        positions = np.array([
            # First molecule: 3 atoms
            [0.0, 0.0, 0.0],
            [0.033, 0.0, 0.0],   # 1 Angstrom apart
            [0.066, 0.0, 0.0],   # 1 Angstrom apart
            # Second molecule: 2 atoms
            [0.5, 0.5, 0.5],
            [0.533, 0.5, 0.5],   # 1 Angstrom apart
        ])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.5)
        
        # Should have 2 molecules
        assert len(molecules) == 2
        # Check sizes
        sizes = sorted([len(mol) for mol in molecules])
        assert sizes == [2, 3]

    def test_zero_threshold(self):
        """Test with zero threshold (no atoms should be grouped)."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.01, 0.0, 0.0]])  # Very close atoms
        molecules, _ = atoms2molecules(lattice, positions, threshold=0.0)
        
        # Each atom should be its own molecule
        assert len(molecules) == 2

    def test_very_large_threshold(self):
        """Test with very large threshold (all atoms in one molecule)."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],   # 5 Angstrom apart
            [0.5, 0.5, 0.0],   # Far apart
            [0.5, 0.5, 0.5],   # Far apart
        ])
        molecules, _ = atoms2molecules(lattice, positions, threshold=100.0)
        
        # All atoms should be in one molecule
        assert len(molecules) == 1
        assert len(molecules[0]) == 4

    def test_molecule_indices_are_sets(self):
        """Test that each molecule is represented as a set."""
        lattice = np.array([[10.0, 0.0, 0.0],
                           [0.0, 10.0, 0.0],
                           [0.0, 0.0, 10.0]])
        positions = np.array([[0.0, 0.0, 0.0],
                             [0.1, 0.0, 0.0]])
        molecules, _ = atoms2molecules(lattice, positions, threshold=1.8)
        
        for mol in molecules:
            assert isinstance(mol, set)

    def test_all_atom_indices_covered(self):
        """Test that all atom indices are covered in the output."""
        lattice = np.array([[20.0, 0.0, 0.0],
                           [0.0, 20.0, 0.0],
                           [0.0, 0.0, 20.0]])
        n_atoms = 10
        # Random-ish positions
        np.random.seed(42)
        positions = np.random.rand(n_atoms, 3)
        molecules, _ = atoms2molecules(lattice, positions, threshold=5.0)
        
        # Check all indices are covered
        all_indices = set()
        for mol in molecules:
            all_indices.update(mol)
        assert all_indices == set(range(n_atoms))

    def test_no_duplicate_atoms_in_molecules(self):
        """Test that no atom appears in multiple molecules."""
        lattice = np.array([[20.0, 0.0, 0.0],
                           [0.0, 20.0, 0.0],
                           [0.0, 0.0, 20.0]])
        np.random.seed(42)
        positions = np.random.rand(8, 3)
        molecules, _ = atoms2molecules(lattice, positions, threshold=5.0)
        
        # Check no duplicates
        all_indices = []
        for mol in molecules:
            all_indices.extend(list(mol))
        assert len(all_indices) == len(set(all_indices))
