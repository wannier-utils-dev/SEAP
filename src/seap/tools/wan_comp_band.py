import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Try to import from external cif2qewan package
try:
    # First, try importing from environment variable path
    cif2qewan_dir = os.environ.get("CIF2QEWAN_DIR")
    if cif2qewan_dir and os.path.exists(cif2qewan_dir):
        sys.path.insert(0, cif2qewan_dir)
        from band_comp import *
    else:
        # Try importing as installed package
        import band_comp
        from band_comp import *
except ImportError:
    # Fallback for when cif2qewan is not available
    print("Warning: cif2qewan.band_comp module not available")
    print("Please install cif2qewan from https://github.com/wannier-utils-dev/cif2qewan")
    print("or set CIF2QEWAN_DIR environment variable to point to the cif2qewan directory")
    # Define dummy functions to prevent errors
    def get_band_data(filename):
        return [], []
    def get_ef_from_scfout():
        return 0.0
    def get_klabel():
        return [], []

def main():
    """
    Main function to compare the band structure of a custom calculation with the SCDM method.
    """
    # Define file paths for the band data and output files
    scdm_band = 'scdm_band.dat'  # File containing SCDM band data
    my_band = 'my_band.dat'  # File containing custom calculation band data
    scfout = 'scf.out'  # File containing self-consistent field output
    pwscf_win = 'pwscf.win'  # File for Wannier90 input
    wannier_band_gnu = 'pwscf_band.gnu'  # File for GNU plot of Wannier bands

    # Retrieve band data from the specified files
    x_scdm, y_scdm = get_band_data(scdm_band)  # SCDM band data
    x, y = get_band_data(my_band)  # Custom calculation band data

    # Get the Fermi energy from the SCF output file
    ef = get_ef_from_scfout()

    # Get k-point labels for the x-axis of the plot
    klabel = get_klabel()

    # Set the font size for the plot
    plt.rcParams["font.size"] = 16

    # Set the title of the plot
    plt.title("Red=scdm, Black=mycalc")

    # Plot the SCDM band data in red
    plt.plot(x_scdm, y_scdm - ef, c="r", lw=1, label="scdm")

    # Plot the custom calculation band data in black
    plt.plot(x, y - ef, c="k", lw=0.5, label="mycalc")

    # Label the y-axis with the energy difference from the Fermi level
    plt.ylabel(r"$E - E_{\mathrm{F}}$[eV]")

    # Set the x-ticks to the k-point labels
    plt.xticks(klabel[0], klabel[1])

    # Set the x-axis limits from 0 to 1
    plt.xlim([0, 1])

    # Calculate the minimum and maximum y-values for the plot
    y_min = np.min(y - ef)
    y_max = np.max(y - ef)

    # Calculate the plot y-limits with a 5% margin
    factor = 1.0  # Default factor value
    py_min = y_min - 0.05 * (y_max - y_min) * factor
    py_max = y_max + 0.05 * (y_max - y_min) * factor
    plt.ylim([py_min, py_max])

    # Draw vertical lines at the k-point positions
    plt.vlines(klabel[0], py_min, py_max, colors='black', linewidth=1.0, zorder=3)

    # Save the plot as a PNG file
    plt.savefig("./wan_band_compare.png", bbox_inches='tight')

    # Save the plot as an EPS file
    plt.savefig("./wan_band_compare.eps", bbox_inches='tight')


if __name__ == '__main__':
    main()
