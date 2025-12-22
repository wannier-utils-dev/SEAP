# SEAP(Semiautomated Estimator for Atomic Projections of Wannier functions)

<!-- 
<div align="center">
  <img src="images/logo.png" alt="SEAP Logo" width="300">
</div>
-->

SEAP (Semiautomated Estimator for Atomic Projections of Wannier functions) is a Python toolkit for generating hydrogen-like initial states for electronic structure calculations using a semi-automated projection method. It streamlines the preparation of input files for Wannier90 and Quantum ESPRESSO, and provides utilities for band selection, orbital projection, and workflow automation for both molecular and solid-state systems.

**Key features:**
- Automated and manual band selection for initial state projection
- Generation of input files for Wannier90 and Quantum ESPRESSO
- Neural network and regression-based orbital analysis
- Workflow scripts for high-throughput calculations
- Support for both molecular and crystalline systems

## Project Structure

```
SEAP/
├── src/                    # Main source code
│   └── seap/              # Core package
│       ├── __init__.py
│       ├── core/          # Core functionality
│       │   ├── __init__.py
│       │   ├── select_bands.py
│       │   ├── postppx.py
│       │   ├── predict.py
│       │   ├── proj.py
│       │   └── mod_win.py
│       ├── common/        # Common utilities
│       │   ├── __init__.py
│       │   ├── lattice.py
│       │   ├── molecule.py
│       │   ├── units.py
│       │   └── xsf.py
│       ├── interface/     # External interfaces
│       │   ├── __init__.py
│       │   └── qe.py
│       ├── prediction/    # Prediction algorithms
│       │   ├── __init__.py
│       │   ├── datasets.py
│       │   ├── grids.py
│       │   ├── integration.py
│       │   ├── lasso.py
│       │   ├── sph_harm.py
│       │   ├── utils.py
│       │   └── wavefunction.py
│       └── tools/         # Utility tools
│           ├── __init__.py
│           ├── compare.py
│           └── wan_comp_band.py
├── config/                # Configuration files
│   ├── cif2qewan.toml
│   ├── pp_psl_rrkj.csv
│   ├── nc-sr-05_pbe_standard_upf.csv
│   └── nc-sr-05_pbe_stringent_upf.csv
├── models/                # Pre-trained models
│   └── encoder_1/
│       ├── encoder.pth
│       └── encoder.py
├── scripts/               # Executable scripts
│   └── cif2qewan.sh
├── examples/              # Usage examples
│   └── neuralnet/
│       ├── entangled_bands/
│       └── isolated_bands/
├── tests/                 # Test files
├── docs/                  # Documentation
├── images/                # Images and logos
├── pyproject.toml         # SEAP project configuration
├── README.md
└── LICENSE
```

## Overview

This toolset consists of five main scripts that work together to generate Wannier90 input files:

1. `select_bands.py` - Selects bands for Wannier function generation
2. `postppx.py` - Processes output from `pp.x` to extract wave function data
3. `predict.py` - Determines orbital projections from the processed data
4. `proj.py` - Generates Wannier90 projection format
5. `mod_win.py` - Modifies Wannier90 input files

## CIF2QEWAN Tool (External Dependency)

This package uses the [cif2qewan](https://github.com/wannier-utils-dev/cif2qewan) tool as an external dependency. The cif2qewan tool is a GPL-licensed independent package that creates Quantum ESPRESSO and Wannier90 input files from CIF (Crystallographic Information File) files.

### Installing CIF2QEWAN

Since cif2qewan is licensed under GPL-2.0 and is maintained as an independent package, you need to download and install it separately:

```bash
# Clone the cif2qewan repository
git clone https://github.com/wannier-utils-dev/cif2qewan.git
cd cif2qewan

# Install cif2qewan
pip install -e .
```

After installation, you can use cif2qewan by specifying its path in your scripts or by adding it to your Python path.

### CIF2QEWAN Features

- **CIF to Quantum ESPRESSO conversion**: Generate input files for SCF, NSCF, and band structure calculations
- **Wannier90 integration**: Create Wannier90 input files for localized orbital analysis
- **Band structure comparison**: Compare DFT and Wannier90 band structures
- **Energy convergence checking**: Validate Wannier90 Hamiltonian accuracy

### References

The cif2qewan tool has been used in several research publications:

- **Iron-based binary ferromagnets for transverse thermoelectric conversion**, A. Sakai, S. Minami, T. Koretsune et al. Nature 581 53-57 (2020)
- **Systematic first-principles study of the on-site spin-orbit coupling in crystals**, Phys. Rev. B 102 045109 (2020)

## Installation

### SEAP Package

```bash
# Install SEAP package
pip install -e .
```

### CIF2QEWAN Tool (External Dependency)

**Important**: cif2qewan is now an external dependency and must be downloaded separately. If you have an older version of this repository that includes `src/cif2qewan/`, you should remove it:

```bash
# Remove the old cif2qewan directory (if present)
rm -rf src/cif2qewan
rm -f pyproject-cif2qewan.toml
```

Then download and install cif2qewan separately:

```bash
# Download and install cif2qewan separately
git clone https://github.com/wannier-utils-dev/cif2qewan.git
cd cif2qewan
pip install -e .
```

See the [CIF2QEWAN section](#cif2qewan-tool-external-dependency) above for more details.

## Usage

### Complete Workflow Example

The following example demonstrates a complete workflow from CIF file to Wannier function analysis:

#### 1. Initial Setup

```bash
# Set environment variables
export PY_DIR="/path/to/seap"
export QE_DIR="/path/to/QuantumESPRESSO"  # version 7.3 or later
export WANNIER90_DIR="/path/to/wannier90"
export BAND_DATA="band.dat"

# Create working directory
mkdir my_calculation
cd my_calculation
```

#### 2. CIF to Quantum ESPRESSO Conversion

```bash
# Convert CIF file to Quantum ESPRESSO input files
# Note: Set CIF2QEWAN_DIR to the path where you installed cif2qewan
export CIF2QEWAN_DIR="/path/to/cif2qewan"
python ${CIF2QEWAN_DIR}/cif2qewan.py *.cif cif2qewan.toml

# Run SCF calculation
mpirun -n 32 ${QE_DIR}/bin/pw.x -nk 8 < scf.in > scf.out

# Copy work directories for later use
cp -r work check_wannier/
cp -r work band/
```

#### 3. Wannier Function Analysis

```bash
# Create prediction directory
mkdir pred
cd pred

# Select bands for Wannier function generation
python ${PY_DIR}/src/seap/core/select_bands.py ../ --man $BAND_DATA --bl 1.2

# Run pp.x for each selected band
tail -n +2 $BAND_DATA | while read -r nb; do
    mpirun -n 1 ${QE_DIR}/bin/pp.x < pwscf.pp_${nb}.in > pwscf.pp_${nb}.out
done

# Process pp.x output and predict orbitals
python ${PY_DIR}/src/seap/core/postppx.py --bl 1.2
python ${PY_DIR}/src/seap/core/predict.py --mode nn --nnid 1
python ${PY_DIR}/src/seap/core/proj.py

cd ../
```

#### 4. Wannier90 Setup and Calculation

```bash
# Get number of bands and modify input files
nbnd=$(cat ./pred/nbnd | awk '{print $1}')
sed -i "s/nbnd .*/nbnd = ${nbnd}/g" nscf.in
python ${PY_DIR}/src/seap/core/mod_win.py --dis

# Run NSCF calculation
mpirun -n 32 ${QE_DIR}/bin/pw.x -nk 8 < nscf.in > nscf.out

# Wannier90 preprocessing
mpirun -n 1 ${WANNIER90_DIR}/wannier90.x -pp pwscf

# Convert to Wannier90 format
mpirun -n 1 ${QE_DIR}/bin/pw2wannier90.x < pw2wan.in > pw2wan.out

rm -r work

# Set frozen window and run Wannier90
echo 'set dis_froz_max = EF+4.5eV'
ef=$(grep Fermi nscf.out | sed -e "s/[^0-9.]//g")
ef_new=$(bc -l <<< "$ef + 4.5")
sed -i "s/dis_froz_max .*/dis_froz_max = $ef_new/g" pwscf.win
mpirun -n 1 ${WANNIER90_DIR}/wannier90.x pwscf
```

#### 5. Validation and Band Structure Comparison

```bash
# Validate Wannier functions
cd check_wannier
sed -i "s/nbnd .*/nbnd = ${nbnd}/g" nscf.in
mpirun -n 32 ${QE_DIR}/bin/pw.x -nk 8 < nscf.in > nscf.out
rm -r work
cd ../
python ${CIF2QEWAN_DIR}/wannier_conv.py

# Compare band structures
cd band
nbnd_new=$(bc -l <<< "$nbnd + 10")
sed -i "s/nbnd .*/nbnd = ${nbnd_new}/g" nscf.in
mpirun -n 32 ${QE_DIR}/bin/pw.x -nk 8 < nscf.in > nscf.out
mpirun -n 1 ${QE_DIR}/bin/bands.x < band.in > band.out
rm -r work
cd ../
python ${CIF2QEWAN_DIR}/band_comp.py
```

### Individual Tool Usage

#### SEAP Core Tools

```bash
# Select bands for Wannier function generation
python src/seap/core/select_bands.py /path/to/work/dir --bl 1.8

# Extract wave function data from pp.x output
python src/seap/core/postppx.py --bl 1.8 --cl 4.0

# Estimate orbital projections using neural network
python src/seap/core/predict.py --mode nn --nnid 1

# Generate Wannier90 projection format
python src/seap/core/proj.py

# Modify Wannier90 input files
python src/seap/core/mod_win.py --dis --maxloc --wplot
```

#### CIF2QEWAN Tool

```bash
# Set the path to your cif2qewan installation
export CIF2QEWAN_DIR="/path/to/cif2qewan"

# Convert CIF file to Quantum ESPRESSO input
python ${CIF2QEWAN_DIR}/cif2qewan.py input.cif cif2qewan.toml

# Compare band structures
python ${CIF2QEWAN_DIR}/band_comp.py

# Convert Wannier functions
python ${CIF2QEWAN_DIR}/wannier_conv.py
```

### Command Line Interface (After Installation)

```bash
# SEAP tools
seap-select-bands /path/to/work/dir --bl 1.8
seap-postppx --bl 1.8 --cl 4.0
seap-predict --mode nn --nnid 1
seap-proj
seap-mod-win --dis --maxloc --wplot

# CIF2QEWAN tool (if installed via pip)
cif2qewan input.cif cif2qewan.toml
# Or use directly:
python /path/to/cif2qewan/cif2qewan.py input.cif cif2qewan.toml
```

## Script Details

### select_bands.py

Selects bands for Wannier function generation and creates input files for `pp.x`.

**Features:**
- Detects molecules based on interatomic distances
- Automatically selects bands closest to the Fermi level
- Allows manual band selection
- Generates `pp.x` input files automatically

**Options:**
- `--man <filename>`: Specify a file containing manually selected band numbers
- `--bl <value>`: Maximum interatomic distance (in angstroms) considered as a molecule
- `--il <value>`: Length for getting interstitial sites. Default is None (not generating interstitial sites).

**Output Files:**
- `band.dat`: Indices of selected bands
- `nbnd`: Total number of bands
- `mol.info`: Molecular information
- `pwscf.pp_*.in`: Input files for `pp.x` for each band

### postppx.py

Extracts information from `pp.x` output (XSF format) and generates data for orbital estimation.

**Features:**
- Reads wave function data from XSF files
- Detects molecular clusters and calculates their centers
- Generates cubic data for orbital estimation

**Options:**
- `--bl <value>`: Maximum interatomic distance (in angstroms) considered as a molecule
- `--cl <value>`: Length of one side of the cubic region (in angstroms) for orbital estimation
- `--il <value>`: Length for getting interstitial sites. Default is None (not generating interstitial sites). 

**Output Files:**
- `image32x32x32.npy`: Wave function data for orbital estimation
- `image_info.npy`: Information about the wave function data
- `center.csv`: Center positions of each molecule
- `calc.dat`: Calculation information

### predict.py

Determines orbital projections from the output data of `postppx.py`.

**Features:**
- Estimates coefficients of real spherical harmonics
- Supports multiple estimation methods

**Options:**
- `--mode <method>`: Specify estimation method (`nn`, `lr`, `itg`)
  - `nn`: Neural network
  - `lr`: Lasso regression
  - `itg`: Integration method
- `--nnid <value>`: Neural network model ID (currently only `1` is available)
- `--lrad <value>`: Radius of the sampling sphere (`0 < x < 0.5`)
- `--optr`: Flag to optimize the radius

**Output Files:**
- `orbital.csv`: Estimated orbital information
- `nn.out`: Neural network output (when using `nn` mode)

### proj.py

Outputs projections in Wannier90 format.

**Features:**
- Generates projection information from `center.csv` and `orbital.csv`
- Creates projection strings in Wannier90 format

**Output Files:**
- `proj.out`: Projection information in Wannier90 format

### mod_win.py

Modifies parts of `pw2wan.in` and `pwscf.win` files.

**Features:**
- Determines the number of Wannier functions and bands from `band.dat` and `nbnd` files
- Modifies `pwscf.win` file to add projection information
- Provides options for handling entangled bands

**Options:**
- `--dis`: Enable option to Wannierize entangled bands
- `--maxloc`: Enable maximum localization
- `--wplot`: Enable plotting of generated Wannier functions

## Notes

- Optimized for band calculations of molecular systems
- Automatic band selection for inorganic crystals is not yet implemented
- Only one neural network model is currently available
- CIF2QEWAN must be downloaded separately from [wannier-utils-dev/cif2qewan](https://github.com/wannier-utils-dev/cif2qewan)
- Requires Quantum ESPRESSO (version 7.3 or later) and Wannier90
- To get information of interstitial sites, [doped](https://github.com/SMTG-Bham/doped) package is used.

## License

This project is licensed under the MIT License - see LICENSE file for details.

**Note**: The cif2qewan tool is a separate GPL-2.0 licensed package that must be downloaded independently from https://github.com/wannier-utils-dev/cif2qewan. It is not included in this package distribution.

