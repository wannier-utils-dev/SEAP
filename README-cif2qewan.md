# CIF2QEWAN (External Dependency)

<div align="center">
  <img src="images/logo.png" alt="CIF2QEWAN Logo" width="250">
</div>

**Note**: CIF2QEWAN is a separate GPL-2.0 licensed package that must be downloaded independently. It is not included in this package distribution.

## Overview

CIF2QEWAN is a specialized tool that automates the conversion process from CIF files to input files for quantum chemistry calculations using Quantum ESPRESSO and Wannier90. This tool is developed and maintained by the [wannier-utils-dev](https://github.com/wannier-utils-dev/cif2qewan) team.

## Installation

Since cif2qewan is licensed under GPL-2.0 and is maintained as an independent package, you need to download and install it separately:

```bash
# Clone the cif2qewan repository
git clone https://github.com/wannier-utils-dev/cif2qewan.git
cd cif2qewan

# Install cif2qewan
pip install -e .
```

Alternatively, you can use cif2qewan directly without installation by setting the path to the repository:

```bash
# Clone the repository
git clone https://github.com/wannier-utils-dev/cif2qewan.git

# Set environment variable
export CIF2QEWAN_DIR="/path/to/cif2qewan"
```

## Features

- **CIF to Quantum ESPRESSO conversion**: Generate input files for SCF, NSCF, and band structure calculations
- **Wannier90 integration**: Create Wannier90 input files for localized orbital analysis
- **Automatic pseudopotential selection**: Choose appropriate pseudopotentials based on element and functional
- **Band structure analysis**: Compare calculated and Wannier-interpolated band structures
- **Energy convergence checking**: Validate Wannier90 Hamiltonian accuracy
- **Flexible configuration**: Customizable settings via TOML configuration files

## References

The cif2qewan tool has been used in several research publications:

- **Iron-based binary ferromagnets for transverse thermoelectric conversion**, A. Sakai, S. Minami, T. Koretsune et al. Nature 581 53-57 (2020)
- **Systematic first-principles study of the on-site spin-orbit coupling in crystals**, Phys. Rev. B 102 045109 (2020)

## Usage

### Basic Usage

```bash
# Set the path to your cif2qewan installation
export CIF2QEWAN_DIR="/path/to/cif2qewan"

# Convert CIF file to Quantum ESPRESSO input
python ${CIF2QEWAN_DIR}/cif2qewan.py input.cif cif2qewan.toml

# Or if installed via pip:
cif2qewan input.cif cif2qewan.toml
```

### Complete Workflow

1. **Convert CIF to Quantum ESPRESSO inputs**:
   ```bash
   export CIF2QEWAN_DIR="/path/to/cif2qewan"
   python ${CIF2QEWAN_DIR}/cif2qewan.py *.cif cif2qewan.toml
   ```

2. **Run SCF calculation**:
   ```bash
   pw.x < scf.in > scf.out
   ```

3. **Run NSCF calculation**:
   ```bash
   pw.x < nscf.in > nscf.out
   ```

4. **Wannier90 preprocessing**:
   ```bash
   wannier90.x -pp pwscf
   ```

5. **Convert to Wannier90 format**:
   ```bash
   pw2wannier90.x < pw2wan.in
   ```

6. **Edit dis_froz_max in pwscf.win** (recommended: EF+1eV ~ EF+3eV)

7. **Wannierize**:
   ```bash
   wannier90.x pwscf
   ```

### Band Structure Comparison

```bash
export CIF2QEWAN_DIR="/path/to/cif2qewan"
cd band
pw.x < ../scf.in > scf.out
pw.x < nscf.in > nscf.out
bands.x < band.in > band.out
cd ..
python ${CIF2QEWAN_DIR}/band_comp.py
```

### Energy Convergence Check

```bash
export CIF2QEWAN_DIR="/path/to/cif2qewan"
cd check_wannier
pw.x < ../scf.in > scf.out
pw.x < nscf.in > nscf.out
cd ..
python ${CIF2QEWAN_DIR}/wannier_conv.py
cat check_wannier/CONV
```

## Configuration

Create a configuration file (`cif2qewan.toml`):

```toml
[calculation]
functional = "pbe"
pseudopotential_type = "standard"
kpoints_density = 0.1

[output]
prefix = "my_calculation"
outdir = "./output"
```

## Integration with SEAP

When using cif2qewan with the SEAP package, set the `CIF2QEWAN_DIR` environment variable to point to your cif2qewan installation:

```bash
export CIF2QEWAN_DIR="/path/to/cif2qewan"
export PY_DIR="/path/to/seap"
```

Then use cif2qewan scripts in your workflow:

```bash
python ${CIF2QEWAN_DIR}/cif2qewan.py *.cif cif2qewan.toml
python ${CIF2QEWAN_DIR}/band_comp.py
python ${CIF2QEWAN_DIR}/wannier_conv.py
```

## Dependencies

- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- ase >= 3.22.0
- pymatgen >= 2022.0.0
- cif2cell (for CIF file processing)

## License

The cif2qewan tool is licensed under the GNU General Public License v2.0 (GPL-2.0). Please refer to the [original repository](https://github.com/wannier-utils-dev/cif2qewan) for the full license text.

## Contributing

Contributions to cif2qewan should be made to the [original repository](https://github.com/wannier-utils-dev/cif2qewan).

## Citation

If you use CIF2QEWAN in your research, please cite:

```bibtex
@software{cif2qewan2024,
  title={cif2qewan: CIF to Quantum ESPRESSO and Wannier90 input generator},
  author={wannier-utils-dev team},
  year={2024},
  url={https://github.com/wannier-utils-dev/cif2qewan}
}
```

## Links

- **Repository**: https://github.com/wannier-utils-dev/cif2qewan
- **Original Project**: Maintained by wannier-utils-dev team
