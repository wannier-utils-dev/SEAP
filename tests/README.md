# SEAP Package Import Test

This document describes how to use the import test script (`tests/test_imports.py`) for the SEAP package.

## Overview

`test_imports.py` is a test script that comprehensively checks module imports across the entire SEAP package. It ensures that all imports work correctly when package structure changes or new modules are added.

## Usage

### Basic Usage

```bash
# Test all modules
python tests/test_imports.py

# Test with detailed output
python tests/test_imports.py --verbose

# Test only specific modules
python tests/test_imports.py --module core
python tests/test_imports.py --module common
python tests/test_imports.py --module prediction
python tests/test_imports.py --module interface
python tests/test_imports.py --module tools
```

### Command Line Options

- `--verbose`, `-v`: Show detailed output for each import test
- `--module`, `-m`: Test only a specific module (options: common, interface, core, prediction, tools)

### Examples

```bash
# Test all modules with detailed output
python tests/test_imports.py --verbose

# Test only the core module
python tests/test_imports.py --module core --verbose

# Basic test (minimal output)
python tests/test_imports.py
```

## Test Content

### Package Import Tests
- `seap` - Main package
- `seap.common` - Common utilities package
- `seap.core` - Core functionality package
- `seap.interface` - Interface package
- `seap.prediction` - Prediction package
- `seap.tools` - Tools package

### Module-Specific Tests

#### common Module
- `seap.common.lattice` - Lattice utilities
- `seap.common.molecule` - Molecule utilities
- `seap.common.units` - Unit conversion utilities
- `seap.common.xsf` - XSF file utilities

#### interface Module
- `seap.interface.qe` - Quantum ESPRESSO interface

#### core Module
- `seap.core.select_bands` - Band selection utilities
- `seap.core.postppx` - Post-processing utilities
- `seap.core.predict` - Prediction utilities
- `seap.core.proj` - Projection utilities
- `seap.core.mod_win` - Wannier90 input modification

#### prediction Module
- `seap.prediction.datasets` - Dataset utilities
- `seap.prediction.grids` - Grid utilities
- `seap.prediction.integration` - Integration utilities
- `seap.prediction.lasso` - Lasso regression utilities
- `seap.prediction.sph_harm` - Spherical harmonics utilities
- `seap.prediction.utils` - Prediction utilities
- `seap.prediction.wavefunction` - Wavefunction utilities

#### tools Module
- `seap.tools.compare` - Comparison utilities
- `seap.tools.wan_comp_band` - Wannier band comparison utilities

## Output Examples

### Success Case
```
🚀 Starting SEAP package import tests...
============================================================

📦 Testing package imports...
✅ ✓ Main SEAP package
✅ ✓ Common utilities package
✅ ✓ Core functionality package
✅ ✓ Interface package
✅ ✓ Prediction package
✅ ✓ Tools package

...

============================================================
📊 TEST SUMMARY
============================================================
Total tests: 25
Passed: 25
Failed: 0
🎉 All tests passed! SEAP package imports are working correctly.
============================================================
```

### Failure Case
```
============================================================
📊 TEST SUMMARY
============================================================
Total tests: 25
Passed: 22
Failed: 3
⚠️ 3 test(s) failed. Please check the import issues above.
============================================================
```

## Troubleshooting

### Common Issues

1. **Relative Import Error**: Relative imports between modules are not correctly configured
2. **Missing Dependencies**: External packages (numpy, pandas, scipy, etc.) are not installed
3. **File Reading Error**: File reading operations are executed during module import

### Solutions

1. **Fix Relative Imports**: Change `from module import` to `from ..module import`
2. **Install Dependencies**: Install required packages
3. **Restructure Code**: Move code that executes during module import into functions

## Continuous Integration

This test script can be integrated into CI/CD pipelines to automatically run import tests when code changes.

```yaml
# GitHub Actions example
- name: Test SEAP imports
  run: python test_imports.py
```

## Maintenance

When adding a new module, add it to the corresponding test function in `test_imports.py`.

```python
def test_core_modules(self):
    """Test all modules in the core package."""
    core_modules = [
        # Existing modules...
        ("seap.core.new_module", "New module description"),
    ]
    # ...
```
