#!/usr/bin/env python3
"""
Comprehensive import test for SEAP package.

This script tests all modules and their imports to ensure the package
structure is correct and all dependencies can be resolved.

Usage:
    python test_imports.py [--verbose] [--module MODULE_NAME]
    
Options:
    --verbose    Show detailed output for each import test
    --module     Test only a specific module (e.g., 'core', 'common', 'prediction')
"""

import sys
import os
import argparse
from typing import Dict, List, Tuple, Any

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class ImportTester:
    """Test class for checking SEAP package imports."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the import tester.
        
        Parameters
        ----------
        verbose : bool
            Whether to show detailed output.
        """
        self.verbose = verbose
        self.results: Dict[str, Dict[str, Any]] = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def log(self, message: str, level: str = "INFO") -> None:
        """
        Log a message if verbose mode is enabled.
        
        Parameters
        ----------
        message : str
            Message to log.
        level : str
            Log level (INFO, SUCCESS, ERROR).
        """
        if self.verbose:
            prefix = {
                "INFO": "ℹ️ ",
                "SUCCESS": "✅",
                "ERROR": "❌"
            }.get(level, "ℹ️ ")
            print(f"{prefix} {message}")
    
    def test_import(self, module_path: str, description: str = None) -> bool:
        """
        Test importing a module.
        
        Parameters
        ----------
        module_path : str
            Module path to import (e.g., 'seap.core.select_bands').
        description : str, optional
            Description of what is being tested.
            
        Returns
        -------
        bool
            True if import succeeded, False otherwise.
        """
        self.total_tests += 1
        test_name = description or module_path
        
        try:
            self.log(f"Testing import: {module_path}")
            __import__(module_path)
            self.passed_tests += 1
            self.log(f"✓ {test_name}", "SUCCESS")
            return True
        except ImportError as e:
            error_msg = str(e)
            # Skip tests for optional dependencies
            if 'doped' in error_msg:
                self.passed_tests += 1
                self.total_tests -= 1  # Don't count skipped tests
                self.log(f"⊘ {test_name}: skipped (optional dependency 'doped' not installed)", "INFO")
                return True
            self.failed_tests += 1
            self.log(f"✗ {test_name}: {e}", "ERROR")
            return False
        except Exception as e:
            self.failed_tests += 1
            self.log(f"✗ {test_name}: Unexpected error - {e}", "ERROR")
            return False
    
    def test_module_components(self, module_path: str, components: List[str]) -> None:
        """
        Test importing specific components from a module.
        
        Parameters
        ----------
        module_path : str
            Module path to import from.
        components : list of str
            List of component names to import.
        """
        try:
            module = __import__(module_path, fromlist=components)
            for component in components:
                if hasattr(module, component):
                    self.log(f"✓ {module_path}.{component} is available", "SUCCESS")
                else:
                    self.log(f"✗ {module_path}.{component} not found", "ERROR")
        except ImportError as e:
            self.log(f"✗ Failed to import {module_path}: {e}", "ERROR")
    
    def test_common_modules(self) -> None:
        """Test all modules in the common package."""
        print("\n🔧 Testing common modules...")
        
        common_modules = [
            ("seap.common.lattice", "Lattice utilities"),
            ("seap.common.molecule", "Molecule utilities"),
            ("seap.common.units", "Unit conversion utilities"),
            ("seap.common.xsf", "XSF file utilities"),
        ]
        
        for module_path, description in common_modules:
            self.test_import(module_path, description)
        
        # Test specific components
        self.test_module_components("seap.common.lattice", ["crys2cart", "cart2crys"])
        self.test_module_components("seap.common.molecule", ["atoms2molecules"])
        self.test_module_components("seap.common.xsf", ["get_params_from_xsf", "output_xsf"])
    
    def test_interface_modules(self) -> None:
        """Test all modules in the interface package."""
        print("\n🔌 Testing interface modules...")
        
        interface_modules = [
            ("seap.interface.qe", "Quantum ESPRESSO interface"),
        ]
        
        for module_path, description in interface_modules:
            self.test_import(module_path, description)
        
        # Test specific components
        self.test_module_components("seap.interface.qe", ["PWIn", "PPIn"])
    
    def test_core_modules(self) -> None:
        """Test all modules in the core package."""
        print("\n⚙️ Testing core modules...")
        
        core_modules = [
            ("seap.core.select_bands", "Band selection utilities"),
            ("seap.core.postppx", "Post-processing utilities"),
            ("seap.core.predict", "Prediction utilities"),
            ("seap.core.proj", "Projection utilities"),
            ("seap.core.mod_win", "Wannier90 input modification"),
        ]
        
        for module_path, description in core_modules:
            self.test_import(module_path, description)
        
        # Test specific components
        self.test_module_components("seap.core.select_bands", [
            "main", "count_cmol", "output_mol_info", "get_data_from_pwout",
            "find_nearest_bands", "get_initial_ppin", "write_band_dat"
        ])
        self.test_module_components("seap.core.postppx", [
            "PostPPX", "output_info", "output_csv"
        ])
        self.test_module_components("seap.core.predict", [
            "main", "deep_learning_model", "sparse_modeling", 
            "direct_calculation", "output_csv", "output_csv_new"
        ])
        self.test_module_components("seap.core.proj", [
            "main", "generate_projection_string", "write_projection_to_file"
        ])
        self.test_module_components("seap.core.mod_win", ["main"])
    
    def test_prediction_modules(self) -> None:
        """Test all modules in the prediction package."""
        print("\n🔮 Testing prediction modules...")
        
        prediction_modules = [
            ("seap.prediction.datasets", "Dataset utilities"),
            ("seap.prediction.grids", "Grid utilities"),
            ("seap.prediction.integration", "Integration utilities"),
            ("seap.prediction.lasso", "Lasso regression utilities"),
            ("seap.prediction.sph_harm", "Spherical harmonics utilities"),
            ("seap.prediction.utils", "Prediction utilities"),
            ("seap.prediction.wavefunction", "Wavefunction utilities"),
        ]
        
        for module_path, description in prediction_modules:
            self.test_import(module_path, description)
        
        # Test specific components
        self.test_module_components("seap.prediction.datasets", ["BoxData"])
        self.test_module_components("seap.prediction.lasso", ["LassoRegression"])
        self.test_module_components("seap.prediction.integration", ["Integration"])
        self.test_module_components("seap.prediction.sph_harm", ["quantum_number", "orb_symbol"])
    
    def test_tools_modules(self) -> None:
        """Test all modules in the tools package."""
        print("\n🛠️ Testing tools modules...")
        
        tools_modules = [
            ("seap.tools.compare", "Comparison utilities"),
            ("seap.tools.wan_comp_band", "Wannier band comparison utilities"),
        ]
        
        for module_path, description in tools_modules:
            self.test_import(module_path, description)
    
    def test_package_imports(self) -> None:
        """Test importing the main package and subpackages."""
        print("\n📦 Testing package imports...")
        
        package_imports = [
            ("seap", "Main SEAP package"),
            ("seap.common", "Common utilities package"),
            ("seap.core", "Core functionality package"),
            ("seap.interface", "Interface package"),
            ("seap.prediction", "Prediction package"),
            ("seap.tools", "Tools package"),
        ]
        
        for module_path, description in package_imports:
            self.test_import(module_path, description)
    
    def run_all_tests(self) -> None:
        """Run all import tests."""
        print("🚀 Starting SEAP package import tests...")
        print("=" * 60)
        
        self.test_package_imports()
        self.test_common_modules()
        self.test_interface_modules()
        self.test_core_modules()
        self.test_prediction_modules()
        self.test_tools_modules()
        
        self.print_summary()
    
    def run_module_tests(self, module_name: str) -> None:
        """
        Run tests for a specific module.
        
        Parameters
        ----------
        module_name : str
            Name of the module to test.
        """
        print(f"🎯 Testing {module_name} module...")
        print("=" * 60)
        
        if module_name == "common":
            self.test_common_modules()
        elif module_name == "interface":
            self.test_interface_modules()
        elif module_name == "core":
            self.test_core_modules()
        elif module_name == "prediction":
            self.test_prediction_modules()
        elif module_name == "tools":
            self.test_tools_modules()
        else:
            print(f"❌ Unknown module: {module_name}")
            print("Available modules: common, interface, core, prediction, tools")
            return
        
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        
        if self.failed_tests == 0:
            print("🎉 All tests passed! SEAP package imports are working correctly.")
            return_code = 0
        else:
            print(f"⚠️ {self.failed_tests} test(s) failed. Please check the import issues above.")
            return_code = 1
        
        print("=" * 60)
        sys.exit(return_code)


def main():
    """Main function to run the import tests."""
    parser = argparse.ArgumentParser(
        description="Test SEAP package imports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_imports.py                    # Test all modules
    python test_imports.py --verbose          # Test all with detailed output
    python test_imports.py --module core      # Test only core module
    python test_imports.py --module common    # Test only common module
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output for each import test"
    )
    
    parser.add_argument(
        "--module", "-m",
        choices=["common", "interface", "core", "prediction", "tools"],
        help="Test only a specific module"
    )
    
    args = parser.parse_args()
    
    tester = ImportTester(verbose=args.verbose)
    
    if args.module:
        tester.run_module_tests(args.module)
    else:
        tester.run_all_tests()


# Pytest test functions
def test_package_imports():
    """Test importing the main package and subpackages."""
    tester = ImportTester(verbose=False)
    tester.test_package_imports()
    assert tester.failed_tests == 0, f"{tester.failed_tests} package import(s) failed"


def test_common_modules():
    """Test all modules in the common package."""
    tester = ImportTester(verbose=False)
    tester.test_common_modules()
    assert tester.failed_tests == 0, f"{tester.failed_tests} common module import(s) failed"


def test_interface_modules():
    """Test all modules in the interface package."""
    tester = ImportTester(verbose=False)
    tester.test_interface_modules()
    assert tester.failed_tests == 0, f"{tester.failed_tests} interface module import(s) failed"


def test_core_modules():
    """Test all modules in the core package."""
    tester = ImportTester(verbose=False)
    tester.test_core_modules()
    assert tester.failed_tests == 0, f"{tester.failed_tests} core module import(s) failed"


def test_prediction_modules():
    """Test all modules in the prediction package."""
    tester = ImportTester(verbose=False)
    tester.test_prediction_modules()
    assert tester.failed_tests == 0, f"{tester.failed_tests} prediction module import(s) failed"


def test_tools_modules():
    """Test all modules in the tools package."""
    tester = ImportTester(verbose=False)
    tester.test_tools_modules()
    assert tester.failed_tests == 0, f"{tester.failed_tests} tools module import(s) failed"


if __name__ == "__main__":
    main()
