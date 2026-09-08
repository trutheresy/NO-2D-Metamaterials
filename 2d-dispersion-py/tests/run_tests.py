"""
Test runner script for all unit tests.

This script runs all tests and generates a summary report.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_all_tests():
    """Run all tests and generate summary."""
    print("=" * 80)
    print("RUNNING ALL UNIT TESTS")
    print("=" * 80)
    
    # Run numerical tests
    print("\n1. Running element function tests...")
    result1 = pytest.main(['tests/test_elements.py', '-v', '--tb=short'])
    
    print("\n2. Running system matrix tests...")
    result2 = pytest.main(['tests/test_system_matrices.py', '-v', '--tb=short'])
    
    print("\n3. Running wavevector tests...")
    result3 = pytest.main(['tests/test_wavevectors.py', '-v', '--tb=short'])
    
    print("\n4. Running utility function tests...")
    result4 = pytest.main(['tests/test_utils.py', '-v', '--tb=short'])
    
    print("\n5. Running design function tests...")
    result5 = pytest.main(['tests/test_design.py', '-v', '--tb=short'])
    
    print("\n6. Running kernel function tests...")
    result6 = pytest.main(['tests/test_kernels.py', '-v', '--tb=short'])
    
    print("\n7. Running dispersion function tests...")
    result7 = pytest.main(['tests/test_dispersion.py', '-v', '--tb=short'])
    
    print("\n" + "=" * 80)
    print("NUMERICAL TESTS COMPLETE")
    print("=" * 80)
    
    # Run plotting tests (generates plots)
    print("\n8. Running plotting tests (generates plots for visual inspection)...")
    from tests.test_plotting import run_all_plotting_tests
    run_all_plotting_tests()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review test output above for any failures")
    print("2. Check plots in test_plots/ directory")
    print("3. Visually compare plots with MATLAB outputs")
    print("4. Update TEST_EQUIVALENCE.md with results")
    print("=" * 80)


if __name__ == '__main__':
    run_all_tests()



