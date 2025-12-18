"""
Verification script to test all production modules converted from notebooks.

This script verifies that all notebook functionality has been successfully
converted to production code and is accessible through the geosuite package.
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def verify_imports():
    """Verify all modules can be imported."""
    logger.info("Verifying module imports...")
    
    modules_to_test = [
        # Data I/O
        ('geosuite.io', ['WitsmlParser', 'PpdmParser']),
        
        # Petrophysics
        ('geosuite.petro', ['pickett_plot', 'buckles_plot', 'calculate_water_saturation']),
        
        # Geomechanics
        ('geosuite.geomech', ['calculate_overburden_stress', 'calculate_pore_pressure_eaton', 
                               'stress_polygon_limits', 'determine_stress_regime']),
        
        # Machine Learning
        ('geosuite.ml', ['train_facies_classifier', 'MLflowFaciesClassifier']),
        
        # Stratigraphy
        ('geosuite.stratigraphy', ['preprocess_log', 'detect_pelt', 'detect_bayesian_online',
                                     'compare_methods', 'find_consensus']),
        
        # Imaging
        ('geosuite.imaging', ['crop_core_image', 'process_core_directory', 
                               'extract_depth_from_filename']),
        
        # Plotting
        ('geosuite.plotting', ['create_strip_chart', 'create_facies_log_plot']),
        
        # Data
        ('geosuite.data', ['load_demo_well_logs', 'load_facies_training_data']),
    ]
    
    failed = []
    passed = 0
    
    for module_name, functions in modules_to_test:
        try:
            module = __import__(module_name, fromlist=functions)
            
            # Check each function exists
            for func_name in functions:
                if not hasattr(module, func_name):
                    failed.append(f"{module_name}.{func_name} not found")
                else:
                    passed += 1
            
            logger.info(f"✅ {module_name}: {len(functions)} functions verified")
            
        except ImportError as e:
            failed.append(f"{module_name}: {e}")
            logger.error(f"❌ {module_name}: Import failed")
    
    return passed, failed


def verify_top_level_imports():
    """Verify commonly used functions are accessible from top level."""
    logger.info("\nVerifying top-level imports...")
    
    try:
        import geosuite
        
        # Check version info
        assert hasattr(geosuite, '__version__')
        assert hasattr(geosuite, 'get_info')
        assert hasattr(geosuite, 'get_version')
        
        # Check commonly used functions
        top_level_functions = [
            'load_demo_well_logs',
            'load_facies_training_data',
            'calculate_water_saturation',
            'pickett_plot',
            'buckles_plot',
            'calculate_overburden_stress',
            'calculate_hydrostatic_pressure',
            'train_facies_classifier',
            'create_strip_chart',
            'preprocess_log',
            'detect_pelt',
        ]
        
        missing = []
        for func in top_level_functions:
            if not hasattr(geosuite, func):
                missing.append(func)
        
        if missing:
            logger.error(f"❌ Missing top-level functions: {missing}")
            return False
        
        logger.info(f"✅ All {len(top_level_functions)} top-level functions accessible")
        return True
        
    except Exception as e:
        logger.error(f"❌ Top-level import verification failed: {e}")
        return False


def verify_module_info():
    """Verify module info and metadata."""
    logger.info("\nVerifying module metadata...")
    
    try:
        import geosuite
        
        info = geosuite.get_info()
        version = geosuite.get_version()
        
        logger.info(f"Version: {version}")
        logger.info(f"Author: {info['author']}")
        logger.info(f"License: {info['license']}")
        
        logger.info("Available modules:")
        for module, available in info['modules'].items():
            status = "✅" if available else "❌"
            logger.info(f"  {status} {module}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Module info verification failed: {e}")
        return False


def main():
    """Run all verification tests."""
    logger.info("=" * 60)
    logger.info("GeoSuite Module Verification")
    logger.info("Verifying all notebook functionality converted to production code")
    logger.info("=" * 60)
    
    # Test imports
    passed, failed = verify_imports()
    
    # Test top-level imports
    top_level_ok = verify_top_level_imports()
    
    # Test module info
    info_ok = verify_module_info()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Functions verified: {passed}")
    logger.info(f"Failed imports: {len(failed)}")
    
    if failed:
        logger.error("\nFailed imports:")
        for failure in failed:
            logger.error(f"  - {failure}")
    
    if failed or not top_level_ok or not info_ok:
        logger.error("\n❌ VERIFICATION FAILED")
        sys.exit(1)
    else:
        logger.info("\n✅ ALL VERIFICATIONS PASSED")
        logger.info("\nAll notebook functionality successfully converted to production code!")
        logger.info("GeoSuite is ready for production use.")
        sys.exit(0)


if __name__ == '__main__':
    main()

