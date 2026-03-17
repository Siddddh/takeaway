import unittest
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_tests():
    """Run the test suite and report results."""
    try:
        # Add src to Python path
        src_path = str(Path(__file__).parent)
        if src_path not in sys.path:
            sys.path.append(src_path)
        
        # Import test suite
        from tests.test_fraud_detection import TestFraudDetection
        
        # Create test suite
        test_suite = TestFraudDetection()
        
        # Run tests
        logger.info("Starting test suite...")
        test_suite.run_all_tests()
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    run_tests() 