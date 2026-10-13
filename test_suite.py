"""
Comprehensive Test Suite Runner for Deep Learning & Computer Vision Suite
Executes unit and integration test assertions across NLP, Vision, and Tabular modules.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from tests.test_nlp import TestNLPPipeline
from tests.test_vision import TestComputerVisionSuite
from tests.test_tabular import TestTabularModels


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestNLPPipeline))
    test_suite.addTest(unittest.makeSuite(TestComputerVisionSuite))
    test_suite.addTest(unittest.makeSuite(TestTabularModels))
    return test_suite


if __name__ == "__main__":
    print("=" * 65)
    print(" RUNNING DEEP LEARNING & COMPUTER VISION SUITE TEST HARNESS")
    print("=" * 65)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    sys.exit(0 if result.wasSuccessful() else 1)
