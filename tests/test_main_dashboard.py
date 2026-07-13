import importlib
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestMainDashboardEntrypoint(unittest.TestCase):
    def test_main_module_imports(self) -> None:
        module = importlib.import_module("main")
        self.assertTrue(hasattr(module, "main"))


if __name__ == "__main__":
    unittest.main()
