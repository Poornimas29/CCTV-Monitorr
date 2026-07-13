"""
Unit tests for Phase 2 — Employee Management module.

Covers:
  - Employee configuration integrity (field presence, types, unique IDs).
  - EmployeeManager.load_employees() — successful load and error on duplicate IDs.
  - EmployeeManager.get_employee_by_id() — found and not-found cases.
  - EmployeeManager.get_all_employees() — returns all records.
  - EmployeeManager.get_employee_images() — correct image discovery and extension filtering.
  - EmployeeManager.validate_image_folders() — folder existence checks.
  - EmployeeManager.print_summary() — smoke test (no exceptions).
  - EmployeeManager guard — RuntimeError when not yet loaded.
"""

import os
import sys
import tempfile
import unittest

# Ensure project root is in sys.path for running in isolated environments
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from employee_management.employees import EMPLOYEES
from employee_management.employee_manager import EmployeeManager


class TestEmployeeConfig(unittest.TestCase):
    """Validates the static employee configuration list."""

    def test_employees_list_not_empty(self) -> None:
        """EMPLOYEES must contain at least one entry."""
        self.assertGreater(len(EMPLOYEES), 0, "EMPLOYEES config must not be empty.")

    def test_required_fields_present(self) -> None:
        """Every employee record must have all required fields."""
        required_fields = {
            "employee_id", "name", "status", "image_folder",
        }
        for record in EMPLOYEES:
            with self.subTest(employee=record.get("employee_id", "UNKNOWN")):
                missing = required_fields - record.keys()
                self.assertFalse(
                    missing,
                    f"Employee record is missing fields: {missing}",
                )

    def test_all_field_values_are_strings(self) -> None:
        """All field values must be non-empty strings."""
        required_fields = [
            "employee_id", "name", "status", "image_folder",
        ]
        for record in EMPLOYEES:
            with self.subTest(employee=record.get("employee_id", "UNKNOWN")):
                for field in required_fields:
                    value = record.get(field, "")
                    self.assertIsInstance(value, str, f"Field '{field}' must be a string.")
                    self.assertTrue(value.strip(), f"Field '{field}' must not be empty.")

    def test_employee_ids_are_unique_in_config(self) -> None:
        """Employee IDs in the config must all be unique."""
        ids = [r["employee_id"] for r in EMPLOYEES]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate employee IDs found in config.")

    def test_status_values_are_valid(self) -> None:
        """Status must be either 'Active' or 'Inactive'."""
        valid_statuses = {"Active", "Inactive"}
        for record in EMPLOYEES:
            with self.subTest(employee=record.get("employee_id")):
                self.assertIn(
                    record["status"],
                    valid_statuses,
                    f"Invalid status '{record['status']}'. Must be Active or Inactive.",
                )


class TestEmployeeManagerLoad(unittest.TestCase):
    """Tests for loading and integrity validation."""

    def test_load_employees_succeeds(self) -> None:
        """load_employees() must complete without raising exceptions."""
        manager = EmployeeManager()
        manager.load_employees()
        self.assertTrue(manager._loaded)

    def test_loaded_count_matches_config(self) -> None:
        """Number of loaded employees must equal the config list length."""
        manager = EmployeeManager()
        manager.load_employees()
        self.assertEqual(len(manager.get_all_employees()), len(EMPLOYEES))

    def test_duplicate_id_raises_value_error(self) -> None:
        """load_employees() must raise ValueError when duplicate IDs exist."""
        manager = EmployeeManager()

        # Temporarily patch the config to inject a duplicate
        from employee_management import employee_manager as em_module
        original = em_module.EMPLOYEES
        try:
            em_module.EMPLOYEES = [
                {
                    "employee_id": "EMP999",
                    "name": "Alice",
                    "department": "Eng",
                    "designation": "Dev",
                    "status": "Active",
                    "image_folder": "employee_images/EMP999",
                },
                {
                    "employee_id": "EMP999",  # duplicate
                    "name": "Bob",
                    "department": "Ops",
                    "designation": "Ops Lead",
                    "status": "Active",
                    "image_folder": "employee_images/EMP999b",
                },
            ]
            with self.assertRaises(ValueError):
                manager.load_employees()
        finally:
            em_module.EMPLOYEES = original

    def test_not_loaded_raises_runtime_error(self) -> None:
        """Accessing data before load_employees() must raise RuntimeError."""
        manager = EmployeeManager()
        with self.assertRaises(RuntimeError):
            manager.get_all_employees()


class TestEmployeeManagerQueries(unittest.TestCase):
    """Tests for employee data retrieval methods."""

    def setUp(self) -> None:
        self.manager = EmployeeManager()
        self.manager.load_employees()

    def test_get_employee_by_valid_id(self) -> None:
        """get_employee_by_id() must return the correct record for a known ID."""
        first_emp = EMPLOYEES[0]
        result = self.manager.get_employee_by_id(first_emp["employee_id"])
        self.assertIsNotNone(result)
        self.assertEqual(result["employee_id"], first_emp["employee_id"])
        self.assertEqual(result["name"], first_emp["name"])

    def test_get_employee_by_invalid_id_returns_none(self) -> None:
        """get_employee_by_id() must return None for an unknown ID."""
        result = self.manager.get_employee_by_id("EMP_DOES_NOT_EXIST")
        self.assertIsNone(result)

    def test_get_all_employees_returns_list(self) -> None:
        """get_all_employees() must return a list."""
        result = self.manager.get_all_employees()
        self.assertIsInstance(result, list)

    def test_get_all_employees_not_empty(self) -> None:
        """get_all_employees() must not be empty after loading."""
        self.assertGreater(len(self.manager.get_all_employees()), 0)

    def test_loaded_records_contain_abs_folder_path(self) -> None:
        """Each loaded record must contain the resolved absolute image folder path."""
        for employee in self.manager.get_all_employees():
            self.assertIn("image_folder_abs", employee)
            self.assertTrue(os.path.isabs(employee["image_folder_abs"]))


class TestEmployeeManagerImages(unittest.TestCase):
    """Tests for image discovery."""

    def setUp(self) -> None:
        # Build a temporary directory with mock employee image files
        self._tmpdir = tempfile.TemporaryDirectory()
        self._root = self._tmpdir.name

        # Create a fake EMP_TEST folder with 3 image files and 1 non-image file
        self._emp_folder = os.path.join(self._root, "employee_images", "EMP_TEST")
        os.makedirs(self._emp_folder)
        for filename in ["face1.jpg", "face2.png", "face3.jpeg", "notes.txt"]:
            open(os.path.join(self._emp_folder, filename), "w").close()

        # Patch EMPLOYEES to contain one test employee pointing to the temp folder
        from employee_management import employee_manager as em_module
        self._em_module = em_module
        self._original_employees = em_module.EMPLOYEES
        em_module.EMPLOYEES = [
            {
                "employee_id": "EMP_TEST",
                "name": "Test Person",
                "department": "QA",
                "designation": "Tester",
                "status": "Active",
                "image_folder": "employee_images/EMP_TEST",
            }
        ]

        self.manager = EmployeeManager(project_root=self._root)
        self.manager.load_employees()

    def tearDown(self) -> None:
        self._em_module.EMPLOYEES = self._original_employees
        self._tmpdir.cleanup()

    def test_get_employee_images_returns_only_image_files(self) -> None:
        """get_employee_images() must return only supported image extensions."""
        images = self.manager.get_employee_images("EMP_TEST")
        self.assertEqual(len(images), 3, "Should find 3 image files, not the .txt file.")

    def test_get_employee_images_returns_sorted_list(self) -> None:
        """get_employee_images() must return a sorted list of absolute paths."""
        images = self.manager.get_employee_images("EMP_TEST")
        self.assertEqual(images, sorted(images))

    def test_get_employee_images_returns_absolute_paths(self) -> None:
        """Every path returned by get_employee_images() must be absolute."""
        for path in self.manager.get_employee_images("EMP_TEST"):
            self.assertTrue(os.path.isabs(path))

    def test_get_employee_images_unknown_id_returns_empty(self) -> None:
        """get_employee_images() must return [] for an unknown employee ID."""
        result = self.manager.get_employee_images("EMP_UNKNOWN")
        self.assertEqual(result, [])

    def test_get_employee_images_missing_folder_returns_empty(self) -> None:
        """get_employee_images() must return [] when the image folder does not exist."""
        from employee_management import employee_manager as em_module
        em_module.EMPLOYEES = [
            {
                "employee_id": "EMP_MISSING",
                "name": "Ghost",
                "department": "None",
                "designation": "None",
                "status": "Active",
                "image_folder": "employee_images/EMP_MISSING_FOLDER",
            }
        ]
        manager = EmployeeManager(project_root=self._root)
        manager.load_employees()
        result = manager.get_employee_images("EMP_MISSING")
        self.assertEqual(result, [])


class TestEmployeeManagerFolderValidation(unittest.TestCase):
    """Tests for image folder existence validation."""

    def test_validate_image_folders_returns_dict(self) -> None:
        """validate_image_folders() must return a dict keyed by employee_id."""
        manager = EmployeeManager()
        manager.load_employees()
        result = manager.validate_image_folders()
        self.assertIsInstance(result, dict)

    def test_validate_image_folders_all_keys_match_employees(self) -> None:
        """validate_image_folders() must contain one entry per loaded employee."""
        manager = EmployeeManager()
        manager.load_employees()
        result = manager.validate_image_folders()
        for employee in manager.get_all_employees():
            self.assertIn(employee["employee_id"], result)


class TestEmployeeManagerPrintSummary(unittest.TestCase):
    """Smoke test for print_summary() — verifies it runs without exceptions."""

    def test_print_summary_no_exception(self) -> None:
        """print_summary() must not raise any exception."""
        manager = EmployeeManager()
        manager.load_employees()
        try:
            manager.print_summary()  # output goes to stdout — just check no crash
        except Exception as exc:
            self.fail(f"print_summary() raised an unexpected exception: {exc}")


if __name__ == "__main__":
    unittest.main()
