"""
Employee Manager module for Phase 2 of the AI Employee Monitoring System.

Responsibilities:
  - Load all employees from the temporary configuration registry.
  - Validate that all employee IDs are unique.
  - Validate that every employee's image folder exists on disk.
  - Return employee details by ID.
  - Return all face image paths for a given employee.
  - Print a formatted startup summary.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from employee_management.employees import EMPLOYEES

logger = logging.getLogger(__name__)

# Supported face image file extensions
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
)


class EmployeeManager:
    """
    Manages the temporary employee registry.

    Loads employee data from the static configuration, validates integrity
    (unique IDs, existing image folders), and exposes query methods that
    the Face Recognition phase will consume.
    """

    def __init__(self, project_root: Optional[str] = None) -> None:
        """
        Initialises the manager.

        Args:
            project_root: Absolute path to the project root directory.
                          Defaults to the parent of the employee_management package.
        """
        if project_root:
            self._root: Path = Path(project_root).resolve()
        else:
            # Resolve project root from this file's location  (…/employee_management/../)
            self._root = Path(__file__).resolve().parent.parent

        self._employees: Dict[str, dict] = {}
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_employees(self) -> None:
        """
        Loads all employees by combining the static config registry with a dynamic scan of new folders on disk.
        """
        logger.info("Loading employee registry (merging static config and dynamic folders)...")

        # Read from local module-level global EMPLOYEES to support test patching
        raw = list(EMPLOYEES)
        existing_ids = {record.get("employee_id") for record in raw if record.get("employee_id")}

        # Scan for any additional folders on disk that are not in the static config
        images_dir = (self._root / "employee_images").resolve()
        if images_dir.exists():
            for item in images_dir.iterdir():
                if item.is_dir():
                    emp_id = item.name
                    if emp_id not in existing_ids:
                        # Determine name dynamically
                        name = emp_id
                        name_file = item / "name.txt"
                        if name_file.exists():
                            try:
                                name = name_file.read_text(encoding="utf-8").strip()
                            except Exception:
                                pass
                        
                        raw.append({
                            "employee_id": emp_id,
                            "name": name,
                            "status": "Active",
                            "image_folder": f"employee_images/{emp_id}"
                        })

        # ── 1. Validate unique IDs ────────────────────────────────────
        self._validate_unique_ids(raw)

        # ── 2. Load into internal dict and validate image folders ─────
        self._employees = {}
        for record in raw:
            emp_id = record["employee_id"]

            # Resolve image folder to absolute path
            folder_path = (self._root / record["image_folder"]).resolve()
            record = dict(record)
            record["image_folder_abs"] = str(folder_path)

            self._employees[emp_id] = record

        # ── 3. Report folder warnings (non-fatal) ────────────────────
        self._validate_image_folders()

        self._loaded = True
        logger.info(
            "Employee registry loaded successfully. Total employees: %d",
            len(self._employees),
        )

    def get_employee_by_id(self, employee_id: str) -> Optional[dict]:
        """
        Returns the employee record matching the given ID.

        Args:
            employee_id: The employee ID string (e.g. 'EMP001').

        Returns:
            Employee dict if found, otherwise None.
        """
        self._ensure_loaded()
        return self._employees.get(employee_id)

    def get_all_employees(self) -> List[dict]:
        """
        Returns a list of all loaded employee records.

        Returns:
            List of employee dicts.
        """
        self._ensure_loaded()
        return list(self._employees.values())

    def get_employee_images(self, employee_id: str) -> List[str]:
        """
        Returns a sorted list of absolute image file paths for the given employee.

        Only files with supported image extensions are included.
        Returns an empty list if the folder is missing or contains no images.

        Args:
            employee_id: The employee ID string (e.g. 'EMP001').

        Returns:
            Sorted list of absolute image file path strings.
        """
        self._ensure_loaded()
        employee = self._employees.get(employee_id)
        if not employee:
            logger.warning("get_employee_images: Unknown employee ID '%s'.", employee_id)
            return []

        folder = Path(employee["image_folder_abs"])
        if not folder.exists():
            logger.warning(
                "Image folder does not exist for %s: %s", employee_id, folder
            )
            return []

        images: List[str] = sorted(
            str(f)
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
        )
        return images

    def validate_image_folders(self) -> Dict[str, bool]:
        """
        Checks whether the image folder exists for every employee.

        Returns:
            Dict mapping employee_id -> True (folder exists) / False (missing).
        """
        self._ensure_loaded()
        return self._validate_image_folders(log=True)

    def print_summary(self) -> None:
        """
        Prints a formatted startup summary to stdout.

        Output columns:
          - Employee ID
          - Employee Name
          - Number of Images found in the image folder
          - Total employees loaded (footer)
        """
        self._ensure_loaded()

        col_w = (10, 22, 12)  # column widths
        divider = "-" * (sum(col_w) + 10)

        print()
        print("=" * (sum(col_w) + 10))
        print("  PHASE 2 — EMPLOYEE REGISTRY LOADED")
        print("=" * (sum(col_w) + 10))
        print(
            f"  {'Employee ID':<{col_w[0]}}  {'Name':<{col_w[1]}}  {'Images':<{col_w[2]}}"
        )
        print(divider)

        for emp_id, record in self._employees.items():
            images = self.get_employee_images(emp_id)
            img_count = len(images)
            img_label = str(img_count) if img_count > 0 else "0 (no images yet)"
            print(
                f"  {emp_id:<{col_w[0]}}  {record['name']:<{col_w[1]}}  {img_label:<{col_w[2]}}"
            )

        print(divider)
        print(f"  Total Employees Loaded : {len(self._employees)}")
        print("=" * (sum(col_w) + 10))
        print()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Raises RuntimeError if load_employees() has not been called yet."""
        if not self._loaded:
            raise RuntimeError(
                "EmployeeManager: call load_employees() before accessing employee data."
            )

    def _validate_unique_ids(self, records: List[dict]) -> None:
        """
        Checks that every employee_id in the config list is unique.

        Args:
            records: Raw list of employee dicts from the config.

        Raises:
            ValueError: If any duplicate employee ID is detected.
        """
        seen: set[str] = set()
        duplicates: set[str] = set()
        for record in records:
            emp_id = record.get("employee_id", "")
            if emp_id in seen:
                duplicates.add(emp_id)
            seen.add(emp_id)

        if duplicates:
            raise ValueError(
                f"Duplicate employee IDs detected in configuration: {sorted(duplicates)}. "
                "Each employee ID must be unique."
            )

    def _validate_image_folders(self, log: bool = False) -> Dict[str, bool]:
        """
        Internal helper that checks image folder existence.

        Args:
            log: When True, emits a warning log for each missing folder.

        Returns:
            Dict mapping employee_id -> folder_exists (bool).
        """
        results: Dict[str, bool] = {}
        for emp_id, record in self._employees.items():
            folder = Path(record["image_folder_abs"])
            exists = folder.exists()
            results[emp_id] = exists
            if not exists:
                msg = (
                    f"Image folder missing for {emp_id} ({record['name']}): {folder}. "
                    "Please create the folder and add face images before running Face Recognition."
                )
                if log:
                    logger.warning(msg)
                else:
                    logger.warning(msg)
        return results
