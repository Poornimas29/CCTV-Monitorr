"""
Phase 2 — Employee Registry Loader
===================================
Standalone startup script for the Employee Management module.

Run this script to verify that all employees are loaded correctly and
their image folders are accessible before moving on to Face Recognition.

Usage:
    python phase2_loader.py
"""

import logging
import os
import sys

# Ensure project root is in sys.path for isolated / embedded environments
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.logging_config import setup_logging
from employee_management.employee_manager import EmployeeManager

# Initialise logging
setup_logging()
logger = logging.getLogger("phase2_loader")


def main() -> None:
    """Entry point for Phase 2 employee registry verification."""
    logger.info("Phase 2 - Employee Management module starting...")

    manager = EmployeeManager()

    try:
        manager.load_employees()
    except ValueError as exc:
        logger.error("Failed to load employee registry: %s", exc)
        sys.exit(1)

    # Print formatted startup summary to console
    manager.print_summary()

    # ── Detailed per-employee report ──────────────────────────────────
    logger.info("Running per-employee image folder validation...")

    all_employees = manager.get_all_employees()
    for employee in all_employees:
        emp_id = employee["employee_id"]
        images = manager.get_employee_images(emp_id)

        logger.info(
            "[%s] %s | Dept: %s | Role: %s | Status: %s | Images: %d",
            emp_id,
            employee["name"],
            employee.get("department", "N/A"),
            employee.get("designation", "N/A"),
            employee["status"],
            len(images),
        )

        if images:
            for img_path in images:
                logger.info("   Image -> %s", img_path)
        else:
            logger.warning(
                "   [%s] No face images found. Add images to: %s",
                emp_id,
                employee["image_folder_abs"],
            )

    logger.info(
        "Phase 2 employee registry ready. %d employee(s) loaded for Face Recognition.",
        len(all_employees),
    )


if __name__ == "__main__":
    main()
