"""
Temporary Employee Configuration for Phase 2.

This module holds the hardcoded employee registry that acts as the employee data
source until the real HR database integration is available in a later phase.

Each employee record contains:
    employee_id   : Unique identifier (e.g. EMP001)
    name          : Full name of the employee
    status        : Active | Inactive
    image_folder  : Path (relative to project root) to the employee's face image folder
"""

# ---------------------------------------------------------------------------
# TEMPORARY EMPLOYEE REGISTRY
# Replace this list with a database call in a future phase.
# ---------------------------------------------------------------------------

EMPLOYEES: list[dict] = [
    {
        "employee_id": "EMP001",
        "name": "Arun Prakash",
        "status": "Active",
        "image_folder": "employee_images/EMP001",
    },
    {
        "employee_id": "EMP002",
        "name": "Sharma",
        "status": "Active",
        "image_folder": "employee_images/EMP002",
    },
    {
        "employee_id": "EMP003",
        "name": "Rahul",
        "status": "Active",
        "image_folder": "employee_images/EMP003",
    },
]
