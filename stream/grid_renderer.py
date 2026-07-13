"""
Grid Renderer for the CCTV Monitor Dashboard.

Combines multiple camera frames into a single monitoring dashboard image
using a dynamic grid layout:

    n cameras  →  cols = ceil(sqrt(n)),  rows = ceil(n / cols)

Examples
--------
    2 cameras  →  1 row  × 2 cols
    4 cameras  →  2 rows × 2 cols
    6 cameras  →  2 rows × 3 cols
    9 cameras  →  3 rows × 3 cols
"""

import math
import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class GridRenderer:
    """
    Builds a combined dashboard image from multiple live camera frames.

    Each camera occupies one *cell* of fixed pixel dimensions.
    Cells are arranged into a rectangular grid, prefixed by a header bar
    that shows the system title, online camera count, and current time.
    """

    # Height of the top dashboard header bar, in pixels.
    HEADER_HEIGHT: int = 52

    def __init__(self, cell_size: Tuple[int, int] = (640, 360)) -> None:
        """
        Args:
            cell_size: ``(width, height)`` in pixels for every camera cell.
        """
        self.cell_w, self.cell_h = cell_size

    # ─────────────────────────────── public API ───────────────────────────────

    def grid_dimensions(self, n: int) -> Tuple[int, int]:
        """
        Calculates the ``(rows, cols)`` grid layout for *n* cameras.

        Args:
            n: Number of cameras to display.

        Returns:
            A ``(rows, cols)`` tuple.
        """
        if n <= 0:
            return 1, 1
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return rows, cols

    def build_dashboard(
        self,
        camera_ids: List[str],
        camera_names: Dict[str, str],
        frames: Dict[str, Optional[np.ndarray]],
        connected: Dict[str, bool],
        fps_map: Dict[str, float],
    ) -> np.ndarray:
        """
        Builds the full dashboard image: header bar stacked above the camera grid.

        Args:
            camera_ids: Ordered list of camera IDs defining display order.
            camera_names: Mapping ``camera_id → display name``.
            frames: Mapping ``camera_id → latest frame`` (``None`` when offline).
            connected: Mapping ``camera_id → connection state``.
            fps_map: Mapping ``camera_id → current measured FPS``.

        Returns:
            A single NumPy image ready to pass to ``cv2.imshow()``.
        """
        grid = self._build_grid(camera_ids, camera_names, frames, connected, fps_map)
        header = self._build_header(grid.shape[1], len(camera_ids), connected)
        return np.vstack([header, grid])

    # ─────────────────────────── private helpers ──────────────────────────────

    def _build_header(
        self,
        width: int,
        cam_count: int,
        connected: Dict[str, bool],
    ) -> np.ndarray:
        """
        Renders the top status bar with title, camera online count, and clock.

        Args:
            width: Total pixel width of the bar (matches the grid width).
            cam_count: Total number of configured cameras.
            connected: Connection state map used to count online cameras.

        Returns:
            A NumPy image of shape ``(HEADER_HEIGHT, width, 3)``.
        """
        bar = np.full((self.HEADER_HEIGHT, width, 3), (18, 14, 12), dtype=np.uint8)

        # Accent rule at the bottom of the header
        cv2.line(bar, (0, self.HEADER_HEIGHT - 2), (width, self.HEADER_HEIGHT - 2), (0, 150, 255), 2)

        # Left: system title
        cv2.putText(
            bar,
            "CCTV MONITOR  |  LIVE DASHBOARD",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 220, 220), 2, cv2.LINE_AA,
        )

        # Right: camera count + current timestamp
        online = sum(1 for v in connected.values() if v)
        ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        right_text = f"Cameras: {online}/{cam_count} Online    {ts}"
        text_w = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        cv2.putText(
            bar,
            right_text,
            (width - text_w - 16, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA,
        )

        return bar

    def _build_grid(
        self,
        camera_ids: List[str],
        camera_names: Dict[str, str],
        frames: Dict[str, Optional[np.ndarray]],
        connected: Dict[str, bool],
        fps_map: Dict[str, float],
    ) -> np.ndarray:
        """
        Assembles the camera cells into the grid image.

        Cells are filled left-to-right, top-to-bottom.  Any trailing
        empty slots (when ``n`` is not a perfect multiple of ``cols``)
        are padded with solid-black cells.

        Args:
            camera_ids: Ordered list of camera IDs.
            camera_names: ``camera_id → display name`` mapping.
            frames: ``camera_id → frame`` mapping.
            connected: ``camera_id → bool`` connection state mapping.
            fps_map: ``camera_id → float`` FPS mapping.

        Returns:
            A NumPy image containing all camera cells in a grid.
        """
        n = len(camera_ids)

        # ── Edge case: nothing configured ────────────────────────────────────
        if n == 0:
            blank = np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "No cameras configured.",
                (32, self.cell_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2, cv2.LINE_AA,
            )
            return blank

        rows, cols = self.grid_dimensions(n)

        # ── Build each cell ───────────────────────────────────────────────────
        cells: List[np.ndarray] = []
        for cam_id in camera_ids:
            cam_name = camera_names.get(cam_id, cam_id)
            is_conn = connected.get(cam_id, False)
            fps = fps_map.get(cam_id, 0.0)
            raw = frames.get(cam_id)

            if raw is not None and is_conn:
                cell = cv2.resize(raw, (self.cell_w, self.cell_h), interpolation=cv2.INTER_AREA)
            else:
                cell = self._make_offline_cell(cam_name)

            cell = self._draw_cell_overlay(cell, cam_name, is_conn, fps)
            # Thin separator border
            cv2.rectangle(cell, (0, 0), (self.cell_w - 1, self.cell_h - 1), (38, 38, 38), 1)
            cells.append(cell)

        # ── Pad trailing empty slots ──────────────────────────────────────────
        total_slots = rows * cols
        while len(cells) < total_slots:
            cells.append(np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8))

        # ── Stack into grid ───────────────────────────────────────────────────
        row_strips = [
            np.hstack(cells[r * cols: (r + 1) * cols])
            for r in range(rows)
        ]
        return np.vstack(row_strips)

    # ─────────────────────────── cell primitives ──────────────────────────────

    def _offline_label_text(self) -> str:
        """Returns the label shown on offline camera placeholders."""
        return "Camera Offline"

    def _make_offline_cell(self, camera_name: str) -> np.ndarray:
        """
        Creates a styled "Camera Offline" placeholder cell.

        Args:
            camera_name: Display name shown beneath the offline indicator.

        Returns:
            A NumPy image of shape ``(cell_h, cell_w, 3)``.
        """
        cell = np.full((self.cell_h, self.cell_w, 3), (14, 12, 10), dtype=np.uint8)

        # Subtle grid texture
        for x in range(0, self.cell_w, 40):
            cv2.line(cell, (x, 0), (x, self.cell_h), (24, 22, 20), 1)
        for y in range(0, self.cell_h, 40):
            cv2.line(cell, (0, y), (self.cell_w, y), (24, 22, 20), 1)

        # Warning icon (filled circle + exclamation mark)
        cx = self.cell_w // 2
        cy = self.cell_h // 2 - 34
        cv2.circle(cell, (cx, cy), 36, (20, 20, 155), -1)
        cv2.circle(cell, (cx, cy), 34, (40, 40, 200), 2)
        cv2.putText(cell, "!", (cx - 9, cy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (240, 240, 255), 3, cv2.LINE_AA)

        # Offline label
        label = self._offline_label_text()
        label_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0][0]
        cv2.putText(
            cell, label,
            ((self.cell_w - label_w) // 2, self.cell_h // 2 + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (70, 70, 210), 2, cv2.LINE_AA,
        )

        # Camera name subtitle
        name_w = cv2.getTextSize(camera_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        cv2.putText(
            cell, camera_name,
            ((self.cell_w - name_w) // 2, self.cell_h // 2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA,
        )

        return cell

    def _draw_cell_overlay(
        self,
        cell: np.ndarray,
        camera_name: str,
        is_connected: bool,
        fps: float,
    ) -> np.ndarray:
        """
        Draws a semi-transparent top and bottom bar on a camera cell with:

        - **Top bar**: camera name (left-aligned).
        - **Bottom bar**: status indicator dot + label (left), FPS (right).

        Args:
            cell: Camera frame or offline placeholder image.
            camera_name: Name to display in the top bar.
            is_connected: Whether the camera is currently connected.
            fps: Current measured FPS of this camera stream.

        Returns:
            The cell image with overlays applied in-place.
        """
        h, w = cell.shape[:2]

        # ── Top bar ──────────────────────────────────────────────────────────
        top_bar = cell.copy()
        cv2.rectangle(top_bar, (0, 0), (w, 44), (10, 8, 8), -1)
        cv2.addWeighted(top_bar, 0.62, cell, 0.38, 0, cell)

        cv2.putText(cell, camera_name, (10, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, (225, 225, 225), 2, cv2.LINE_AA)

        # ── Bottom bar ───────────────────────────────────────────────────────
        bot_bar = cell.copy()
        cv2.rectangle(bot_bar, (0, h - 38), (w, h), (10, 8, 8), -1)
        cv2.addWeighted(bot_bar, 0.62, cell, 0.38, 0, cell)

        # Status dot + label
        dot_color = (0, 200, 0) if is_connected else (30, 30, 210)
        status_text = "Online" if is_connected else "Offline"
        status_color = (30, 210, 30) if is_connected else (70, 70, 220)
        cv2.circle(cell, (14, h - 14), 6, dot_color, -1)
        cv2.putText(cell, status_text, (28, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1, cv2.LINE_AA)

        # FPS label (right-aligned)
        fps_text = f"{fps:.1f} FPS"
        fps_w = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
        cv2.putText(cell, fps_text, (w - fps_w - 10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (175, 175, 175), 1, cv2.LINE_AA)

        return cell
