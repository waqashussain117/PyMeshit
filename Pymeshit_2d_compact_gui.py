"""Compact one-click 2D mesh window (no tabs)."""

from __future__ import annotations

import csv
import os
from itertools import cycle
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from Pymeshit.two_d_workflow import TwoDInputFeature, TwoDRunConfig, TwoDRunResult, run_two_d_one_click


class TwoDCompactMeshWindow(QDialog):
    """Small popup for one-click 2D meshing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PyMeshIt 2D Mesh (Compact)")
        self.setMinimumSize(860, 600)
        self.resize(900, 640)
        self.setMaximumSize(1000, 760)

        self.features: List[TwoDInputFeature] = []
        self.last_result: Optional[TwoDRunResult] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        self.load_btn = QPushButton("Load 2D Inputs")
        self.load_btn.clicked.connect(self._load_inputs)
        toolbar.addWidget(self.load_btn)

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run_one_click)
        self.run_btn.setEnabled(False)
        toolbar.addWidget(self.run_btn)

        self.export_plc_btn = QPushButton("Export PLC")
        self.export_plc_btn.clicked.connect(self._export_plc)
        self.export_plc_btn.setEnabled(False)
        toolbar.addWidget(self.export_plc_btn)

        self.export_mesh_btn = QPushButton("Export Mesh CSV")
        self.export_mesh_btn.clicked.connect(self._export_mesh_csv)
        self.export_mesh_btn.setEnabled(False)
        toolbar.addWidget(self.export_mesh_btn)

        self.export_fig_btn = QPushButton("Export Figure")
        self.export_fig_btn.clicked.connect(self._export_figure)
        self.export_fig_btn.setEnabled(False)
        toolbar.addWidget(self.export_fig_btn)

        toolbar.addStretch()
        root.addLayout(toolbar)

        middle = QHBoxLayout()

        left_panel = QVBoxLayout()
        feature_group = QGroupBox("Loaded Features")
        feature_layout = QVBoxLayout(feature_group)
        self.feature_list = QListWidget()
        self.feature_list.setMinimumWidth(280)
        feature_layout.addWidget(self.feature_list)
        left_panel.addWidget(feature_group)

        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        self.status_box = QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMaximumHeight(170)
        status_layout.addWidget(self.status_box)
        left_panel.addWidget(status_group)
        left_panel.addStretch()

        middle.addLayout(left_panel, 0)

        self.figure = Figure(figsize=(5.5, 4.5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("2D Input Preview")
        self.ax.grid(True, alpha=0.2)
        self.canvas = FigureCanvasQTAgg(self.figure)
        middle.addWidget(self.canvas, 1)

        root.addLayout(middle, 1)

        self._append_status("Load 2D line/polygon files, then click Run.")
        self._plot_loaded_inputs()

    def _append_status(self, text: str) -> None:
        self.status_box.append(text)

    def _refresh_feature_list(self) -> None:
        self.feature_list.clear()
        for feat in self.features:
            kind = "CLOSED" if feat.is_closed else "OPEN"
            self.feature_list.addItem(f"{feat.name}  |  {kind}  |  {len(feat.coords)} pts")
        self.run_btn.setEnabled(bool(self.features))

    def _read_xy_file(self, file_path: str) -> np.ndarray:
        points = []
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                row = line.strip()
                if not row or row.startswith("#"):
                    continue
                clean = row.replace(";", " ").replace(",", " ").replace("\t", " ")
                parts = [p for p in clean.split(" ") if p]
                if len(parts) < 2:
                    continue
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                except ValueError:
                    continue
                points.append([x, y])

        if len(points) < 2:
            raise ValueError("needs at least 2 numeric XY points")

        return np.asarray(points, dtype=float)

    def _load_inputs(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select 2D input files",
            "",
            "Data files (*.txt *.csv *.dat);;All files (*.*)",
        )
        if not file_paths:
            return

        loaded = 0
        for path in file_paths:
            name = os.path.basename(path)
            try:
                xy = self._read_xy_file(path)
                is_closed = bool(len(xy) >= 3 and np.linalg.norm(xy[0] - xy[-1]) <= 1e-7)
                self.features.append(
                    TwoDInputFeature(name=name, coords=xy, is_closed=is_closed, source_path=path)
                )
                loaded += 1
            except Exception as exc:
                self._append_status(f"{name}: failed to load ({exc})")

        self._refresh_feature_list()
        self.last_result = None
        self.export_plc_btn.setEnabled(False)
        self.export_mesh_btn.setEnabled(False)
        self.export_fig_btn.setEnabled(False)
        self._append_status(f"Loaded {loaded} file(s).")
        self._plot_loaded_inputs()

    def _plot_loaded_inputs(self) -> None:
        """Preview raw loaded input geometry before meshing."""
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.2)
        self.ax.set_title("2D Input Preview")
        self.ax.set_xlabel("Distance (km)")
        self.ax.set_ylabel("Depth/Elevation (km)")

        if not self.features:
            self.ax.text(
                0.5,
                0.5,
                "Load 2D inputs to preview geometry",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
                color="#666666",
                fontsize=10,
            )
            self.figure.tight_layout()
            self.canvas.draw_idle()
            return

        closed_palette = cycle(["#0D47A1", "#1565C0", "#1E88E5", "#283593"])
        open_palette = cycle(["#B71C1C", "#C62828", "#D32F2F", "#E53935"])

        legend_flags = {"closed": False, "open": False}
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")

        for feat in self.features:
            pts = np.asarray(feat.coords, dtype=float)
            if pts.size == 0 or pts.shape[0] < 2:
                continue

            min_x = min(min_x, float(np.min(pts[:, 0])))
            max_x = max(max_x, float(np.max(pts[:, 0])))
            min_y = min(min_y, float(np.min(pts[:, 1])))
            max_y = max(max_y, float(np.max(pts[:, 1])))

            if feat.is_closed:
                color = next(closed_palette)
                label = "Closed polygon(s)" if not legend_flags["closed"] else None
                legend_flags["closed"] = True
                self.ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.8, alpha=0.95, label=label)
            else:
                color = next(open_palette)
                label = "Open line(s)" if not legend_flags["open"] else None
                legend_flags["open"] = True
                self.ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    color=color,
                    linewidth=1.4,
                    linestyle="--",
                    alpha=0.9,
                    label=label,
                )

        if min_x < max_x and min_y < max_y:
            dx = max_x - min_x
            dy = max_y - min_y
            pad_x = max(dx * 0.05, 0.2)
            pad_y = max(dy * 0.05, 0.2)
            self.ax.set_xlim(min_x - pad_x, max_x + pad_x)
            self.ax.set_ylim(min_y - pad_y, max_y + pad_y)

        if legend_flags["closed"] or legend_flags["open"]:
            self.ax.legend(loc="best", fontsize=8, frameon=False)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _run_one_click(self) -> None:
        if not self.features:
            QMessageBox.warning(self, "No Data", "Load at least one input first.")
            return

        self._append_status("Running one-click 2D workflow...")
        QApplication.processEvents()

        try:
            result = run_two_d_one_click(self.features, TwoDRunConfig())
            self.last_result = result
            self._plot_result(result)

            msg = (
                f"Done: {len(result.vertices)} vertices, {len(result.triangles)} triangles, "
                f"{len(result.regions)} region(s), {len(result.constraint_lines)} open-line constraint(s)."
            )
            self._append_status(msg)
            for warning in result.warnings:
                self._append_status(f"Warning: {warning}")

            self.export_plc_btn.setEnabled(True)
            self.export_mesh_btn.setEnabled(True)
            self.export_fig_btn.setEnabled(True)

        except Exception as exc:
            self._append_status(f"Run failed: {exc}")
            QMessageBox.critical(self, "2D Run Failed", str(exc))

    def _plot_result(self, result: TwoDRunResult) -> None:
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.2)
        self.ax.set_title("Final 2D Mesh + Boundaries")

        if result.vertices.size > 0 and result.triangles.size > 0:
            tri = Triangulation(result.vertices[:, 0], result.vertices[:, 1], result.triangles)
            self.ax.triplot(tri, color="#8A8A8A", linewidth=0.5, alpha=0.75)

        for loop in result.boundary_loops:
            if len(loop) >= 2:
                self.ax.plot(loop[:, 0], loop[:, 1], color="black", linewidth=1.6)

        for line in result.constraint_lines:
            if len(line) >= 2:
                self.ax.plot(line[:, 0], line[:, 1], color="#C62828", linewidth=1.2, linestyle="--")

        for region in result.regions:
            seed = region.get("seed")
            rid = region.get("id")
            if seed and rid is not None:
                self.ax.text(float(seed[0]), float(seed[1]), str(rid), fontsize=7, color="#0D47A1")

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _export_plc(self) -> None:
        if self.last_result is None:
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export PLC",
            "mesh_2d.poly",
            "Triangle PLC (*.poly)",
        )
        if not out_path:
            return

        result = self.last_result
        with open(out_path, "w", encoding="utf-8") as fh:
            pts = result.plc_points
            segs = result.plc_segments
            holes = result.plc_holes

            fh.write(f"{len(pts)} 2 0 0\n")
            for i, p in enumerate(pts, start=1):
                fh.write(f"{i} {p[0]:.12g} {p[1]:.12g}\n")

            fh.write(f"{len(segs)} 0\n")
            for i, seg in enumerate(segs, start=1):
                fh.write(f"{i} {int(seg[0]) + 1} {int(seg[1]) + 1}\n")

            fh.write(f"{len(holes)}\n")
            for i, hp in enumerate(holes, start=1):
                fh.write(f"{i} {hp[0]:.12g} {hp[1]:.12g}\n")

            regions = result.regions
            fh.write(f"{len(regions)}\n")
            for i, region in enumerate(regions, start=1):
                seed = region.get("seed", (0.0, 0.0))
                attr = int(region.get("id", i))
                max_area = float(region.get("max_area", 0.0))
                fh.write(f"{i} {seed[0]:.12g} {seed[1]:.12g} {attr} {max_area:.12g}\n")

        self._append_status(f"PLC exported: {out_path}")

    def _export_mesh_csv(self) -> None:
        if self.last_result is None:
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Mesh CSV",
            "mesh_2d.csv",
            "CSV (*.csv)",
        )
        if not out_path:
            return

        result = self.last_result
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["[vertices]"])
            writer.writerow(["id", "x", "y"])
            for i, p in enumerate(result.vertices):
                writer.writerow([i, float(p[0]), float(p[1])])

            writer.writerow([])
            writer.writerow(["[triangles]"])
            writer.writerow(["id", "v0", "v1", "v2"])
            for i, tri in enumerate(result.triangles):
                writer.writerow([i, int(tri[0]), int(tri[1]), int(tri[2])])

        self._append_status(f"Mesh CSV exported: {out_path}")

    def _export_figure(self) -> None:
        if self.last_result is None:
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Figure",
            "mesh_2d_preview.png",
            "PNG (*.png);;SVG (*.svg)",
        )
        if not out_path:
            return

        self.figure.savefig(out_path, dpi=300, bbox_inches="tight")
        self._append_status(f"Figure exported: {out_path}")
