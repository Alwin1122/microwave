"""GUI panel for beamforming reconstruction and image display."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from data_loader.dataset_info import MicrowaveDataset
from quality.beamformer_selector import select_best_beamformer
from reconstruction.reconstruction_manager import (
    ROIRefinement,
    infer_reconstruction_assessment,
    infer_reconstruction_config,
    reconstruct_all,
    reconstruct_high_resolution_roi,
)
from roi.roi_detector import ROIResult, detect_roi
from utils.logger import StatusLog, get_logger

logger = get_logger(__name__)


class ReconstructionPanel(QWidget):
    """GUI panel for running beamforming reconstruction and displaying results."""

    reconstruction_completed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dataset: MicrowaveDataset | None = None
        self.processed_dataset: MicrowaveDataset | None = None
        self.status_log = StatusLog()
        self.last_roi_result: ROIResult | None = None
        self.last_roi_refinement: ROIRefinement | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        title = QLabel("Module 3 — Reconstruction & Beamformer Selection")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        config_group = QGroupBox("Reconstruction Configuration")
        form = QFormLayout()

        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["Processed Dataset", "Raw Dataset"])
        form.addRow("Source Signal:", self.data_source_combo)

        self.grid_combo = QComboBox()
        self.grid_combo.addItems(["64x64", "128x128"])
        form.addRow("Grid Resolution:", self.grid_combo)

        self.radius_combo = QComboBox()
        self.radius_combo.addItems(["8 cm", "10 cm", "12 cm"])
        self.radius_combo.setCurrentText("8 cm")
        form.addRow("Antenna Radius:", self.radius_combo)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["3.0e8 m/s (air)", "2.1e8 m/s", "1.8e8 m/s"])
        self.speed_combo.setCurrentText("3.0e8 m/s (air)")
        form.addRow("Wave Speed:", self.speed_combo)

        self.span_combo = QComboBox()
        self.span_combo.addItems(["10 cm x 10 cm", "12 cm x 12 cm", "16 cm x 16 cm"])
        self.span_combo.setCurrentText("10 cm x 10 cm")
        form.addRow("Field of View:", self.span_combo)

        config_group.setLayout(form)
        layout.addWidget(config_group)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run Reconstruction")
        self.run_button.clicked.connect(self.on_run_clicked)
        self.run_button.setEnabled(False)
        run_row.addWidget(self.run_button)

        self.status_label = QLabel("No dataset loaded.")
        run_row.addWidget(self.status_label, stretch=1)
        layout.addLayout(run_row)

        self.content_tabs = QTabWidget()
        layout.addWidget(self.content_tabs, stretch=1)

        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)

        selected_group = QGroupBox("Selected Reconstruction")
        selected_layout = QVBoxLayout()
        self.selected_figure = Figure(figsize=(7, 6))
        self.selected_canvas = FigureCanvasQTAgg(self.selected_figure)
        selected_layout.addWidget(self.selected_canvas)
        selected_group.setLayout(selected_layout)
        overview_layout.addWidget(selected_group, stretch=3)

        overview_bottom = QHBoxLayout()

        summary_group = QGroupBox("Selected Reconstruction Summary")
        summary_layout = QVBoxLayout()
        self.summary_table = QTableWidget(0, 2)
        self.summary_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        summary_layout.addWidget(self.summary_table)
        summary_group.setLayout(summary_layout)
        overview_bottom.addWidget(summary_group, stretch=1)

        roi_group = QGroupBox("ROI Localization")
        roi_layout = QVBoxLayout()
        self.roi_table = QTableWidget(0, 2)
        self.roi_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.roi_table.horizontalHeader().setStretchLastSection(True)
        self.roi_table.verticalHeader().setVisible(False)
        self.roi_table.setEditTriggers(QTableWidget.NoEditTriggers)
        roi_layout.addWidget(self.roi_table)
        roi_group.setLayout(roi_layout)
        overview_bottom.addWidget(roi_group, stretch=1)

        log_group = QGroupBox("Status Log")
        log_layout = QVBoxLayout()
        self.log_view = QLabel()
        self.log_view.setWordWrap(True)
        log_layout.addWidget(self.log_view)
        log_group.setLayout(log_layout)
        overview_bottom.addWidget(log_group, stretch=1)

        overview_layout.addLayout(overview_bottom, stretch=1)

        assumptions_group = QGroupBox("Reconstruction Assumptions")
        assumptions_layout = QVBoxLayout()
        self.assumptions_label = QLabel("No reconstruction assumptions available yet.")
        self.assumptions_label.setWordWrap(True)
        assumptions_layout.addWidget(self.assumptions_label)
        assumptions_group.setLayout(assumptions_layout)
        overview_layout.addWidget(assumptions_group)
        self.content_tabs.addTab(overview_tab, "Overview")

        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(comparison_tab)

        plots_group = QGroupBox("Beamformer Comparison")
        plots_layout = QVBoxLayout()
        self.figure = Figure(figsize=(8, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)
        plots_layout.addWidget(self.canvas)
        plots_group.setLayout(plots_layout)
        comparison_layout.addWidget(plots_group, stretch=3)

        metrics_group = QGroupBox("Beamformer Quality Metrics")
        metrics_layout = QVBoxLayout()
        self.metrics_table = QTableWidget(0, 6)
        self.metrics_table.setHorizontalHeaderLabels(["Algorithm", "SNR", "SCR", "Contrast", "Time (s)", "Score"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        metrics_layout.addWidget(self.metrics_table)
        metrics_group.setLayout(metrics_layout)
        comparison_layout.addWidget(metrics_group, stretch=1)
        self.content_tabs.addTab(comparison_tab, "Beamformer Comparison")

        refinement_tab = QWidget()
        refinement_layout_root = QHBoxLayout(refinement_tab)

        refinement_plot_group = QGroupBox("High-Resolution ROI Reconstruction")
        refinement_plot_layout = QVBoxLayout()
        self.refined_figure = Figure(figsize=(10, 7.0))
        self.refined_canvas = FigureCanvasQTAgg(self.refined_figure)
        self.refined_canvas.setMinimumHeight(500)
        refinement_plot_layout.addWidget(self.refined_canvas)
        refinement_plot_group.setLayout(refinement_plot_layout)
        refinement_layout_root.addWidget(refinement_plot_group, stretch=4)

        refinement_info_group = QGroupBox("Refined ROI Details")
        refinement_info_layout = QVBoxLayout()
        self.refinement_table = QTableWidget(0, 2)
        self.refinement_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.refinement_table.horizontalHeader().setStretchLastSection(True)
        self.refinement_table.verticalHeader().setVisible(False)
        self.refinement_table.setEditTriggers(QTableWidget.NoEditTriggers)
        refinement_info_layout.addWidget(self.refinement_table)
        refinement_info_group.setLayout(refinement_info_layout)
        refinement_layout_root.addWidget(refinement_info_group, stretch=1)
        self.content_tabs.addTab(refinement_tab, "ROI Refinement")

    def _log(self, message: str, level: str = "INFO") -> None:
        line = self.status_log.add(message, level)
        self.log_view.setText(line)

    def set_dataset(self, dataset: MicrowaveDataset) -> None:
        self.dataset = dataset
        self.status_label.setText(f"Loaded dataset: {dataset.file_name}")
        self.run_button.setEnabled(True)
        self._apply_dataset_defaults(dataset)
        self._log(f"Dataset '{dataset.file_name}' ready for reconstruction.")

    def set_processed_dataset(self, dataset: MicrowaveDataset) -> None:
        self.processed_dataset = dataset
        self._log("Processed dataset available for reconstruction.")

    def on_run_clicked(self) -> None:
        if self.dataset is None:
            self._log("No dataset loaded.", "ERROR")
            return

        source = self.data_source_combo.currentText()
        if source == "Processed Dataset" and self.processed_dataset is not None:
            dataset_to_use = self.processed_dataset
        else:
            dataset_to_use = self.dataset

        try:
            self._run_reconstruction(dataset_to_use)
        except Exception as exc:
            self._log(f"Reconstruction failed: {exc}", "ERROR")

    def _run_reconstruction(self, dataset: MicrowaveDataset) -> None:
        self._log("Starting reconstruction...")

        s_params = dataset.s_parameters
        freqs = dataset.frequencies
        config = self._current_reconstruction_config(dataset)

        images, timings = reconstruct_all(
            s_params,
            freqs,
            x_span=config.x_span,
            y_span=config.y_span,
            n_x=config.n_x,
            n_y=config.n_y,
            zero_padding=config.zero_padding,
            wave_speed=config.wave_speed,
            antenna_radius=config.antenna_radius,
            return_timings=True,
        )
        selected_name, quality_metrics = select_best_beamformer(images, timings)
        roi_result = detect_roi(images[selected_name])
        refinement = reconstruct_high_resolution_roi(
            s_params,
            freqs,
            selected_name,
            roi_result.bounding_box,
            images[selected_name].shape,
            full_x_span=config.x_span,
            full_y_span=config.y_span,
            min_grid_size=max(96, config.n_x),
            wave_speed=config.wave_speed,
            antenna_radius=config.antenna_radius,
        )
        self.last_roi_result = roi_result
        self.last_roi_refinement = refinement

        self._plot_selected_image(
            images[selected_name],
            selected_name,
            roi_result,
            config.x_span,
            config.y_span,
        )
        self._plot_images(
            images,
            selected_name,
            roi_result,
            config.x_span,
            config.y_span,
        )
        self._plot_refined_image(refinement)
        self._populate_summary(images, selected_name)
        self._populate_metrics(quality_metrics)
        self._populate_roi(roi_result)
        self._populate_refinement(refinement)
        self._log(
            f"Reconstruction completed. Selected beamformer: {selected_name}. "
            f"ROI bbox={roi_result.bounding_box}. Refined grid={refinement.grid_shape[1]}x{refinement.grid_shape[0]}",
            "SUCCESS",
        )
        self.reconstruction_completed.emit(selected_name)

    def _apply_dataset_defaults(self, dataset: MicrowaveDataset) -> None:
        assessment = infer_reconstruction_assessment(dataset)
        config = assessment.config
        self.radius_combo.setCurrentText(self._radius_text(config.antenna_radius))
        self.speed_combo.setCurrentText(self._speed_text(config.wave_speed))
        self.span_combo.setCurrentText(self._span_text(config.x_span, config.y_span))
        self._update_assumption_summary(assessment.source_notes, assessment.warnings)

    def _current_reconstruction_config(self, dataset: MicrowaveDataset):
        config = infer_reconstruction_config(dataset)
        grid_size = self._selected_grid_size()
        config.n_x = grid_size
        config.n_y = grid_size
        config.antenna_radius = self._selected_radius_m()
        config.wave_speed = self._selected_wave_speed()
        config.x_span, config.y_span = self._selected_span()
        return config

    def _update_assumption_summary(self, source_notes: dict[str, str], warnings: list[str]) -> None:
        lines = [f"{key}: {value}" for key, value in source_notes.items()]
        if warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in warnings)
        else:
            lines.append("No fallback assumptions detected.")
        self.assumptions_label.setText("\n".join(lines))

    def _selected_grid_size(self) -> int:
        text = self.grid_combo.currentText()
        try:
            return int(text.split("x", maxsplit=1)[0])
        except (TypeError, ValueError, IndexError):
            return 64

    def _selected_radius_m(self) -> float:
        text = self.radius_combo.currentText()
        return {
            "8 cm": 0.08,
            "10 cm": 0.10,
            "12 cm": 0.12,
        }.get(text, 0.08)

    def _selected_wave_speed(self) -> float:
        text = self.speed_combo.currentText()
        return {
            "3.0e8 m/s (air)": 3.0e8,
            "2.1e8 m/s": 2.1e8,
            "1.8e8 m/s": 1.8e8,
        }.get(text, 3.0e8)

    def _selected_span(self) -> tuple[tuple[float, float], tuple[float, float]]:
        text = self.span_combo.currentText()
        extent_cm = {
            "10 cm x 10 cm": 0.05,
            "12 cm x 12 cm": 0.06,
            "16 cm x 16 cm": 0.08,
        }.get(text, 0.05)
        return (-extent_cm, extent_cm), (-extent_cm, extent_cm)

    def _radius_text(self, radius_m: float) -> str:
        value_cm = int(round(radius_m * 100.0))
        mapping = {8: "8 cm", 10: "10 cm", 12: "12 cm"}
        return mapping.get(value_cm, "8 cm")

    def _speed_text(self, wave_speed: float) -> str:
        if abs(wave_speed - 2.1e8) < 1e6:
            return "2.1e8 m/s"
        if abs(wave_speed - 1.8e8) < 1e6:
            return "1.8e8 m/s"
        return "3.0e8 m/s (air)"

    def _span_text(self, x_span: tuple[float, float], y_span: tuple[float, float]) -> str:
        half_span = max(abs(float(x_span[0])), abs(float(x_span[1])), abs(float(y_span[0])), abs(float(y_span[1])))
        mapping = {
            0.05: "10 cm x 10 cm",
            0.06: "12 cm x 12 cm",
            0.08: "16 cm x 16 cm",
        }
        for key, value in mapping.items():
            if abs(half_span - key) < 1e-6:
                return value
        return "10 cm x 10 cm"

    def _configure_image_axes(
        self,
        ax,
        image: np.ndarray,
        title: str,
        x_span: tuple[float, float] | None = None,
        y_span: tuple[float, float] | None = None,
    ) -> None:
        vmin = float(np.percentile(image, 5))
        vmax = float(np.percentile(image, 99))
        if vmax <= vmin:
            vmin = float(np.min(image))
            vmax = float(np.max(image) + 1e-12)
        extent = None
        if x_span is not None and y_span is not None:
            extent = [x_span[0] * 100.0, x_span[1] * 100.0, y_span[0] * 100.0, y_span[1] * 100.0]
        im = ax.imshow(
            image,
            cmap="inferno",
            origin="lower",
            aspect="equal",
            interpolation="bicubic",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        ax.set_title(title)
        if extent is None:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlabel("x (cm)")
            ax.set_ylabel("y (cm)")
        return im

    def _plot_selected_image(
        self,
        image: np.ndarray,
        selected_name: str,
        roi_result: ROIResult,
        x_span: tuple[float, float],
        y_span: tuple[float, float],
    ) -> None:
        self.selected_figure.clear()
        ax = self.selected_figure.add_subplot(1, 1, 1)
        im = self._configure_image_axes(ax, image, f"Selected Reconstruction: {selected_name}", x_span=x_span, y_span=y_span)
        x0, y0, x1, y1 = roi_result.bounding_box
        x_axis = np.linspace(x_span[0] * 100.0, x_span[1] * 100.0, image.shape[1])
        y_axis = np.linspace(y_span[0] * 100.0, y_span[1] * 100.0, image.shape[0])
        rect = Rectangle(
            (x_axis[max(0, x0)], y_axis[max(0, y0)]),
            x_axis[min(image.shape[1] - 1, x1 - 1)] - x_axis[max(0, x0)],
            y_axis[min(image.shape[0] - 1, y1 - 1)] - y_axis[max(0, y0)],
            linewidth=2.0,
            edgecolor="cyan",
            facecolor="none",
        )
        ax.add_patch(rect)
        centroid_x = np.interp(roi_result.centroid[0], np.arange(image.shape[1]), x_axis)
        centroid_y = np.interp(roi_result.centroid[1], np.arange(image.shape[0]), y_axis)
        ax.plot(centroid_x, centroid_y, "wo", markersize=5)
        self.selected_figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.selected_figure.tight_layout()
        self.selected_canvas.draw()

    def _plot_images(
        self,
        images: dict[str, np.ndarray],
        selected_name: str,
        roi_result: ROIResult,
        x_span: tuple[float, float],
        y_span: tuple[float, float],
    ) -> None:
        self.figure.clear()
        titles = ["DAS", "DMAS", "DMAS-D4", f"Selected: {selected_name}"]
        keys = ["DAS", "DMAS", "DMAS-D4", selected_name]
        x_axis = np.linspace(x_span[0] * 100.0, x_span[1] * 100.0, images[selected_name].shape[1])
        y_axis = np.linspace(y_span[0] * 100.0, y_span[1] * 100.0, images[selected_name].shape[0])

        for idx, key in enumerate(keys, start=1):
            ax = self.figure.add_subplot(2, 2, idx)
            im = self._configure_image_axes(ax, images[key], titles[idx - 1], x_span=x_span, y_span=y_span)
            if key == selected_name:
                x0, y0, x1, y1 = roi_result.bounding_box
                rect = Rectangle(
                    (x_axis[max(0, x0)], y_axis[max(0, y0)]),
                    x_axis[min(images[key].shape[1] - 1, x1 - 1)] - x_axis[max(0, x0)],
                    y_axis[min(images[key].shape[0] - 1, y1 - 1)] - y_axis[max(0, y0)],
                    linewidth=2.0,
                    edgecolor="cyan",
                    facecolor="none",
                )
                ax.add_patch(rect)
                centroid_x = np.interp(roi_result.centroid[0], np.arange(images[key].shape[1]), x_axis)
                centroid_y = np.interp(roi_result.centroid[1], np.arange(images[key].shape[0]), y_axis)
                ax.plot(centroid_x, centroid_y, "wo", markersize=4)
            self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_refined_image(self, refinement: ROIRefinement) -> None:
        self.refined_figure.clear()
        ax = self.refined_figure.add_subplot(1, 1, 1)
        im = self._configure_image_axes(
            ax,
            refinement.image,
            f"{refinement.algorithm} ROI Refinement",
            x_span=refinement.x_span,
            y_span=refinement.y_span,
        )
        self.refined_figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.refined_figure.tight_layout()
        self.refined_canvas.draw()

    def _populate_summary(self, images: dict[str, np.ndarray], selected_name: str) -> None:
        selected = images[selected_name]
        mean_val = float(np.mean(selected))
        std_val = float(np.std(selected))
        peak_val = float(np.max(selected))
        contrast = float((np.max(selected) - np.min(selected)) / (np.max(selected) + np.min(selected) + 1e-12))

        info = {
            "Selected Beamformer": selected_name,
            "Mean Intensity": f"{mean_val:.5f}",
            "Std Intensity": f"{std_val:.5f}",
            "Peak Intensity": f"{peak_val:.5f}",
            "Contrast": f"{contrast:.5f}",
        }

        self.summary_table.setRowCount(len(info))
        for row, (key, value) in enumerate(info.items()):
            self.summary_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.summary_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.summary_table.resizeColumnsToContents()

    def _populate_metrics(self, quality_metrics: dict[str, dict[str, float]]) -> None:
        self.metrics_table.setRowCount(len(quality_metrics))
        for row, (name, metrics) in enumerate(quality_metrics.items()):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{metrics['snr']:.4f}"))
            self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{metrics['scr']:.4f}"))
            self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{metrics['contrast']:.4f}"))
            self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{metrics['time']:.4f}"))
            self.metrics_table.setItem(row, 5, QTableWidgetItem(f"{metrics['score']:.4f}"))
        self.metrics_table.resizeColumnsToContents()

    def _populate_roi(self, roi_result: ROIResult) -> None:
        x0, y0, x1, y1 = roi_result.bounding_box
        info = {
            "Bounding Box": f"({x0}, {y0}) - ({x1}, {y1})",
            "Centroid": f"({roi_result.centroid[0]:.2f}, {roi_result.centroid[1]:.2f})",
            "Area": str(roi_result.area),
            "Threshold": f"{roi_result.threshold:.2f}",
            "ROI Score": f"{roi_result.score:.5f}",
        }

        self.roi_table.setRowCount(len(info))
        for row, (key, value) in enumerate(info.items()):
            self.roi_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.roi_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.roi_table.resizeColumnsToContents()

    def _populate_refinement(self, refinement: ROIRefinement) -> None:
        x0, x1 = refinement.x_span
        y0, y1 = refinement.y_span
        info = {
            "Algorithm": refinement.algorithm,
            "Grid": f"{refinement.grid_shape[1]} x {refinement.grid_shape[0]}",
            "X Span (m)": f"{x0:.4f} to {x1:.4f}",
            "Y Span (m)": f"{y0:.4f} to {y1:.4f}",
            "Mean Intensity": f"{float(np.mean(refinement.image)):.5f}",
            "Peak Intensity": f"{float(np.max(refinement.image)):.5f}",
        }

        self.refinement_table.setRowCount(len(info))
        for row, (key, value) in enumerate(info.items()):
            self.refinement_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.refinement_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.refinement_table.resizeColumnsToContents()
