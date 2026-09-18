"""Save GUI-style reconstruction figures for automation (no PySide required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def _extent(x_span: tuple[float, float], y_span: tuple[float, float]) -> list[float]:
    return [x_span[0] * 100.0, x_span[1] * 100.0, y_span[0] * 100.0, y_span[1] * 100.0]


def _draw_roi_boxes(
    ax,
    boxes: list[tuple[int, int, int, int]],
    nx: int,
    ny: int,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
) -> None:
    colors = ("cyan", "yellow", "magenta", "white")
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        width = (x1 - x0) / max(nx - 1, 1) * (x_span[1] - x_span[0]) * 100.0
        height = (y1 - y0) / max(ny - 1, 1) * (y_span[1] - y_span[0]) * 100.0
        left = x_span[0] * 100.0 + (x0 / max(nx - 1, 1)) * (x_span[1] - x_span[0]) * 100.0
        bottom = y_span[0] * 100.0 + (y0 / max(ny - 1, 1)) * (y_span[1] - y_span[0]) * 100.0
        ax.add_patch(
            Rectangle(
                (left, bottom),
                width,
                height,
                fill=False,
                edgecolor=colors[idx % len(colors)],
                linewidth=1.4,
            )
        )


def write_case_figures(
    output_dir: str | Path,
    stem: str,
    *,
    images: dict[str, np.ndarray],
    selected: str,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    roi_boxes: list[tuple[int, int, int, int]] | None = None,
    roi_box: tuple[int, int, int, int] | None = None,
    refined_image: np.ndarray | None = None,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    extent = _extent(x_span, y_span)
    paths: dict[str, str] = {}
    boxes = list(roi_boxes or [])
    if not boxes and roi_box is not None:
        boxes = [roi_box]

    selected_img = np.asarray(images[selected], dtype=float)
    ny, nx = selected_img.shape
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(selected_img, origin="lower", extent=extent, cmap="inferno")
    _draw_roi_boxes(ax, boxes, nx, ny, x_span, y_span)
    ax.set_title(f"Selected ({selected})")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    selected_path = out / f"{stem}_selected.png"
    fig.savefig(selected_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths["selected"] = str(selected_path)

    names = [name for name in ("DAS", "DMAS", "DMAS-D4") if name in images]
    fig, axes = plt.subplots(1, max(len(names), 1), figsize=(3.4 * max(len(names), 1), 3.4))
    if len(names) <= 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        im = ax.imshow(np.asarray(images[name], dtype=float), origin="lower", extent=extent, cmap="inferno")
        ax.set_title(name)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Beamformer comparison")
    fig.tight_layout()
    beam_path = out / f"{stem}_beamformers.png"
    fig.savefig(beam_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths["beamformers"] = str(beam_path)

    if refined_image is not None:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(np.asarray(refined_image, dtype=float), origin="lower", cmap="inferno")
        ax.set_title("ROI refine")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        refine_path = out / f"{stem}_roi_refine.png"
        fig.savefig(refine_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths["roi_refine"] = str(refine_path)

    return paths
