"""High-level reconstruction manager for multiple beamforming algorithms."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from reconstruction.das import das_from_frequency
from reconstruction.dmas import dmas_reconstruct
from reconstruction.dmas_d4 import dmas_d4_reconstruct
from reconstruction.ifft import frequency_to_time


@dataclass
class ROIRefinement:
    """High-resolution reconstruction result for a coarse ROI."""

    algorithm: str
    image: np.ndarray
    x_span: tuple[float, float]
    y_span: tuple[float, float]
    grid_shape: tuple[int, int]


@dataclass
class ReconstructionConfig:
    """Physical and numerical settings for beamforming."""

    x_span: tuple[float, float] = (-0.05, 0.05)
    y_span: tuple[float, float] = (-0.05, 0.05)
    n_x: int = 64
    n_y: int = 64
    antenna_radius: float = 0.08
    wave_speed: float = 3e8
    zero_padding: int = 0
    # Geometry convention knobs (BMID / array alignment).
    antenna_angle_offset_deg: float = 0.0
    antenna_clockwise: bool = False
    antenna_flip_x: bool = False
    antenna_flip_y: bool = False
    antenna_span_deg: float = 360.0
    use_bmid_phase_delay_radius: bool = False


@dataclass
class ReconstructionAssessment:
    """Reconstruction config plus human-readable assumptions."""

    config: ReconstructionConfig
    source_notes: dict[str, str]
    warnings: list[str]


def bmid_phase_delayed_radius(scan_rad_m: float) -> float:
    """Effective antenna radius accounting for UM-BMID antenna phase delay.

    Empirical model from Rodriguez-Herrera (UManitoba M.Sc. thesis), also used
    in the UM-BMID ``umbmid.antennas.get_phase_delay_rad`` helper:

        delayed_rad = 0.97 * (scan_rad - 0.106) + 0.148
    """
    scan_rad_m = float(scan_rad_m)
    return float(0.97 * (scan_rad_m - 0.106) + 0.148)


def effective_antenna_radius(config: ReconstructionConfig) -> float:
    """Return the radius used for array placement (optionally phase-delayed)."""
    radius = float(config.antenna_radius)
    if config.use_bmid_phase_delay_radius:
        return bmid_phase_delayed_radius(radius)
    return radius


def build_circular_antenna_array(
    n_antennas: int,
    radius: float = 0.08,
    angle_offset_deg: float = 0.0,
    clockwise: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
    span_deg: float = 360.0,
) -> np.ndarray:
    """Create a circular antenna array in the x-y plane.

    Args:
        n_antennas: Number of antenna positions.
        radius: Array radius in meters.
        angle_offset_deg: Rotation of the first antenna from +x (degrees).
        clockwise: If True, advance angles clockwise instead of CCW.
        flip_x: Mirror x coordinates.
        flip_y: Mirror y coordinates.
        span_deg: Angular arc covered by the array (BMID uses ~355°).
    """
    if n_antennas < 1:
        raise ValueError("n_antennas must be >= 1.")

    span_rad = float(np.deg2rad(span_deg))
    if abs(span_deg - 360.0) < 1e-6:
        angles = np.linspace(0.0, 2.0 * np.pi, n_antennas, endpoint=False)
    else:
        angles = np.linspace(0.0, span_rad, n_antennas, endpoint=True)

    if clockwise:
        angles = -angles
    angles = angles + np.deg2rad(float(angle_offset_deg))

    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    if flip_x:
        x = -x
    if flip_y:
        y = -y
    return np.stack([x, y], axis=-1)


def antenna_positions_from_config(n_antennas: int, config: ReconstructionConfig) -> np.ndarray:
    """Build antenna positions from a ReconstructionConfig."""
    return build_circular_antenna_array(
        n_antennas,
        radius=effective_antenna_radius(config),
        angle_offset_deg=config.antenna_angle_offset_deg,
        clockwise=config.antenna_clockwise,
        flip_x=config.antenna_flip_x,
        flip_y=config.antenna_flip_y,
        span_deg=config.antenna_span_deg,
    )


def _build_grid(x_span: tuple[float, float], y_span: tuple[float, float], n_x: int = 64, n_y: int = 64) -> tuple[np.ndarray, np.ndarray]:
    grid_x = np.linspace(x_span[0], x_span[1], n_x)
    grid_y = np.linspace(y_span[0], y_span[1], n_y)
    return grid_x, grid_y


def _reconstruct_single(
    algorithm: str,
    s_parameters: np.ndarray,
    frequencies: np.ndarray,
    antenna_positions: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    zero_padding: int = 0,
    wave_speed: float = 3e8,
) -> np.ndarray:
    if algorithm == "DAS":
        return das_from_frequency(
            s_parameters,
            frequencies,
            antenna_positions,
            grid_x,
            grid_y,
            zero_padding=zero_padding,
            wave_speed=wave_speed,
        )

    time_signals = frequency_to_time(s_parameters, frequencies, zero_padding=zero_padding)
    if algorithm == "DMAS":
        return dmas_reconstruct(time_signals, frequencies, antenna_positions, grid_x, grid_y, wave_speed=wave_speed)
    if algorithm == "DMAS-D4":
        return dmas_d4_reconstruct(time_signals, frequencies, antenna_positions, grid_x, grid_y, wave_speed=wave_speed)

    raise ValueError(f"Unsupported beamformer '{algorithm}'.")


def _pixel_bbox_to_span(
    start_idx: int,
    stop_idx: int,
    axis_span: tuple[float, float],
    axis_size: int,
) -> tuple[float, float]:
    if axis_size <= 1:
        return axis_span

    axis = np.linspace(axis_span[0], axis_span[1], axis_size)
    start_idx = int(np.clip(start_idx, 0, axis_size - 1))
    stop_idx = int(np.clip(max(start_idx + 1, stop_idx) - 1, 0, axis_size - 1))
    span_start = float(axis[start_idx])
    span_stop = float(axis[stop_idx])

    if span_start == span_stop:
        step = float(abs(axis[1] - axis[0]))
        span_stop = span_start + step

    return (min(span_start, span_stop), max(span_start, span_stop))


def reconstruct_all(
    s_parameters: np.ndarray,
    frequencies: np.ndarray,
    antenna_positions: np.ndarray | None = None,
    x_span: tuple[float, float] = (-0.05, 0.05),
    y_span: tuple[float, float] = (-0.05, 0.05),
    n_x: int = 64,
    n_y: int = 64,
    zero_padding: int = 0,
    return_timings: bool = False,
    wave_speed: float = 3e8,
    antenna_radius: float = 0.08,
    config: ReconstructionConfig | None = None,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, float]]:
    """Run all supported reconstructions and return image maps.

    Args:
        s_parameters: Frequency-domain data, shape (n_freq, n_traces).
        frequencies: Frequency vector in Hz.
        antenna_positions: Optional antenna layout. If None, a circular array is used.
        x_span: X-axis extent of the reconstruction grid in meters.
        y_span: Y-axis extent of the reconstruction grid in meters.
        n_x: Number of x grid points.
        n_y: Number of y grid points.
        zero_padding: Optional zero padding for the IFFT.
        return_timings: If True, also return algorithm execution timings.
        config: Optional full config; when set, overrides span/grid/speed/radius/geometry.

    Returns:
        A dictionary of images for DAS, DMAS, and DMAS-D4, and optionally timings.
    """
    if s_parameters.ndim != 2:
        raise ValueError("s_parameters must be 2D (frequency, traces).")

    if config is not None:
        x_span = config.x_span
        y_span = config.y_span
        n_x = config.n_x
        n_y = config.n_y
        zero_padding = config.zero_padding
        wave_speed = config.wave_speed
        antenna_radius = config.antenna_radius

    n_traces = s_parameters.shape[1]
    if antenna_positions is None:
        if config is not None:
            antenna_positions = antenna_positions_from_config(n_traces, config)
        else:
            antenna_positions = build_circular_antenna_array(n_traces, radius=antenna_radius)
    grid_x, grid_y = _build_grid(x_span, y_span, n_x=n_x, n_y=n_y)
    time_signals = frequency_to_time(s_parameters, frequencies, zero_padding=zero_padding)

    images: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    start = time.perf_counter()
    images["DAS"] = das_from_frequency(
        s_parameters,
        frequencies,
        antenna_positions,
        grid_x,
        grid_y,
        zero_padding=zero_padding,
        wave_speed=wave_speed,
    )
    timings["DAS"] = time.perf_counter() - start

    start = time.perf_counter()
    images["DMAS"] = dmas_reconstruct(time_signals, frequencies, antenna_positions, grid_x, grid_y, wave_speed=wave_speed)
    timings["DMAS"] = time.perf_counter() - start

    start = time.perf_counter()
    images["DMAS-D4"] = dmas_d4_reconstruct(time_signals, frequencies, antenna_positions, grid_x, grid_y, wave_speed=wave_speed)
    timings["DMAS-D4"] = time.perf_counter() - start

    if return_timings:
        return images, timings
    return images


def reconstruct_high_resolution_roi(
    s_parameters: np.ndarray,
    frequencies: np.ndarray,
    algorithm: str,
    bounding_box: tuple[int, int, int, int],
    coarse_shape: tuple[int, int],
    antenna_positions: np.ndarray | None = None,
    full_x_span: tuple[float, float] = (-0.05, 0.05),
    full_y_span: tuple[float, float] = (-0.05, 0.05),
    upscale_factor: int = 4,
    min_grid_size: int = 96,
    max_grid_size: int = 256,
    zero_padding: int = 0,
    wave_speed: float = 3e8,
    antenna_radius: float = 0.08,
    config: ReconstructionConfig | None = None,
) -> ROIRefinement:
    """Reconstruct a denser image limited to a previously detected coarse ROI."""
    if len(coarse_shape) != 2:
        raise ValueError("coarse_shape must be a 2D image shape.")

    if config is not None:
        full_x_span = config.x_span
        full_y_span = config.y_span
        zero_padding = config.zero_padding
        wave_speed = config.wave_speed
        antenna_radius = config.antenna_radius

    coarse_ny, coarse_nx = coarse_shape
    x0, y0, x1, y1 = bounding_box
    roi_width = max(1, int(x1 - x0))
    roi_height = max(1, int(y1 - y0))
    refined_nx = min(max(min_grid_size, roi_width * upscale_factor), max_grid_size)
    refined_ny = min(max(min_grid_size, roi_height * upscale_factor), max_grid_size)

    x_span = _pixel_bbox_to_span(x0, x1, full_x_span, coarse_nx)
    y_span = _pixel_bbox_to_span(y0, y1, full_y_span, coarse_ny)

    n_traces = s_parameters.shape[1]
    if antenna_positions is None:
        if config is not None:
            antenna_positions = antenna_positions_from_config(n_traces, config)
        else:
            antenna_positions = build_circular_antenna_array(n_traces, radius=antenna_radius)
    grid_x, grid_y = _build_grid(x_span, y_span, n_x=refined_nx, n_y=refined_ny)
    refined_image = _reconstruct_single(
        algorithm,
        s_parameters,
        frequencies,
        antenna_positions,
        grid_x,
        grid_y,
        zero_padding=zero_padding,
        wave_speed=wave_speed,
    )

    return ROIRefinement(
        algorithm=algorithm,
        image=refined_image,
        x_span=x_span,
        y_span=y_span,
        grid_shape=(refined_ny, refined_nx),
    )


def infer_reconstruction_config(dataset) -> ReconstructionConfig:
    """Build a reconstruction config from dataset metadata when available."""
    return infer_reconstruction_assessment(dataset).config


def infer_reconstruction_assessment(dataset) -> ReconstructionAssessment:
    """Build a reconstruction config together with assumption notes."""
    metadata = getattr(dataset, "metadata", {}) or {}

    def _extract_span(key: str, default: tuple[float, float]) -> tuple[tuple[float, float], str, bool]:
        raw_span = metadata.get(key, default)
        span = (float(raw_span[0]), float(raw_span[1]))
        used_default = key not in metadata
        source = "metadata" if not used_default else "default fallback"
        return span, source, used_default

    x_span, x_source, x_default = _extract_span("reconstruction_x_span_m", (-0.05, 0.05))
    y_span, y_source, y_default = _extract_span("reconstruction_y_span_m", (-0.05, 0.05))
    wave_speed = float(metadata.get("wave_speed_m_per_s", 3e8))
    antenna_radius = float(metadata.get("antenna_radius_m", 0.08))
    wave_speed_source = "metadata" if "wave_speed_m_per_s" in metadata else "default fallback"
    antenna_radius_source = "metadata" if "antenna_radius_m" in metadata else "default fallback"
    is_bmid = str(metadata.get("dataset_family", "")).upper() == "UM-BMID"

    warnings: list[str] = []
    if x_default or y_default:
        warnings.append("Reconstruction field of view uses default bounds because the dataset did not provide imaging spans.")
    if wave_speed_source == "default fallback":
        warnings.append("Wave speed uses the default value; verify it against the tissue / coupling medium.")
    if antenna_radius_source == "default fallback":
        warnings.append("Antenna radius uses the default value; verify it against the physical array geometry.")

    config = ReconstructionConfig(
        x_span=x_span,
        y_span=y_span,
        antenna_radius=antenna_radius,
        wave_speed=wave_speed,
    )
    source_notes = {
        "Field of View": f"x: {x_source}, y: {y_source}",
        "Wave Speed": wave_speed_source,
        "Antenna Radius": antenna_radius_source,
    }
    if is_bmid:
        # Keep conservative defaults; use Auto Tweak to search BMID geometry options.
        config.antenna_span_deg = 360.0
        config.use_bmid_phase_delay_radius = False
        source_notes["Antenna Span"] = "360° (use Auto Tweak to try 355°)"
        source_notes["Phase-delay Radius"] = "off (use Auto Tweak to compare)"
        warnings.append(
            "UM-BMID loaded: start from phase-delay off / 360° arc. "
            "Click Auto Tweak Settings to search angle, flip, arc, and beamformer."
        )

    assessment = ReconstructionAssessment(
        config=config,
        source_notes=source_notes,
        warnings=warnings,
    )
    return assessment


def select_best_reconstruction(
    images: dict[str, np.ndarray], weights: dict[str, float] | None = None
) -> tuple[str, np.ndarray]:
    """Select the best reconstruction image using a simple weighted quality score."""
    scores: dict[str, float] = {}
    weights = weights or {"contrast": 0.4, "snr": 0.3, "peak": 0.3}

    for name, image in images.items():
        mean = np.mean(image)
        std = np.std(image)
        snr = std / (mean + 1e-12)
        contrast = (np.max(image) - np.min(image)) / (np.max(image) + np.min(image) + 1e-12)
        peak = np.max(image)
        scores[name] = weights["contrast"] * contrast + weights["snr"] * snr + weights["peak"] * peak

    selected = max(scores, key=scores.get)
    return selected, images[selected]
