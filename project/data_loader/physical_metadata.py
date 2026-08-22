"""Helpers for extracting reconstruction-related physical metadata from text."""

from __future__ import annotations

import re


_PATTERNS = {
    "antenna_radius_m": [
        re.compile(r"antenna\s*radius\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(m|cm)?", re.IGNORECASE),
        re.compile(r"radius\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(m|cm)?", re.IGNORECASE),
    ],
    "wave_speed_m_per_s": [
        re.compile(r"wave\s*speed\s*[:=]\s*([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)\s*(m/s|mps|m\s*/\s*s)?", re.IGNORECASE),
        re.compile(r"speed\s*[:=]\s*([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)\s*(m/s|mps|m\s*/\s*s)?", re.IGNORECASE),
    ],
    "reconstruction_x_span_m": [
        re.compile(r"x\s*span\s*[:=]\s*([-0-9]*\.?[0-9]+)\s*(m|cm)?\s*(?:to|[-,])\s*([-0-9]*\.?[0-9]+)\s*(m|cm)?", re.IGNORECASE),
    ],
    "reconstruction_y_span_m": [
        re.compile(r"y\s*span\s*[:=]\s*([-0-9]*\.?[0-9]+)\s*(m|cm)?\s*(?:to|[-,])\s*([-0-9]*\.?[0-9]+)\s*(m|cm)?", re.IGNORECASE),
    ],
}


def _to_meters(value: float, unit: str | None) -> float:
    if unit and unit.lower() == "cm":
        return value / 100.0
    return value


def extract_physical_metadata_from_text(text: str | None) -> dict:
    """Parse reconstruction-related metadata from a free-form text blob."""
    if not text:
        return {}

    extracted: dict = {}
    for key, patterns in _PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(text)
            if not match:
                continue
            if key in ("antenna_radius_m", "wave_speed_m_per_s"):
                value = float(match.group(1))
                unit = match.group(2)
                if key == "antenna_radius_m":
                    extracted[key] = _to_meters(value, unit)
                else:
                    extracted[key] = value
                break
            if key in ("reconstruction_x_span_m", "reconstruction_y_span_m"):
                left = _to_meters(float(match.group(1)), match.group(2))
                right = _to_meters(float(match.group(3)), match.group(4))
                extracted[key] = (left, right)
                break
    return extracted


def extract_physical_metadata_from_values(values: dict) -> dict:
    """Scan dictionary values for strings that might encode physical metadata."""
    combined: dict = {}
    for value in values.values():
        if isinstance(value, str):
            combined.update(extract_physical_metadata_from_text(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    combined.update(extract_physical_metadata_from_text(item))
    return combined