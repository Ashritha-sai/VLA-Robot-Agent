"""
Shared utility functions for the VLA Robot Agent.
"""

import numpy as np
from typing import Dict, List, Tuple


def normalize_direction(direction: List[float]) -> np.ndarray:
    """Normalize a 2D or 3D direction vector to unit length.

    Args:
        direction: Direction vector (2D or 3D).

    Returns:
        Unit-length numpy array of the same dimensionality.

    Raises:
        ValueError: If the vector has zero length.
    """
    d = np.array(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        raise ValueError("Cannot normalize a zero-length direction vector.")
    return d / norm


def position_on_table(
    x: float,
    y: float,
    table_height: float = 0.05,
    object_half_height: float = 0.02,
) -> List[float]:
    """Return an [x, y, z] position so the object sits on the table surface.

    Args:
        x: X coordinate.
        y: Y coordinate.
        table_height: Z of the table top surface (default 0.05).
        object_half_height: Half-height of the object (default 0.02 for a 0.04 m cube).

    Returns:
        [x, y, z] list with z set so the object rests on the table.
    """
    return [x, y, table_height + object_half_height]


def objects_within_distance(
    objects: Dict[str, np.ndarray],
    reference: np.ndarray,
    max_distance: float,
) -> List[Tuple[str, float]]:
    """Find all objects within *max_distance* of a reference point.

    Args:
        objects: Mapping of object name to (3,) position array.
        reference: (3,) reference position.
        max_distance: Maximum Euclidean distance threshold.

    Returns:
        List of (name, distance) tuples sorted by distance (nearest first).
    """
    ref = np.array(reference, dtype=float)
    results = []
    for name, pos in objects.items():
        dist = float(np.linalg.norm(np.array(pos, dtype=float) - ref))
        if dist <= max_distance:
            results.append((name, dist))
    results.sort(key=lambda t: t[1])
    return results


def clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* to the range [low, high].

    Args:
        value: The value to clamp.
        low: Lower bound.
        high: Upper bound.

    Returns:
        Clamped value.
    """
    return max(low, min(value, high))
