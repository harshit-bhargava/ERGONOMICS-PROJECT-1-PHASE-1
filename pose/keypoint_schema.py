"""50-point hand anthropometric keypoint schema and measurement topology."""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class KeypointData:
    """Represents a 3D coordinate landmark."""
    id: int
    name: str
    x: float
    y: float
    z: float
    visibility: float = 1.0


# Longitudinal vectors L1 - L13 (Fig. b)
LONGITUDINAL_SEGMENTS: Dict[str, Tuple[int, int]] = {
    "L1": (1, 29),
    "L2": (4, 31),
    "L3": (7, 33),
    "L4": (10, 35),
    "L5": (14, 37),
    "L6": (50, 16),
    "L7": (50, 17),
    "L8": (50, 19),
    "L9": (50, 21),
    "L10": (50, 28),
    "L11": (50, 23),
    "L12": (50, 21),
    "L13": (50, 18),
}

# Transverse widths H1 - H10 (Fig. c)
TRANSVERSE_WIDTHS: Dict[str, Tuple[int, int]] = {
    "H1": (1, 3),
    "H2": (4, 6),
    "H3": (7, 9),
    "H4": (10, 12),
    "H5": (13, 15),
    "H6": (16, 17),
    "H7": (17, 19),
    "H8": (19, 21),
    "H9": (21, 24),
    "H10": (26, 28),
}

# Polygon surface regions S1 - S8 (Fig. d)
SURFACE_POLYGONS: Dict[str, List[int]] = {
    "S1": [1, 29, 3, 18],
    "S2": [4, 31, 6, 20],
    "S3": [7, 33, 9, 22],
    "S4": [10, 35, 12, 23],
    "S5": [13, 37, 15, 25],
    "S6": [23, 24, 45, 46, 25, 49, 50],
    "S7": [16, 17, 40, 18, 50],
    "S8": [26, 27, 28, 48, 47],
}