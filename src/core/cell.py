"""
Cell and Pin representations for VLSI design.

A Cell represents a macro-block or standard cell with physical dimensions,
placement coordinates, and connection pins. This forms the fundamental
building block of the netlist and floorplan.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np


class CellType(Enum):
    """Classification of cell types in the design."""
    HARD_MACRO = auto()     # Fixed-shape macro (e.g., SRAM, IP blocks)
    SOFT_MACRO = auto()     # Flexible-shape macro (area preserved, aspect ratio varies)
    STANDARD_CELL = auto()  # Small logic cells
    IO_PAD = auto()         # I/O pads on chip boundary
    FIXED = auto()          # Pre-placed, immovable blocks


class PinDirection(Enum):
    """Pin signal direction."""
    INPUT = auto()
    OUTPUT = auto()
    INOUT = auto()
    POWER = auto()
    GROUND = auto()


@dataclass
class Pin:
    """
    A connection point on a cell.

    Attributes:
        name: Pin identifier within the cell.
        direction: Signal direction (input/output/inout).
        offset_x: X offset from cell origin (bottom-left corner).
        offset_y: Y offset from cell origin.
        net_name: Name of the net this pin is connected to (None if unconnected).
    """
    name: str
    direction: PinDirection
    offset_x: float = 0.0
    offset_y: float = 0.0
    net_name: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        return self.net_name is not None

    def absolute_position(self, cell_x: float, cell_y: float) -> tuple[float, float]:
        """Get absolute coordinates given the parent cell's position."""
        return (cell_x + self.offset_x, cell_y + self.offset_y)


@dataclass
class Cell:
    """
    A physical block in the VLSI design.

    Coordinate system: (x, y) is the bottom-left corner of the cell.
    The cell extends to (x + width, y + height).

    Attributes:
        name: Unique cell identifier.
        width: Cell width in microns.
        height: Cell height in microns.
        cell_type: Classification of the cell.
        x: X coordinate of bottom-left corner (placement).
        y: Y coordinate of bottom-left corner (placement).
        pins: Dictionary of pins on this cell.
        fixed: Whether the cell is immovable.
        power_density: Power dissipation per unit area (W/μm²).
    """
    name: str
    width: float
    height: float
    cell_type: CellType = CellType.HARD_MACRO
    x: float = 0.0
    y: float = 0.0
    pins: dict[str, Pin] = field(default_factory=dict)
    fixed: bool = False
    power_density: float = 0.001  # W/μm² default

    # ── Geometric Properties ──────────────────────────────────────────

    @property
    def area(self) -> float:
        """Cell area in μm²."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Width-to-height ratio."""
        return self.width / self.height if self.height > 0 else float('inf')

    @property
    def center(self) -> tuple[float, float]:
        """Center coordinates of the cell."""
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box as (x_min, y_min, x_max, y_max)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def total_power(self) -> float:
        """Total power dissipation of this cell in Watts."""
        return self.power_density * self.area

    # ── Placement Methods ─────────────────────────────────────────────

    def place(self, x: float, y: float) -> None:
        """Place the cell at the given coordinates."""
        if not self.fixed:
            self.x = x
            self.y = y

    def move(self, dx: float, dy: float) -> None:
        """Move the cell by a relative offset."""
        if not self.fixed:
            self.x += dx
            self.y += dy

    def set_position_from_center(self, cx: float, cy: float) -> None:
        """Place the cell so its center is at (cx, cy)."""
        if not self.fixed:
            self.x = cx - self.width / 2.0
            self.y = cy - self.height / 2.0

    # ── Pin Methods ───────────────────────────────────────────────────

    def add_pin(self, pin: Pin) -> None:
        """Add a pin to this cell."""
        self.pins[pin.name] = pin

    def get_pin_position(self, pin_name: str) -> tuple[float, float]:
        """Get absolute position of a named pin."""
        pin = self.pins[pin_name]
        return pin.absolute_position(self.x, self.y)

    # ── Overlap Detection ─────────────────────────────────────────────

    def overlaps(self, other: Cell) -> bool:
        """Check if this cell overlaps with another cell."""
        x1_min, y1_min, x1_max, y1_max = self.bbox
        x2_min, y2_min, x2_max, y2_max = other.bbox
        return not (x1_max <= x2_min or x2_max <= x1_min or
                    y1_max <= y2_min or y2_max <= y1_min)

    def overlap_area(self, other: Cell) -> float:
        """Calculate the overlapping area with another cell."""
        x1_min, y1_min, x1_max, y1_max = self.bbox
        x2_min, y2_min, x2_max, y2_max = other.bbox

        dx = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        dy = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        return dx * dy

    def distance_to(self, other: Cell) -> float:
        """Euclidean distance between cell centers."""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    # ── Soft Macro Reshape ────────────────────────────────────────────

    def reshape(self, new_aspect_ratio: float) -> None:
        """
        Reshape a soft macro to a new aspect ratio while preserving area.
        Only works for SOFT_MACRO type cells.
        """
        if self.cell_type != CellType.SOFT_MACRO:
            return
        area = self.area
        self.width = np.sqrt(area * new_aspect_ratio)
        self.height = area / self.width

    def __repr__(self) -> str:
        return (f"Cell('{self.name}', {self.width:.1f}×{self.height:.1f}, "
                f"pos=({self.x:.1f},{self.y:.1f}), type={self.cell_type.name})")
