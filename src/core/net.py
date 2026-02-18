"""
Net representation for VLSI design.

A Net represents an electrical connection between pins across multiple cells.
It is used for wirelength estimation, connectivity analysis, and routing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class NetPin:
    """
    Reference to a specific pin on a specific cell within a net.

    Attributes:
        cell_name: Name of the parent cell.
        pin_name: Name of the pin on the cell.
    """
    cell_name: str
    pin_name: str

    def __repr__(self) -> str:
        return f"{self.cell_name}/{self.pin_name}"


@dataclass
class Net:
    """
    An electrical net connecting multiple pins.

    Used for wirelength estimation and routing. Supports HPWL (Half-Perimeter
    Wirelength) and Star model wirelength estimation.

    Attributes:
        name: Unique net identifier.
        pins: List of (cell_name, pin_name) references.
        weight: Net weight for cost function (critical nets get higher weight).
        clock: Whether this is a clock net.
    """
    name: str
    pins: list[NetPin] = field(default_factory=list)
    weight: float = 1.0
    clock: bool = False

    @property
    def degree(self) -> int:
        """Number of pins (fanout + 1) on this net."""
        return len(self.pins)

    @property
    def is_two_pin(self) -> bool:
        """Whether this is a simple two-pin net."""
        return len(self.pins) == 2

    def add_pin(self, cell_name: str, pin_name: str) -> None:
        """Add a pin to this net."""
        self.pins.append(NetPin(cell_name, pin_name))

    def get_cell_names(self) -> set[str]:
        """Get the set of unique cell names connected by this net."""
        return {pin.cell_name for pin in self.pins}

    # ── Wirelength Estimation ─────────────────────────────────────────

    def hpwl(self, cell_positions: dict[str, tuple[float, float]]) -> float:
        """
        Half-Perimeter Wirelength (HPWL) estimation.

        HPWL is the industry-standard wirelength proxy. It computes the
        half-perimeter of the bounding box enclosing all pin positions.

        Args:
            cell_positions: Dict mapping cell names to (center_x, center_y).

        Returns:
            HPWL value for this net.
        """
        if len(self.pins) < 2:
            return 0.0

        xs, ys = [], []
        for pin in self.pins:
            if pin.cell_name in cell_positions:
                cx, cy = cell_positions[pin.cell_name]
                xs.append(cx)
                ys.append(cy)

        if len(xs) < 2:
            return 0.0

        hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
        return hpwl * self.weight

    def star_wirelength(self, cell_positions: dict[str, tuple[float, float]]) -> float:
        """
        Star model wirelength estimation.

        Connects all pins to the centroid of the net. Often more accurate
        than HPWL for high-fanout nets.

        Args:
            cell_positions: Dict mapping cell names to (center_x, center_y).

        Returns:
            Star model wirelength for this net.
        """
        if len(self.pins) < 2:
            return 0.0

        positions = []
        for pin in self.pins:
            if pin.cell_name in cell_positions:
                positions.append(cell_positions[pin.cell_name])

        if len(positions) < 2:
            return 0.0

        positions_arr = np.array(positions)
        centroid = positions_arr.mean(axis=0)
        total = np.sum(np.abs(positions_arr - centroid))
        return float(total) * self.weight

    def bounding_box(self, cell_positions: dict[str, tuple[float, float]]) -> Optional[tuple[float, float, float, float]]:
        """
        Get bounding box of the net.

        Returns:
            (x_min, y_min, x_max, y_max) or None if insufficient pins.
        """
        xs, ys = [], []
        for pin in self.pins:
            if pin.cell_name in cell_positions:
                cx, cy = cell_positions[pin.cell_name]
                xs.append(cx)
                ys.append(cy)

        if len(xs) < 1:
            return None

        return (min(xs), min(ys), max(xs), max(ys))

    def __repr__(self) -> str:
        pin_str = ", ".join(str(p) for p in self.pins[:4])
        if len(self.pins) > 4:
            pin_str += f", ... (+{len(self.pins) - 4} more)"
        return f"Net('{self.name}', [{pin_str}])"
