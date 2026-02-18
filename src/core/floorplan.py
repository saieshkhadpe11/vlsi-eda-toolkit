"""
Floorplan: snapshot of a placement solution.

A Floorplan captures the state of all cell placements within a chip outline,
and provides metrics (area utilization, overlap, wirelength) for evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .cell import Cell
from .netlist import Netlist


@dataclass
class Floorplan:
    """
    Represents a placement solution within a chip outline.

    Attributes:
        chip_width: Total chip width (die boundary).
        chip_height: Total chip height (die boundary).
        netlist: Reference to the underlying netlist.
    """
    chip_width: float
    chip_height: float
    netlist: Optional[Netlist] = None

    # ── Chip Properties ───────────────────────────────────────────────

    @property
    def chip_area(self) -> float:
        return self.chip_width * self.chip_height

    @property
    def utilization(self) -> float:
        """Area utilization ratio (0 to 1)."""
        if self.netlist is None or self.chip_area == 0:
            return 0.0
        return self.netlist.total_cell_area / self.chip_area

    # ── Overlap Metrics ───────────────────────────────────────────────

    def total_overlap_area(self) -> float:
        """Total pairwise overlap area between all cells."""
        if self.netlist is None:
            return 0.0
        cells = list(self.netlist.cells.values())
        total = 0.0
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                total += cells[i].overlap_area(cells[j])
        return total

    def has_overlaps(self) -> bool:
        """Check if any cells overlap."""
        return self.total_overlap_area() > 0

    def overlap_pairs(self) -> list[tuple[str, str, float]]:
        """Get all overlapping cell pairs with their overlap areas."""
        if self.netlist is None:
            return []
        cells = list(self.netlist.cells.values())
        pairs = []
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                oa = cells[i].overlap_area(cells[j])
                if oa > 0:
                    pairs.append((cells[i].name, cells[j].name, oa))
        return pairs

    # ── Boundary Violations ───────────────────────────────────────────

    def boundary_violations(self) -> list[tuple[str, float]]:
        """
        Find cells that extend beyond the chip boundary.

        Returns:
            List of (cell_name, violation_area) tuples.
        """
        if self.netlist is None:
            return []

        violations = []
        for cell in self.netlist.cells.values():
            x_min, y_min, x_max, y_max = cell.bbox

            # Compute how far out of bounds in each direction
            left = max(0, -x_min)
            bottom = max(0, -y_min)
            right = max(0, x_max - self.chip_width)
            top = max(0, y_max - self.chip_height)

            violation = (left + right) * cell.height + (bottom + top) * cell.width
            if violation > 0:
                violations.append((cell.name, violation))

        return violations

    def all_within_bounds(self) -> bool:
        """Check if all cells are within the chip boundary."""
        return len(self.boundary_violations()) == 0

    # ── Wirelength ────────────────────────────────────────────────────

    def total_hpwl(self) -> float:
        """Total HPWL of the current placement."""
        if self.netlist is None:
            return 0.0
        return self.netlist.total_hpwl()

    # ── Quality Score ─────────────────────────────────────────────────

    def quality_score(self, w_wl: float = 0.4, w_area: float = 0.3,
                      w_overlap: float = 0.3) -> float:
        """
        Composite quality score (lower is better).

        Combines normalized wirelength, area utilization penalty,
        and overlap penalty into a single score.
        """
        wl = self.total_hpwl()
        overlap = self.total_overlap_area()
        area_penalty = max(0, 1.0 - self.utilization)  # penalize underutilization

        # Normalize by chip dimensions
        wl_norm = wl / (self.chip_width + self.chip_height) if (self.chip_width + self.chip_height) > 0 else 0
        overlap_norm = overlap / self.chip_area if self.chip_area > 0 else 0

        return w_wl * wl_norm + w_area * area_penalty + w_overlap * overlap_norm

    # ── Summary ───────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of floorplan quality."""
        overlaps = self.total_overlap_area()
        violations = self.boundary_violations()
        lines = [
            f"┌──────────────────────────────────────────┐",
            f"│  Floorplan Summary                       │",
            f"├──────────────────────────────────────────┤",
            f"│  Chip Size:     {self.chip_width:.1f} × {self.chip_height:.1f}" + " " * max(0, 18 - len(f"{self.chip_width:.1f} × {self.chip_height:.1f}")) + "│",
            f"│  Utilization:   {self.utilization * 100:.1f}%" + " " * max(0, 22 - len(f"{self.utilization * 100:.1f}%")) + "│",
            f"│  Total HPWL:    {self.total_hpwl():.1f}" + " " * max(0, 22 - len(f"{self.total_hpwl():.1f}")) + "│",
            f"│  Overlap Area:  {overlaps:.1f}" + " " * max(0, 22 - len(f"{overlaps:.1f}")) + "│",
            f"│  Violations:    {len(violations)}" + " " * max(0, 22 - len(f"{len(violations)}")) + "│",
            f"│  Quality Score: {self.quality_score():.4f}" + " " * max(0, 22 - len(f"{self.quality_score():.4f}")) + "│",
            f"└──────────────────────────────────────────┘",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"Floorplan({self.chip_width:.0f}×{self.chip_height:.0f}, "
                f"util={self.utilization:.1%})")
