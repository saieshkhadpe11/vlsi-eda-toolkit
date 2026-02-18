"""
Cost functions for floorplanning optimization.

Provides configurable, weighted cost functions that combine:
  - Wirelength (HPWL)
  - Area utilization
  - Overlap penalty
  - Thermal penalty (hotspot proximity)
  - Boundary constraint violations
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from core.netlist import Netlist
from core.floorplan import Floorplan


@dataclass
class CostWeights:
    """Weights for the multi-objective cost function."""
    wirelength: float = 0.40
    overlap: float = 0.30
    boundary: float = 0.15
    thermal: float = 0.10
    aspect_ratio: float = 0.05

    def normalize(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = (self.wirelength + self.overlap + self.boundary +
                 self.thermal + self.aspect_ratio)
        if total > 0:
            self.wirelength /= total
            self.overlap /= total
            self.boundary /= total
            self.thermal /= total
            self.aspect_ratio /= total


class CostFunction:
    """
    Multi-objective cost function for floorplan evaluation.

    Combines multiple design objectives into a single scalar cost for
    optimization. Each component is normalized relative to the chip
    dimensions for fair weighting.
    """

    def __init__(self, floorplan: Floorplan,
                 weights: CostWeights | None = None):
        self.floorplan = floorplan
        self.weights = weights or CostWeights()
        self.weights.normalize()

        # Normalization factors
        self._chip_diag = np.sqrt(
            floorplan.chip_width ** 2 + floorplan.chip_height ** 2
        )
        self._chip_area = floorplan.chip_area

    def evaluate(self) -> float:
        """
        Compute the total cost of the current floorplan.

        Returns:
            Scalar cost value (lower is better).
        """
        cost = 0.0
        cost += self.weights.wirelength * self._wirelength_cost()
        cost += self.weights.overlap * self._overlap_cost()
        cost += self.weights.boundary * self._boundary_cost()
        cost += self.weights.thermal * self._thermal_cost()
        cost += self.weights.aspect_ratio * self._aspect_ratio_cost()
        return cost

    def evaluate_detailed(self) -> dict[str, float]:
        """Return per-component cost breakdown."""
        return {
            "wirelength": self._wirelength_cost(),
            "overlap": self._overlap_cost(),
            "boundary": self._boundary_cost(),
            "thermal": self._thermal_cost(),
            "aspect_ratio": self._aspect_ratio_cost(),
            "total": self.evaluate(),
        }

    # ── Individual Cost Components ────────────────────────────────────

    def _wirelength_cost(self) -> float:
        """Normalized HPWL cost."""
        if self.floorplan.netlist is None:
            return 0.0
        hpwl = self.floorplan.total_hpwl()
        return hpwl / self._chip_diag if self._chip_diag > 0 else 0.0

    def _overlap_cost(self) -> float:
        """
        Overlap penalty: quadratic in overlap area for aggressive penalization.
        """
        overlap = self.floorplan.total_overlap_area()
        normalized = overlap / self._chip_area if self._chip_area > 0 else 0.0
        return normalized ** 2 * 100  # Quadratic penalty

    def _boundary_cost(self) -> float:
        """Penalty for cells extending beyond chip boundary."""
        violations = self.floorplan.boundary_violations()
        total_violation = sum(v[1] for v in violations)
        return total_violation / self._chip_area if self._chip_area > 0 else 0.0

    def _thermal_cost(self) -> float:
        """
        Thermal cost: penalizes clustering of high-power cells.

        Computes a proximity-weighted thermal penalty where nearby
        high-power cells contribute more to the cost.
        """
        if self.floorplan.netlist is None:
            return 0.0

        cells = list(self.floorplan.netlist.cells.values())
        if len(cells) < 2:
            return 0.0

        thermal_cost = 0.0
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                dist = cells[i].distance_to(cells[j])
                if dist < 1.0:
                    dist = 1.0

                # Thermal coupling inversely proportional to distance
                power_product = cells[i].total_power * cells[j].total_power
                thermal_cost += power_product / dist

        # Normalize
        max_power = max(c.total_power for c in cells) if cells else 1.0
        return thermal_cost / (max_power ** 2 * len(cells)) if max_power > 0 else 0.0

    def _aspect_ratio_cost(self) -> float:
        """Penalize extreme aspect ratios in soft macros."""
        if self.floorplan.netlist is None:
            return 0.0

        penalty = 0.0
        count = 0
        for cell in self.floorplan.netlist.cells.values():
            ar = cell.aspect_ratio
            if ar > 0:
                # Penalize aspect ratios far from 1:1
                penalty += (max(ar, 1.0 / ar) - 1.0) ** 2
                count += 1

        return penalty / count if count > 0 else 0.0
