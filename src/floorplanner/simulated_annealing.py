"""
Simulated Annealing Floorplanner.

Implements the classical SA-based approach to VLSI macro block placement.
Uses a sequence of random perturbations (moves, swaps, rotations) with
temperature-based acceptance to explore the solution space.

Key features:
  - Configurable cooling schedule (geometric, adaptive)
  - Multiple move types: translate, swap, rotate, reshape (soft macros)
  - Convergence detection via cost stagnation
  - Full solution trajectory logging
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import time
import copy
import numpy as np

from core.cell import Cell, CellType
from core.netlist import Netlist
from core.floorplan import Floorplan
from core.design import Design
from floorplanner.cost import CostFunction, CostWeights


@dataclass
class SAConfig:
    """Configuration for the Simulated Annealing algorithm."""
    initial_temperature: float = 1000.0
    cooling_rate: float = 0.995
    min_temperature: float = 0.01
    moves_per_temperature: int = 100
    max_iterations: int = 500000
    restart_threshold: int = 5000       # Restart if no improvement for this many steps
    adaptive_cooling: bool = True       # Adjust cooling rate based on acceptance
    target_acceptance: float = 0.44     # Target acceptance ratio for adaptive cooling
    seed: Optional[int] = None
    verbose: bool = True
    log_interval: int = 1000            # Log every N iterations


@dataclass
class SAResult:
    """Result of the Simulated Annealing optimization."""
    best_cost: float = float('inf')
    final_cost: float = float('inf')
    iterations: int = 0
    runtime_seconds: float = 0.0
    cost_history: list[float] = field(default_factory=list)
    temperature_history: list[float] = field(default_factory=list)
    acceptance_history: list[float] = field(default_factory=list)
    best_iteration: int = 0


class SimulatedAnnealingFloorplanner:
    """
    Simulated Annealing based VLSI floorplanner.

    Optimizes cell placement by iteratively perturbing the solution
    and accepting/rejecting moves based on the Metropolis criterion.

    Usage:
        sa = SimulatedAnnealingFloorplanner(design)
        result = sa.run()
        print(result.best_cost)
    """

    def __init__(self, design: Design,
                 config: SAConfig | None = None,
                 cost_weights: CostWeights | None = None):
        self.design = design
        self.config = config or SAConfig()
        self.cost_weights = cost_weights
        self.rng = np.random.default_rng(self.config.seed)

        if design.floorplan is None:
            raise ValueError("Design must have a floorplan with chip dimensions set.")

        self.floorplan = design.floorplan
        self.netlist = design.netlist
        self.cost_fn = CostFunction(self.floorplan, self.cost_weights)

        # Move probabilities [translate, swap, rotate, reshape]
        self._move_probs = np.array([0.45, 0.25, 0.20, 0.10])

        # Callback for progress reporting
        self.on_progress: Optional[Callable[[int, float, float], None]] = None

    # ── Main Optimization Loop ────────────────────────────────────────

    def run(self) -> SAResult:
        """
        Execute the Simulated Annealing optimization.

        Returns:
            SAResult with optimization history and final metrics.
        """
        result = SAResult()
        start_time = time.time()

        # Initialize: random placement
        self._random_initial_placement()
        current_cost = self.cost_fn.evaluate()
        best_cost = current_cost
        result.best_cost = best_cost

        # Save best placement
        best_positions = self._save_positions()

        temperature = self.config.initial_temperature
        no_improve_count = 0
        accepted = 0
        total_moves = 0

        if self.config.verbose:
            print(f"╔══════════════════════════════════════════════════════╗")
            print(f"║  Simulated Annealing Floorplanner                   ║")
            print(f"║  Cells: {self.netlist.num_cells:<6d}  Nets: {self.netlist.num_nets:<6d}              ║")
            print(f"║  Chip:  {self.floorplan.chip_width:.0f} × {self.floorplan.chip_height:.0f}" +
                  " " * max(0, 30 - len(f"{self.floorplan.chip_width:.0f} × {self.floorplan.chip_height:.0f}")) + "║")
            print(f"║  Initial Cost: {current_cost:.4f}" +
                  " " * max(0, 27 - len(f"{current_cost:.4f}")) + "║")
            print(f"╠══════════════════════════════════════════════════════╣")

        iteration = 0
        while temperature > self.config.min_temperature and iteration < self.config.max_iterations:
            moves_at_temp = 0
            accepted_at_temp = 0

            for _ in range(self.config.moves_per_temperature):
                iteration += 1
                total_moves += 1

                # Perform a random move
                undo_data = self._random_move(temperature)

                # Evaluate new cost
                new_cost = self.cost_fn.evaluate()
                delta = new_cost - current_cost

                # Metropolis acceptance criterion
                if delta < 0 or self.rng.random() < np.exp(-delta / max(temperature, 1e-10)):
                    current_cost = new_cost
                    accepted += 1
                    accepted_at_temp += 1

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_positions = self._save_positions()
                        result.best_iteration = iteration
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Reject: undo the move
                    self._undo_move(undo_data)
                    no_improve_count += 1

                moves_at_temp += 1

            # Log progress
            acceptance_ratio = accepted_at_temp / max(moves_at_temp, 1)
            result.cost_history.append(current_cost)
            result.temperature_history.append(temperature)
            result.acceptance_history.append(acceptance_ratio)

            if self.config.verbose and iteration % self.config.log_interval < self.config.moves_per_temperature:
                elapsed = time.time() - start_time
                print(f"║  iter={iteration:>7d}  T={temperature:>8.2f}  "
                      f"cost={current_cost:>8.4f}  best={best_cost:>8.4f}  "
                      f"acc={acceptance_ratio:.2f}  ║")

            if self.on_progress:
                self.on_progress(iteration, current_cost, temperature)

            # Cooling
            if self.config.adaptive_cooling:
                temperature = self._adaptive_cool(temperature, acceptance_ratio)
            else:
                temperature *= self.config.cooling_rate

            # Restart if stuck
            if no_improve_count > self.config.restart_threshold:
                temperature = max(temperature * 10, self.config.initial_temperature * 0.1)
                no_improve_count = 0
                if self.config.verbose:
                    print(f"║  *** REHEAT *** T -> {temperature:.2f}" +
                          " " * max(0, 35 - len(f"T -> {temperature:.2f}")) + "║")

        # Restore best solution
        self._restore_positions(best_positions)

        elapsed = time.time() - start_time
        result.best_cost = best_cost
        result.final_cost = self.cost_fn.evaluate()
        result.iterations = iteration
        result.runtime_seconds = elapsed

        if self.config.verbose:
            print(f"╠══════════════════════════════════════════════════════╣")
            print(f"║  Optimization Complete!                              ║")
            print(f"║  Best Cost:   {best_cost:<8.4f}" +
                  " " * max(0, 28 - len(f"{best_cost:<8.4f}")) + "║")
            print(f"║  Iterations:  {iteration}" +
                  " " * max(0, 28 - len(f"{iteration}")) + "║")
            print(f"║  Runtime:     {elapsed:.2f}s" +
                  " " * max(0, 28 - len(f"{elapsed:.2f}s")) + "║")
            print(f"╚══════════════════════════════════════════════════════╝")

        # Record metrics
        self.design.snapshot_metrics("simulated_annealing", elapsed)

        return result

    # ── Move Operations ───────────────────────────────────────────────

    def _random_move(self, temperature: float) -> dict:
        """
        Perform a random perturbation move.

        Move types:
          1. Translate: move a cell to a random nearby position
          2. Swap: swap positions of two cells
          3. Rotate: rotate a cell 90 degrees
          4. Reshape: change aspect ratio of soft macro

        Returns:
            Undo data for reverting the move.
        """
        movable = self.netlist.movable_cells
        if not movable:
            return {}

        move_type = self.rng.choice(4, p=self._move_probs)

        if move_type == 0:
            return self._move_translate(movable, temperature)
        elif move_type == 1:
            return self._move_swap(movable)
        elif move_type == 2:
            return self._move_rotate(movable)
        else:
            return self._move_reshape(movable)

    def _move_translate(self, movable: list[Cell], temperature: float) -> dict:
        """Move a cell to a nearby position (distance scaled by temperature)."""
        cell = self.rng.choice(movable)
        old_x, old_y = cell.x, cell.y

        # Move distance proportional to temperature
        max_dist = (temperature / self.config.initial_temperature) * \
                   max(self.floorplan.chip_width, self.floorplan.chip_height) * 0.3
        max_dist = max(max_dist, 5.0)

        dx = self.rng.uniform(-max_dist, max_dist)
        dy = self.rng.uniform(-max_dist, max_dist)

        new_x = np.clip(cell.x + dx, 0, self.floorplan.chip_width - cell.width)
        new_y = np.clip(cell.y + dy, 0, self.floorplan.chip_height - cell.height)

        cell.x = new_x
        cell.y = new_y

        return {"type": "translate", "cell": cell.name, "old_x": old_x, "old_y": old_y}

    def _move_swap(self, movable: list[Cell]) -> dict:
        """Swap positions of two cells."""
        if len(movable) < 2:
            return self._move_translate(movable, self.config.initial_temperature)

        idx = self.rng.choice(len(movable), size=2, replace=False)
        c1, c2 = movable[idx[0]], movable[idx[1]]

        old_x1, old_y1 = c1.x, c1.y
        old_x2, old_y2 = c2.x, c2.y

        c1.x, c1.y = old_x2, old_y2
        c2.x, c2.y = old_x1, old_y1

        return {"type": "swap", "cell1": c1.name, "cell2": c2.name,
                "old_x1": old_x1, "old_y1": old_y1,
                "old_x2": old_x2, "old_y2": old_y2}

    def _move_rotate(self, movable: list[Cell]) -> dict:
        """Rotate a cell 90 degrees (swap width and height)."""
        cell = self.rng.choice(movable)
        if cell.cell_type == CellType.SOFT_MACRO or cell.cell_type == CellType.HARD_MACRO:
            old_w, old_h = cell.width, cell.height
            cell.width, cell.height = old_h, old_w
            return {"type": "rotate", "cell": cell.name, "old_w": old_w, "old_h": old_h}
        return self._move_translate(movable, self.config.initial_temperature * 0.5)

    def _move_reshape(self, movable: list[Cell]) -> dict:
        """Change aspect ratio of a soft macro while preserving area."""
        soft_macros = [c for c in movable if c.cell_type == CellType.SOFT_MACRO]
        if not soft_macros:
            return self._move_translate(movable, self.config.initial_temperature * 0.5)

        cell = self.rng.choice(soft_macros)
        old_w, old_h = cell.width, cell.height

        new_ar = self.rng.uniform(0.5, 2.0)
        cell.reshape(new_ar)

        return {"type": "reshape", "cell": cell.name, "old_w": old_w, "old_h": old_h}

    def _undo_move(self, undo_data: dict) -> None:
        """Revert a move using undo data."""
        if not undo_data:
            return

        move_type = undo_data["type"]

        if move_type == "translate":
            cell = self.netlist.cells[undo_data["cell"]]
            cell.x = undo_data["old_x"]
            cell.y = undo_data["old_y"]

        elif move_type == "swap":
            c1 = self.netlist.cells[undo_data["cell1"]]
            c2 = self.netlist.cells[undo_data["cell2"]]
            c1.x, c1.y = undo_data["old_x1"], undo_data["old_y1"]
            c2.x, c2.y = undo_data["old_x2"], undo_data["old_y2"]

        elif move_type == "rotate":
            cell = self.netlist.cells[undo_data["cell"]]
            cell.width = undo_data["old_w"]
            cell.height = undo_data["old_h"]

        elif move_type == "reshape":
            cell = self.netlist.cells[undo_data["cell"]]
            cell.width = undo_data["old_w"]
            cell.height = undo_data["old_h"]

    # ── Initialization ────────────────────────────────────────────────

    def _random_initial_placement(self) -> None:
        """Place all movable cells at random non-overlapping positions."""
        for cell in self.netlist.movable_cells:
            max_x = max(0, self.floorplan.chip_width - cell.width)
            max_y = max(0, self.floorplan.chip_height - cell.height)
            cell.x = self.rng.uniform(0, max_x)
            cell.y = self.rng.uniform(0, max_y)

    # ── State Management ──────────────────────────────────────────────

    def _save_positions(self) -> dict[str, tuple[float, float, float, float]]:
        """Save current cell positions and dimensions."""
        return {
            name: (cell.x, cell.y, cell.width, cell.height)
            for name, cell in self.netlist.cells.items()
        }

    def _restore_positions(self, positions: dict[str, tuple[float, float, float, float]]) -> None:
        """Restore cell positions from saved state."""
        for name, (x, y, w, h) in positions.items():
            if name in self.netlist.cells:
                cell = self.netlist.cells[name]
                cell.x, cell.y = x, y
                cell.width, cell.height = w, h

    # ── Adaptive Cooling ──────────────────────────────────────────────

    def _adaptive_cool(self, temperature: float, acceptance_ratio: float) -> float:
        """
        Adaptive cooling: adjust rate based on acceptance ratio.

        If acceptance is too high → cool faster (exploring too freely)
        If acceptance is too low  → cool slower (getting stuck)
        """
        if acceptance_ratio > self.config.target_acceptance + 0.1:
            rate = 0.990  # Cool faster
        elif acceptance_ratio < self.config.target_acceptance - 0.1:
            rate = 0.999  # Cool slower
        else:
            rate = self.config.cooling_rate

        return temperature * rate
