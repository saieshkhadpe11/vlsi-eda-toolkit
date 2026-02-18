"""
Physics-Inspired Agent-Based Floorplanner (PIAB-FP).

Implements a novel physics-inspired approach to VLSI macro-block placement
where each cell is modeled as an autonomous agent subject to physical forces:

  1. REPULSIVE FORCES  — Spring-like overlap resolution (cells push each other)
  2. ATTRACTIVE FORCES — Net-based attraction (connected cells pull together)
  3. BOUNDARY FORCES   — Elastic containment within chip outline
  4. THERMAL FORCES    — Heat diffusion drives hot blocks apart
  5. GRAVITATIONAL     — Gentle drift toward chip center for compaction

Each agent independently resolves its net force, creating emergent global
optimization through local interactions — no global objective function needed.

Key innovations:
  - Adaptive force scheduling (force weights evolve across phases)
  - Velocity damping for convergence stability
  - Agent-level autonomy with configurable response strategies
  - Integrated thermal-awareness during placement (not post-hoc)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import time
import numpy as np

from core.cell import Cell, CellType
from core.netlist import Netlist
from core.floorplan import Floorplan
from core.design import Design
from floorplanner.cost import CostFunction, CostWeights


@dataclass
class PIABConfig:
    """Configuration for the Physics-Inspired Agent-Based Floorplanner."""
    max_iterations: int = 2000
    dt: float = 0.5                     # Time step for physics simulation
    damping: float = 0.85               # Velocity damping factor
    max_velocity: float = 50.0          # Maximum velocity cap

    # Force weights (adaptive — these are initial values)
    w_repulsion: float = 8.0
    w_attraction: float = 2.0
    w_boundary: float = 5.0
    w_thermal: float = 1.5
    w_gravity: float = 0.3

    # Adaptive scheduling
    num_phases: int = 3                 # Coarse → Medium → Fine
    convergence_threshold: float = 0.001  # Stop when avg displacement < this

    seed: Optional[int] = None
    verbose: bool = True
    log_interval: int = 100


@dataclass
class AgentState:
    """Dynamic state of a cell agent."""
    vx: float = 0.0                     # X velocity
    vy: float = 0.0                     # Y velocity
    fx: float = 0.0                     # X net force (accumulated)
    fy: float = 0.0                     # Y net force (accumulated)
    temperature: float = 0.0            # Local temperature estimate


@dataclass
class PIABResult:
    """Result of the PIAB-FP optimization."""
    final_cost: float = float('inf')
    iterations: int = 0
    runtime_seconds: float = 0.0
    cost_history: list[float] = field(default_factory=list)
    displacement_history: list[float] = field(default_factory=list)
    phase_transitions: list[int] = field(default_factory=list)


class PIABFloorplanner:
    """
    Physics-Inspired Agent-Based Floorplanner.

    Models each macro-block as an autonomous agent in a physical system.
    Global optimization emerges from local force interactions between agents.

    Usage:
        piab = PIABFloorplanner(design)
        result = piab.run()
    """

    def __init__(self, design: Design,
                 config: PIABConfig | None = None,
                 cost_weights: CostWeights | None = None):
        self.design = design
        self.config = config or PIABConfig()
        self.cost_weights = cost_weights
        self.rng = np.random.default_rng(self.config.seed)

        if design.floorplan is None:
            raise ValueError("Design must have a floorplan with chip dimensions set.")

        self.floorplan = design.floorplan
        self.netlist = design.netlist
        self.cost_fn = CostFunction(self.floorplan, self.cost_weights)

        # Initialize agent states
        self.agents: dict[str, AgentState] = {}
        for name in self.netlist.cells:
            self.agents[name] = AgentState()

        # Precompute connectivity for attraction forces
        self._connectivity = self._build_connectivity()

        # Callback
        self.on_progress: Optional[Callable[[int, float, float, int], None]] = None

    def _build_connectivity(self) -> dict[str, list[str]]:
        """Build adjacency list from net connectivity."""
        adj: dict[str, list[str]] = {name: [] for name in self.netlist.cells}
        for net in self.netlist.nets.values():
            cell_names = list(net.get_cell_names())
            for i in range(len(cell_names)):
                for j in range(i + 1, len(cell_names)):
                    if cell_names[j] not in adj.get(cell_names[i], []):
                        adj.setdefault(cell_names[i], []).append(cell_names[j])
                    if cell_names[i] not in adj.get(cell_names[j], []):
                        adj.setdefault(cell_names[j], []).append(cell_names[i])
        return adj

    # ── Main Simulation Loop ──────────────────────────────────────────

    def run(self) -> PIABResult:
        """
        Execute the PIAB-FP simulation.

        The algorithm proceeds in phases with adaptive force scheduling:
          Phase 1 (Coarse):  Strong repulsion, weak attraction → spread cells
          Phase 2 (Medium):  Balanced forces → organize by connectivity
          Phase 3 (Fine):    Strong attraction, weak repulsion → compact layout
        """
        result = PIABResult()
        start_time = time.time()

        # Initialize placement
        self._initialize_placement()

        # Phase schedule: (repulsion_scale, attraction_scale, boundary_scale)
        phase_schedule = [
            (2.0, 0.5, 1.0),   # Phase 1: Spread out
            (1.0, 1.0, 1.5),   # Phase 2: Balance
            (0.3, 2.0, 2.0),   # Phase 3: Compact
        ]

        iters_per_phase = self.config.max_iterations // self.config.num_phases

        if self.config.verbose:
            print(f"╔══════════════════════════════════════════════════════════════╗")
            print(f"║  PIAB-FP: Physics-Inspired Agent-Based Floorplanner        ║")
            print(f"║  Cells: {self.netlist.num_cells:<6d}  Nets: {self.netlist.num_nets:<6d}                        ║")
            print(f"║  Phases: {self.config.num_phases}  |  Iterations/Phase: {iters_per_phase}" +
                  " " * max(0, 22 - len(f"{iters_per_phase}")) + "║")
            print(f"╠══════════════════════════════════════════════════════════════╣")

        global_iter = 0
        for phase_idx, (rep_scale, att_scale, bnd_scale) in enumerate(phase_schedule):
            result.phase_transitions.append(global_iter)

            if self.config.verbose:
                print(f"║  Phase {phase_idx + 1}: rep={rep_scale:.1f}x  att={att_scale:.1f}x  "
                      f"bnd={bnd_scale:.1f}x" +
                      " " * max(0, 26 - len(f"rep={rep_scale:.1f}x  att={att_scale:.1f}x  bnd={bnd_scale:.1f}x")) + "║")

            for step in range(iters_per_phase):
                global_iter += 1

                # Reset forces
                for agent in self.agents.values():
                    agent.fx = 0.0
                    agent.fy = 0.0

                # Compute forces
                self._compute_repulsive_forces(rep_scale)
                self._compute_attractive_forces(att_scale)
                self._compute_boundary_forces(bnd_scale)
                self._compute_thermal_forces()
                self._compute_gravity_forces()

                # Update positions
                avg_displacement = self._update_positions()

                # Log
                cost = self.cost_fn.evaluate()
                result.cost_history.append(cost)
                result.displacement_history.append(avg_displacement)

                if self.config.verbose and global_iter % self.config.log_interval == 0:
                    print(f"║  iter={global_iter:>6d}  cost={cost:>8.4f}  "
                          f"disp={avg_displacement:>8.3f}  phase={phase_idx + 1}     ║")

                if self.on_progress:
                    self.on_progress(global_iter, cost, avg_displacement, phase_idx)

                # Convergence check
                if avg_displacement < self.config.convergence_threshold:
                    if self.config.verbose:
                        print(f"║  Converged at iteration {global_iter}" +
                              " " * max(0, 35 - len(f"Converged at iteration {global_iter}")) + "║")
                    break

        elapsed = time.time() - start_time
        result.final_cost = self.cost_fn.evaluate()
        result.iterations = global_iter
        result.runtime_seconds = elapsed

        if self.config.verbose:
            print(f"╠══════════════════════════════════════════════════════════════╣")
            print(f"║  PIAB-FP Complete!                                         ║")
            print(f"║  Final Cost:  {result.final_cost:<8.4f}" +
                  " " * max(0, 34 - len(f"{result.final_cost:<8.4f}")) + "║")
            print(f"║  Iterations:  {global_iter}" +
                  " " * max(0, 34 - len(f"{global_iter}")) + "║")
            print(f"║  Runtime:     {elapsed:.2f}s" +
                  " " * max(0, 34 - len(f"{elapsed:.2f}s")) + "║")
            print(f"╚══════════════════════════════════════════════════════════════╝")

        self.design.snapshot_metrics("piab_fp", elapsed)
        return result

    # ── Force Computations ────────────────────────────────────────────

    def _compute_repulsive_forces(self, scale: float) -> None:
        """
        Repulsive forces between overlapping or nearby cells.

        Uses a spring-like force: F = k * overlap_distance.
        Direction: pushes cells apart along the line connecting their centers.
        """
        w = self.config.w_repulsion * scale
        cells = list(self.netlist.cells.values())

        for i in range(len(cells)):
            if cells[i].fixed:
                continue
            for j in range(i + 1, len(cells)):
                ci, cj = cells[i], cells[j]
                cx_i, cy_i = ci.center
                cx_j, cy_j = cj.center

                # Distance between centers
                dx = cx_i - cx_j
                dy = cy_i - cy_j
                dist = np.sqrt(dx ** 2 + dy ** 2)

                # Minimum distance to avoid overlap
                min_dist = (ci.width + cj.width) / 2.0 + (ci.height + cj.height) / 2.0
                min_dist *= 0.5  # Approximate as circles

                if dist < min_dist:
                    if dist < 1.0:
                        # Jitter to prevent deadlock
                        dx = self.rng.uniform(-1, 1)
                        dy = self.rng.uniform(-1, 1)
                        dist = np.sqrt(dx ** 2 + dy ** 2) + 0.01

                    # Force magnitude proportional to penetration depth
                    penetration = min_dist - dist
                    force = w * penetration

                    # Normalize direction
                    fx = force * dx / dist
                    fy = force * dy / dist

                    if not ci.fixed:
                        self.agents[ci.name].fx += fx
                        self.agents[ci.name].fy += fy
                    if not cj.fixed:
                        self.agents[cj.name].fx -= fx
                        self.agents[cj.name].fy -= fy

    def _compute_attractive_forces(self, scale: float) -> None:
        """
        Attraction between connected cells.

        Pulls cells connected by nets toward each other using a
        spring force: F = k * distance.
        """
        w = self.config.w_attraction * scale

        for cell_name, neighbors in self._connectivity.items():
            cell = self.netlist.cells.get(cell_name)
            if cell is None or cell.fixed:
                continue

            cx, cy = cell.center

            for neighbor_name in neighbors:
                neighbor = self.netlist.cells.get(neighbor_name)
                if neighbor is None:
                    continue

                nx, ny = neighbor.center
                dx = nx - cx
                dy = ny - cy
                dist = np.sqrt(dx ** 2 + dy ** 2)

                if dist > 1.0:
                    force = w * np.log(1 + dist)  # Sub-linear attraction

                    self.agents[cell_name].fx += force * dx / dist
                    self.agents[cell_name].fy += force * dy / dist

    def _compute_boundary_forces(self, scale: float) -> None:
        """
        Elastic boundary containment.

        Pushes cells back inside the chip when they extend beyond boundaries.
        """
        w = self.config.w_boundary * scale

        for name, cell in self.netlist.cells.items():
            if cell.fixed:
                continue

            agent = self.agents[name]
            x_min, y_min, x_max, y_max = cell.bbox

            # Left boundary
            if x_min < 0:
                agent.fx += w * abs(x_min)
            # Right boundary
            if x_max > self.floorplan.chip_width:
                agent.fx -= w * (x_max - self.floorplan.chip_width)
            # Bottom boundary
            if y_min < 0:
                agent.fy += w * abs(y_min)
            # Top boundary
            if y_max > self.floorplan.chip_height:
                agent.fy -= w * (y_max - self.floorplan.chip_height)

    def _compute_thermal_forces(self) -> None:
        """
        Thermal dispersion forces.

        High-power cells repel each other proportionally to their
        combined power density, encouraging thermal distribution.
        """
        w = self.config.w_thermal
        cells = list(self.netlist.cells.values())

        for i in range(len(cells)):
            if cells[i].fixed:
                continue
            for j in range(i + 1, len(cells)):
                ci, cj = cells[i], cells[j]

                # Only apply for high-power cells
                combined_power = ci.total_power + cj.total_power
                if combined_power < 0.001:
                    continue

                cx_i, cy_i = ci.center
                cx_j, cy_j = cj.center

                dx = cx_i - cx_j
                dy = cy_i - cy_j
                dist = np.sqrt(dx ** 2 + dy ** 2) + 1.0

                # Thermal force: inversely proportional to distance²
                force = w * combined_power / (dist ** 2)

                if dist > 0.1:
                    fx = force * dx / dist
                    fy = force * dy / dist

                    if not ci.fixed:
                        self.agents[ci.name].fx += fx
                        self.agents[ci.name].fy += fy
                    if not cj.fixed:
                        self.agents[cj.name].fx -= fx
                        self.agents[cj.name].fy -= fy

    def _compute_gravity_forces(self) -> None:
        """
        Gentle gravitational pull toward chip center.

        Provides compaction pressure to reduce deadspace.
        """
        w = self.config.w_gravity
        center_x = self.floorplan.chip_width / 2
        center_y = self.floorplan.chip_height / 2

        for name, cell in self.netlist.cells.items():
            if cell.fixed:
                continue

            cx, cy = cell.center
            dx = center_x - cx
            dy = center_y - cy
            dist = np.sqrt(dx ** 2 + dy ** 2)

            if dist > 1.0:
                force = w * np.sqrt(dist)  # Sub-linear gravity
                self.agents[name].fx += force * dx / dist
                self.agents[name].fy += force * dy / dist

    # ── Position Update ───────────────────────────────────────────────

    def _update_positions(self) -> float:
        """
        Update cell positions using Verlet-like integration.

        Applies velocity damping and maximum velocity capping
        for numerical stability.

        Returns:
            Average displacement of all movable cells.
        """
        total_displacement = 0.0
        movable_count = 0

        for name, cell in self.netlist.cells.items():
            if cell.fixed:
                continue

            agent = self.agents[name]

            # Update velocity: v = damping * v + dt * F/m (mass ~ area)
            mass = max(cell.area, 1.0)
            agent.vx = self.config.damping * agent.vx + self.config.dt * agent.fx / mass
            agent.vy = self.config.damping * agent.vy + self.config.dt * agent.fy / mass

            # Velocity capping
            speed = np.sqrt(agent.vx ** 2 + agent.vy ** 2)
            if speed > self.config.max_velocity:
                agent.vx *= self.config.max_velocity / speed
                agent.vy *= self.config.max_velocity / speed

            # Update position
            dx = agent.vx * self.config.dt
            dy = agent.vy * self.config.dt

            cell.x += dx
            cell.y += dy

            displacement = np.sqrt(dx ** 2 + dy ** 2)
            total_displacement += displacement
            movable_count += 1

        return total_displacement / max(movable_count, 1)

    # ── Initialization ────────────────────────────────────────────────

    def _initialize_placement(self) -> None:
        """
        Initialize cells on a grid with some randomness.

        Uses a grid-based layout to start with a spread-out configuration,
        then adds random perturbation to break symmetry.
        """
        movable = self.netlist.movable_cells
        n = len(movable)
        if n == 0:
            return

        # Grid layout
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        cell_w = self.floorplan.chip_width / (cols + 1)
        cell_h = self.floorplan.chip_height / (rows + 1)

        for idx, cell in enumerate(movable):
            row = idx // cols
            col = idx % cols

            x = (col + 0.5) * cell_w + self.rng.uniform(-cell_w * 0.2, cell_w * 0.2)
            y = (row + 0.5) * cell_h + self.rng.uniform(-cell_h * 0.2, cell_h * 0.2)

            x = np.clip(x, 0, self.floorplan.chip_width - cell.width)
            y = np.clip(y, 0, self.floorplan.chip_height - cell.height)

            cell.place(x, y)

            # Initialize with zero velocity
            self.agents[cell.name].vx = 0.0
            self.agents[cell.name].vy = 0.0
