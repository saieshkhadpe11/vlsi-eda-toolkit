"""
Global Router for VLSI Physical Design.

Implements A*-based global routing on a grid graph overlaid on the chip.
Each routing grid cell (GCell) has a capacity representing available
routing resources. The router finds paths for each net while respecting
capacity constraints.

Key features:
  - A* shortest path with congestion-aware cost
  - Net ordering by criticality and bounding box
  - Rip-up and re-route for congestion resolution
  - Overflow detection and reporting
  - Via minimization (layer change penalty)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import heapq
import time
import numpy as np

from core.netlist import Netlist
from core.net import Net
from core.floorplan import Floorplan
from core.design import Design


@dataclass
class GCell:
    """
    Global routing cell (grid tile).

    Attributes:
        row: Row index in the routing grid.
        col: Column index in the routing grid.
        capacity: Maximum routing tracks available.
        usage: Current number of tracks used.
        blockage: Proportion of cell blocked by macros (0-1).
    """
    row: int
    col: int
    capacity: int = 10
    usage: int = 0
    blockage: float = 0.0

    @property
    def available(self) -> int:
        """Available routing resources."""
        effective_cap = int(self.capacity * (1 - self.blockage))
        return max(0, effective_cap - self.usage)

    @property
    def overflow(self) -> int:
        """Overflow (negative = within capacity)."""
        effective_cap = int(self.capacity * (1 - self.blockage))
        return max(0, self.usage - effective_cap)

    @property
    def congestion(self) -> float:
        """Congestion ratio (0-1+, >1 means overflow)."""
        effective_cap = int(self.capacity * (1 - self.blockage))
        return self.usage / max(effective_cap, 1)


@dataclass
class RoutePath:
    """A routed path for a single net."""
    net_name: str
    gcells: list[tuple[int, int]] = field(default_factory=list)  # (row, col) sequence
    wirelength: float = 0.0
    vias: int = 0

    @property
    def num_segments(self) -> int:
        return max(0, len(self.gcells) - 1)


@dataclass
class RoutingResult:
    """Result of the global routing process."""
    routed_nets: int = 0
    total_nets: int = 0
    total_overflow: int = 0
    total_wirelength: float = 0.0
    max_congestion: float = 0.0
    runtime_seconds: float = 0.0
    paths: dict[str, RoutePath] = field(default_factory=dict)
    failed_nets: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.routed_nets / max(self.total_nets, 1)


class RoutingGrid:
    """
    2D routing grid overlaid on the chip floorplan.

    Divides the chip into a grid of GCells, each with a routing capacity.
    Macro blocks create blockages that reduce local capacity.
    """

    def __init__(self, chip_width: float, chip_height: float,
                 grid_cols: int = 20, grid_rows: int = 20,
                 default_capacity: int = 10):
        self.chip_width = chip_width
        self.chip_height = chip_height
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows

        self.cell_width = chip_width / grid_cols
        self.cell_height = chip_height / grid_rows

        # Create grid
        self.grid: list[list[GCell]] = []
        for r in range(grid_rows):
            row = []
            for c in range(grid_cols):
                row.append(GCell(row=r, col=c, capacity=default_capacity))
            self.grid.append(row)

    def get_gcell(self, row: int, col: int) -> Optional[GCell]:
        """Get a GCell by grid coordinates."""
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            return self.grid[row][col]
        return None

    def coord_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert chip coordinates to grid (row, col)."""
        col = min(int(x / self.cell_width), self.grid_cols - 1)
        row = min(int(y / self.cell_height), self.grid_rows - 1)
        return (max(0, row), max(0, col))

    def grid_to_coord(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid (row, col) to chip coordinates (center of GCell)."""
        x = (col + 0.5) * self.cell_width
        y = (row + 0.5) * self.cell_height
        return (x, y)

    def apply_blockages(self, netlist: Netlist) -> None:
        """Mark GCells blocked by macro cells."""
        for cell in netlist.cells.values():
            x_min, y_min, x_max, y_max = cell.bbox

            r_min, c_min = self.coord_to_grid(x_min, y_min)
            r_max, c_max = self.coord_to_grid(x_max, y_max)

            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    gcell = self.get_gcell(r, c)
                    if gcell:
                        # Calculate proportion of GCell blocked
                        gx_min = c * self.cell_width
                        gx_max = (c + 1) * self.cell_width
                        gy_min = r * self.cell_height
                        gy_max = (r + 1) * self.cell_height

                        overlap_x = max(0, min(x_max, gx_max) - max(x_min, gx_min))
                        overlap_y = max(0, min(y_max, gy_max) - max(y_min, gy_min))
                        overlap_area = overlap_x * overlap_y
                        gcell_area = self.cell_width * self.cell_height

                        gcell.blockage = min(1.0, gcell.blockage + overlap_area / gcell_area)

    def neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        """Get valid 4-connected neighbors."""
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                result.append((nr, nc))
        return result

    def total_overflow(self) -> int:
        """Total overflow across all GCells."""
        return sum(
            self.grid[r][c].overflow
            for r in range(self.grid_rows)
            for c in range(self.grid_cols)
        )

    def max_congestion(self) -> float:
        """Maximum congestion across all GCells."""
        return max(
            self.grid[r][c].congestion
            for r in range(self.grid_rows)
            for c in range(self.grid_cols)
        )

    def congestion_map(self) -> np.ndarray:
        """Return congestion as a 2D numpy array for visualization."""
        cmap = np.zeros((self.grid_rows, self.grid_cols))
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                cmap[r][c] = self.grid[r][c].congestion
        return cmap


class GlobalRouter:
    """
    A*-based global router with congestion-aware pathfinding.

    Routes all nets in the design by finding paths through the routing
    grid while minimizing wirelength and avoiding congested regions.

    Usage:
        router = GlobalRouter(design)
        result = router.route()
    """

    def __init__(self, design: Design,
                 grid_cols: int = 20, grid_rows: int = 20,
                 default_capacity: int = 10,
                 congestion_penalty: float = 5.0,
                 max_rip_up_iterations: int = 3):
        self.design = design
        self.floorplan = design.floorplan
        self.netlist = design.netlist
        self.congestion_penalty = congestion_penalty
        self.max_rip_up_iterations = max_rip_up_iterations

        # Create routing grid
        self.routing_grid = RoutingGrid(
            chip_width=self.floorplan.chip_width,
            chip_height=self.floorplan.chip_height,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            default_capacity=default_capacity,
        )

        # Apply macro blockages
        self.routing_grid.apply_blockages(self.netlist)

    def route(self, verbose: bool = True) -> RoutingResult:
        """
        Route all nets in the design.

        Process:
          1. Order nets by criticality (weighted, smaller bbox first)
          2. Route each net using A* with congestion-aware cost
          3. Rip-up-and-reroute to resolve congestion
        """
        result = RoutingResult()
        start_time = time.time()

        # Order nets: smaller bounding box first, then by weight
        nets = self._order_nets()
        result.total_nets = len(nets)

        if verbose:
            print(f"┌──────────────────────────────────────────┐")
            print(f"│  Global Router - A* with Congestion      │")
            print(f"│  Nets: {len(nets):<6d}                          │")
            print(f"│  Grid: {self.routing_grid.grid_rows}×{self.routing_grid.grid_cols}" +
                  " " * max(0, 28 - len(f"{self.routing_grid.grid_rows}×{self.routing_grid.grid_cols}")) + "│")
            print(f"├──────────────────────────────────────────┤")

        # Initial routing
        for net in nets:
            path = self._route_net(net)
            if path:
                result.paths[net.name] = path
                result.routed_nets += 1
                result.total_wirelength += path.wirelength
            else:
                result.failed_nets.append(net.name)

        if verbose:
            print(f"│  Initial: {result.routed_nets}/{result.total_nets} routed"
                  f"  overflow={self.routing_grid.total_overflow()}" +
                  " " * 4 + "│")

        # Rip-up and re-route iterations
        for rr_iter in range(self.max_rip_up_iterations):
            overflow = self.routing_grid.total_overflow()
            if overflow == 0:
                break

            # Find congested nets and rip them up
            ripped = self._rip_up_congested_nets(result)
            if not ripped:
                break

            # Re-route ripped nets with higher congestion penalty
            self.congestion_penalty *= 1.5
            for net_name in ripped:
                net = self.netlist.nets[net_name]
                path = self._route_net(net)
                if path:
                    result.paths[net_name] = path
                else:
                    result.failed_nets.append(net_name)

            if verbose:
                overflow = self.routing_grid.total_overflow()
                print(f"│  RR iter {rr_iter + 1}: ripped={len(ripped)}"
                      f"  overflow={overflow}" +
                      " " * max(0, 13 - len(f"ripped={len(ripped)}  overflow={overflow}")) + "│")

        elapsed = time.time() - start_time
        result.runtime_seconds = elapsed
        result.total_overflow = self.routing_grid.total_overflow()
        result.max_congestion = self.routing_grid.max_congestion()

        # Recalculate total wirelength
        result.total_wirelength = sum(p.wirelength for p in result.paths.values())
        result.routed_nets = len(result.paths)

        if verbose:
            print(f"├──────────────────────────────────────────┤")
            print(f"│  Routing Complete!                       │")
            print(f"│  Routed:     {result.routed_nets}/{result.total_nets}" +
                  " " * max(0, 25 - len(f"{result.routed_nets}/{result.total_nets}")) + "│")
            print(f"│  Wirelength: {result.total_wirelength:.1f}" +
                  " " * max(0, 25 - len(f"{result.total_wirelength:.1f}")) + "│")
            print(f"│  Overflow:   {result.total_overflow}" +
                  " " * max(0, 25 - len(f"{result.total_overflow}")) + "│")
            print(f"│  Runtime:    {elapsed:.2f}s" +
                  " " * max(0, 25 - len(f"{elapsed:.2f}s")) + "│")
            print(f"└──────────────────────────────────────────┘")

        self.design.snapshot_metrics("global_routing", elapsed)
        return result

    def _route_net(self, net: Net) -> Optional[RoutePath]:
        """
        Route a single net using A* algorithm.

        For multi-pin nets, uses a minimum spanning tree decomposition:
        routes between pairs of pins incrementally.
        """
        cell_names = list(net.get_cell_names())
        valid_cells = [n for n in cell_names if n in self.netlist.cells]

        if len(valid_cells) < 2:
            return RoutePath(net_name=net.name)

        # Get grid positions of all cells
        grid_positions = []
        for cell_name in valid_cells:
            cell = self.netlist.cells[cell_name]
            cx, cy = cell.center
            r, c = self.routing_grid.coord_to_grid(cx, cy)
            grid_positions.append((r, c))

        # Decompose into 2-pin connections (star topology from first pin)
        all_gcells = [grid_positions[0]]
        total_wl = 0.0

        for i in range(1, len(grid_positions)):
            source = grid_positions[0]  # Star center
            target = grid_positions[i]

            path_cells = self._astar(source, target)
            if path_cells is None:
                return None

            all_gcells.extend(path_cells[1:])  # Avoid duplicating source
            total_wl += len(path_cells) - 1

        # Update grid usage
        seen = set()
        for r, c in all_gcells:
            if (r, c) not in seen:
                gcell = self.routing_grid.get_gcell(r, c)
                if gcell:
                    gcell.usage += 1
                seen.add((r, c))

        # Calculate wirelength in physical units
        wl = total_wl * (self.routing_grid.cell_width + self.routing_grid.cell_height) / 2

        return RoutePath(
            net_name=net.name,
            gcells=list(dict.fromkeys(all_gcells)),  # Deduplicate preserving order
            wirelength=wl,
        )

    def _astar(self, source: tuple[int, int], target: tuple[int, int]) -> Optional[list[tuple[int, int]]]:
        """
        A* pathfinding with congestion-aware cost.

        g(n) = distance from source + congestion penalty
        h(n) = Manhattan distance to target (admissible heuristic)
        """
        sr, sc = source
        tr, tc = target

        if source == target:
            return [source]

        # Priority queue: (f_cost, g_cost, row, col)
        open_set = [(0, 0, sr, sc)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {source: 0}

        while open_set:
            f, g, r, c = heapq.heappop(open_set)

            if (r, c) == target:
                # Reconstruct path
                path = [(r, c)]
                current = (r, c)
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for nr, nc in self.routing_grid.neighbors(r, c):
                gcell = self.routing_grid.get_gcell(nr, nc)
                if gcell is None:
                    continue

                # Cost: base + congestion penalty
                base_cost = 1.0
                cong_cost = 0.0
                if gcell.available <= 0:
                    cong_cost = self.congestion_penalty * (1 + gcell.overflow)
                elif gcell.congestion > 0.7:
                    cong_cost = self.congestion_penalty * gcell.congestion

                # Blockage cost
                if gcell.blockage > 0.9:
                    cong_cost += self.congestion_penalty * 3

                tentative_g = g + base_cost + cong_cost

                if tentative_g < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = tentative_g
                    came_from[(nr, nc)] = (r, c)
                    h = abs(nr - tr) + abs(nc - tc)  # Manhattan heuristic
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, nr, nc))

        return None  # No path found

    def _order_nets(self) -> list[Net]:
        """Order nets for routing: smaller HPWL first, higher weight first."""
        positions = self.netlist.get_cell_positions()

        def net_priority(net: Net) -> tuple[float, float]:
            hpwl = net.hpwl(positions)
            return (-net.weight, hpwl)  # High weight first, small HPWL first

        sorted_nets = sorted(self.netlist.nets.values(), key=net_priority)
        return [n for n in sorted_nets if n.degree >= 2]

    def _rip_up_congested_nets(self, result: RoutingResult) -> list[str]:
        """Rip up nets passing through congested GCells."""
        ripped = []

        for net_name, path in list(result.paths.items()):
            congested = False
            for r, c in path.gcells:
                gcell = self.routing_grid.get_gcell(r, c)
                if gcell and gcell.overflow > 0:
                    congested = True
                    break

            if congested:
                # Remove path's usage from grid
                seen = set()
                for r, c in path.gcells:
                    if (r, c) not in seen:
                        gcell = self.routing_grid.get_gcell(r, c)
                        if gcell:
                            gcell.usage = max(0, gcell.usage - 1)
                        seen.add((r, c))

                del result.paths[net_name]
                ripped.append(net_name)

        return ripped
