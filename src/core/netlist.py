"""
Netlist: the complete design connectivity graph.

A Netlist holds all cells and nets in the design and provides methods for
connectivity queries, hypergraph traversal, and design statistics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .cell import Cell, CellType
from .net import Net


@dataclass
class Netlist:
    """
    Complete design netlist containing cells and nets.

    Provides a unified interface for querying design connectivity,
    computing wirelength, and analyzing the design.

    Attributes:
        name: Design name.
        cells: Dictionary of cells keyed by name.
        nets: Dictionary of nets keyed by name.
    """
    name: str = "design"
    cells: dict[str, Cell] = field(default_factory=dict)
    nets: dict[str, Net] = field(default_factory=dict)

    # ── Cell Management ───────────────────────────────────────────────

    def add_cell(self, cell: Cell) -> None:
        """Add a cell to the netlist."""
        self.cells[cell.name] = cell

    def get_cell(self, name: str) -> Optional[Cell]:
        """Get a cell by name."""
        return self.cells.get(name)

    def remove_cell(self, name: str) -> None:
        """Remove a cell and all its net connections."""
        if name in self.cells:
            # Remove from nets
            for net in self.nets.values():
                net.pins = [p for p in net.pins if p.cell_name != name]
            del self.cells[name]

    @property
    def movable_cells(self) -> list[Cell]:
        """Get all non-fixed cells."""
        return [c for c in self.cells.values() if not c.fixed]

    @property
    def fixed_cells(self) -> list[Cell]:
        """Get all fixed/pre-placed cells."""
        return [c for c in self.cells.values() if c.fixed]

    @property
    def macros(self) -> list[Cell]:
        """Get all hard and soft macros."""
        return [c for c in self.cells.values()
                if c.cell_type in (CellType.HARD_MACRO, CellType.SOFT_MACRO)]

    # ── Net Management ────────────────────────────────────────────────

    def add_net(self, net: Net) -> None:
        """Add a net to the netlist."""
        self.nets[net.name] = net

    def get_nets_of_cell(self, cell_name: str) -> list[Net]:
        """Get all nets connected to a given cell."""
        result = []
        for net in self.nets.values():
            for pin in net.pins:
                if pin.cell_name == cell_name:
                    result.append(net)
                    break
        return result

    def get_connected_cells(self, cell_name: str) -> set[str]:
        """Get all cells directly connected to a given cell."""
        connected = set()
        for net in self.get_nets_of_cell(cell_name):
            for pin in net.pins:
                if pin.cell_name != cell_name:
                    connected.add(pin.cell_name)
        return connected

    # ── Position Queries ──────────────────────────────────────────────

    def get_cell_positions(self) -> dict[str, tuple[float, float]]:
        """Get center positions of all cells as a dictionary."""
        return {name: cell.center for name, cell in self.cells.items()}

    # ── Wirelength ────────────────────────────────────────────────────

    def total_hpwl(self) -> float:
        """Compute total HPWL across all nets."""
        positions = self.get_cell_positions()
        return sum(net.hpwl(positions) for net in self.nets.values())

    def total_star_wirelength(self) -> float:
        """Compute total star-model wirelength across all nets."""
        positions = self.get_cell_positions()
        return sum(net.star_wirelength(positions) for net in self.nets.values())

    # ── Design Statistics ─────────────────────────────────────────────

    @property
    def num_cells(self) -> int:
        return len(self.cells)

    @property
    def num_nets(self) -> int:
        return len(self.nets)

    @property
    def total_cell_area(self) -> float:
        """Total area of all cells."""
        return sum(c.area for c in self.cells.values())

    @property
    def total_power(self) -> float:
        """Total power dissipation."""
        return sum(c.total_power for c in self.cells.values())

    def avg_net_degree(self) -> float:
        """Average number of pins per net."""
        if not self.nets:
            return 0.0
        return np.mean([net.degree for net in self.nets.values()])

    def connectivity_matrix(self) -> np.ndarray:
        """
        Build cell-to-cell connectivity matrix.

        Returns an NxN matrix where entry (i,j) is the number of nets
        connecting cell i to cell j.
        """
        cell_list = list(self.cells.keys())
        cell_idx = {name: i for i, name in enumerate(cell_list)}
        n = len(cell_list)
        matrix = np.zeros((n, n), dtype=int)

        for net in self.nets.values():
            cell_names = list(net.get_cell_names())
            for i in range(len(cell_names)):
                for j in range(i + 1, len(cell_names)):
                    ci = cell_idx.get(cell_names[i])
                    cj = cell_idx.get(cell_names[j])
                    if ci is not None and cj is not None:
                        matrix[ci][cj] += 1
                        matrix[cj][ci] += 1

        return matrix

    def summary(self) -> str:
        """Return a human-readable summary of the netlist."""
        lines = [
            f"╔══════════════════════════════════════════╗",
            f"║  Netlist: {self.name:<30s} ║",
            f"╠══════════════════════════════════════════╣",
            f"║  Cells:           {self.num_cells:<22d} ║",
            f"║  Nets:            {self.num_nets:<22d} ║",
            f"║  Total Area:      {self.total_cell_area:<22.1f} ║",
            f"║  Total Power:     {self.total_power:<22.4f} ║",
            f"║  Avg Net Degree:  {self.avg_net_degree():<22.2f} ║",
            f"║  Movable Cells:   {len(self.movable_cells):<22d} ║",
            f"║  Fixed Cells:     {len(self.fixed_cells):<22d} ║",
            f"╚══════════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Netlist('{self.name}', cells={self.num_cells}, nets={self.num_nets})"
