"""
Benchmark parser for VLSI floorplanning benchmarks.

Supports:
  - MCNC-style .blocks / .nets format
  - YAL format
  - Custom JSON format for easy experimentation
  - Random benchmark generation for testing

The parser produces a Netlist object that can be directly used by
the floorplanner and other downstream tools.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional
import numpy as np

from core.cell import Cell, CellType, Pin, PinDirection
from core.net import Net
from core.netlist import Netlist
from core.design import Design


class BenchmarkParser:
    """
    Unified parser for multiple VLSI benchmark formats.

    Usage:
        parser = BenchmarkParser()
        design = parser.parse("ami33.blocks", "ami33.nets")
        # or
        design = parser.from_json("my_design.json")
        # or
        design = parser.generate_random(num_cells=25, num_nets=40)
    """

    def parse(self, blocks_file: str, nets_file: str,
              design_name: Optional[str] = None) -> Design:
        """
        Parse MCNC-style .blocks and .nets files.

        .blocks format:
            NumSoftRectangularBlocks : <N>
            NumHardRectilinearBlocks : <M>
            NumTerminals : <T>
            <block_name> hardrectilinear <N_vertices> (<x1>, <y1>) ... (<xN>, <yN>)
            <terminal_name> terminal

        .nets format:
            NumNets : <N>
            NetDegree : <D> <net_name>
            <block_name> B
            ...
        """
        if design_name is None:
            design_name = Path(blocks_file).stem

        design = Design(name=design_name)
        netlist = Netlist(name=design_name)

        # Parse blocks
        self._parse_blocks_file(blocks_file, netlist)

        # Parse nets
        self._parse_nets_file(nets_file, netlist)

        design.netlist = netlist
        return design

    def _parse_blocks_file(self, filepath: str, netlist: Netlist) -> None:
        """Parse MCNC .blocks file."""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('UMICH') or line.startswith('Num'):
                continue

            # Parse hard rectilinear block
            match = re.match(
                r'(\w+)\s+hardrectilinear\s+(\d+)\s+(.*)', line
            )
            if match:
                name = match.group(1)
                vertices_str = match.group(3)
                coords = re.findall(r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)', vertices_str)
                if coords:
                    xs = [float(c[0]) for c in coords]
                    ys = [float(c[1]) for c in coords]
                    width = max(xs) - min(xs)
                    height = max(ys) - min(ys)
                    cell = Cell(name=name, width=width, height=height,
                                cell_type=CellType.HARD_MACRO)
                    netlist.add_cell(cell)
                continue

            # Parse soft rectangular block
            match = re.match(
                r'(\w+)\s+softrectangular\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)',
                line
            )
            if match:
                name = match.group(1)
                area = float(match.group(2))
                min_ar = float(match.group(3))
                max_ar = float(match.group(4))
                width = np.sqrt(area * (min_ar + max_ar) / 2)
                height = area / width
                cell = Cell(name=name, width=width, height=height,
                            cell_type=CellType.SOFT_MACRO)
                netlist.add_cell(cell)
                continue

            # Parse terminal
            match = re.match(r'(\w+)\s+terminal(?:\s+.*)?', line)
            if match:
                name = match.group(1)
                cell = Cell(name=name, width=0, height=0,
                            cell_type=CellType.IO_PAD, fixed=True)
                netlist.add_cell(cell)

    def _parse_nets_file(self, filepath: str, netlist: Netlist) -> None:
        """Parse MCNC .nets file."""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        current_net = None
        net_counter = 0

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('UCLA') or line.startswith('Num'):
                continue

            # NetDegree line
            match = re.match(r'NetDegree\s*:\s*(\d+)\s*(\w*)', line)
            if match:
                degree = int(match.group(1))
                net_name = match.group(2) if match.group(2) else f"n{net_counter}"
                current_net = Net(name=net_name)
                netlist.add_net(current_net)
                net_counter += 1
                continue

            # Pin line
            if current_net is not None:
                parts = line.split()
                if parts:
                    cell_name = parts[0]
                    pin_dir = parts[1] if len(parts) > 1 else "B"
                    current_net.add_pin(cell_name, "default")

    def from_json(self, filepath: str) -> Design:
        """
        Parse a JSON format design file.

        JSON schema:
        {
            "name": "design_name",
            "chip_width": 1000,
            "chip_height": 1000,
            "cells": [
                {"name": "c1", "width": 100, "height": 50, "type": "hard"},
                ...
            ],
            "nets": [
                {"name": "n1", "pins": [["c1", "out"], ["c2", "in"]]},
                ...
            ]
        }
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        design = Design(name=data.get("name", "json_design"))
        netlist = Netlist(name=design.name)

        # Parse cells
        type_map = {
            "hard": CellType.HARD_MACRO,
            "soft": CellType.SOFT_MACRO,
            "stdcell": CellType.STANDARD_CELL,
            "io": CellType.IO_PAD,
            "fixed": CellType.FIXED,
        }

        for cell_data in data.get("cells", []):
            cell = Cell(
                name=cell_data["name"],
                width=float(cell_data["width"]),
                height=float(cell_data["height"]),
                cell_type=type_map.get(cell_data.get("type", "hard"), CellType.HARD_MACRO),
                fixed=cell_data.get("fixed", False),
                power_density=cell_data.get("power_density", 0.001),
            )
            if "x" in cell_data and "y" in cell_data:
                cell.place(float(cell_data["x"]), float(cell_data["y"]))
            netlist.add_cell(cell)

        # Parse nets
        for net_data in data.get("nets", []):
            net = Net(
                name=net_data["name"],
                weight=net_data.get("weight", 1.0),
                clock=net_data.get("clock", False),
            )
            for pin_ref in net_data.get("pins", []):
                if isinstance(pin_ref, list) and len(pin_ref) >= 2:
                    net.add_pin(pin_ref[0], pin_ref[1])
                elif isinstance(pin_ref, str):
                    net.add_pin(pin_ref, "default")
            netlist.add_net(net)

        design.netlist = netlist

        # Create floorplan if chip dimensions given
        if "chip_width" in data and "chip_height" in data:
            design.create_floorplan(
                float(data["chip_width"]),
                float(data["chip_height"])
            )

        return design

    def generate_random(self, num_cells: int = 20, num_nets: int = 30,
                        chip_size: float = 1000.0,
                        min_cell_size: float = 30.0,
                        max_cell_size: float = 200.0,
                        avg_net_degree: int = 3,
                        seed: Optional[int] = None) -> Design:
        """
        Generate a random benchmark for testing.

        Creates a random set of cells and nets with configurable parameters.
        Useful for algorithm development and testing.

        Args:
            num_cells: Number of cells to generate.
            num_nets: Number of nets to generate.
            chip_size: Chip dimension (square die).
            min_cell_size: Minimum cell dimension.
            max_cell_size: Maximum cell dimension.
            avg_net_degree: Average pins per net.
            seed: Random seed for reproducibility.

        Returns:
            A Design with randomly generated netlist.
        """
        rng = np.random.default_rng(seed)

        design = Design(name=f"random_{num_cells}c_{num_nets}n")
        netlist = Netlist(name=design.name)

        # Generate cells with varied sizes
        cell_names = []
        for i in range(num_cells):
            width = rng.uniform(min_cell_size, max_cell_size)
            height = rng.uniform(min_cell_size, max_cell_size)

            # Mix of hard and soft macros
            cell_type = CellType.SOFT_MACRO if rng.random() < 0.3 else CellType.HARD_MACRO
            power = rng.uniform(0.0005, 0.005)

            cell = Cell(
                name=f"block_{i:03d}",
                width=round(width, 1),
                height=round(height, 1),
                cell_type=cell_type,
                power_density=round(power, 5),
            )
            netlist.add_cell(cell)
            cell_names.append(cell.name)

        # Add a few I/O pads
        num_pads = max(4, num_cells // 5)
        pad_positions = [
            (0, chip_size * i / (num_pads // 4 + 1))
            for i in range(1, num_pads // 4 + 1)
        ] + [
            (chip_size, chip_size * i / (num_pads // 4 + 1))
            for i in range(1, num_pads // 4 + 1)
        ] + [
            (chip_size * i / (num_pads // 4 + 1), 0)
            for i in range(1, num_pads // 4 + 1)
        ] + [
            (chip_size * i / (num_pads // 4 + 1), chip_size)
            for i in range(1, num_pads // 4 + 1)
        ]

        for i, (px, py) in enumerate(pad_positions[:num_pads]):
            pad = Cell(
                name=f"pad_{i:02d}",
                width=5.0, height=5.0,
                cell_type=CellType.IO_PAD,
                x=px, y=py,
                fixed=True,
            )
            netlist.add_cell(pad)
            cell_names.append(pad.name)

        # Generate nets
        for i in range(num_nets):
            degree = max(2, int(rng.poisson(avg_net_degree - 1) + 1))
            degree = min(degree, len(cell_names))

            net = Net(
                name=f"net_{i:03d}",
                weight=round(rng.uniform(0.5, 2.0), 2),
            )

            # Pick random cells for this net (no duplicates)
            connected = rng.choice(len(cell_names), size=degree, replace=False)
            for idx in connected:
                net.add_pin(cell_names[idx], "default")

            netlist.add_net(net)

        design.netlist = netlist
        design.create_floorplan(chip_size, chip_size)
        return design


# ── Convenience Functions ─────────────────────────────────────────────

def parse_yal(filepath: str) -> Design:
    """Parse a YAL format file (Yale format)."""
    parser = BenchmarkParser()
    design = Design(name=Path(filepath).stem)
    netlist = Netlist(name=design.name)

    with open(filepath, 'r') as f:
        content = f.read()

    # Simple YAL parsing: MODULE definitions
    modules = re.findall(
        r'MODULE\s+(\w+).*?DIMENSIONS\s+([\d.]+)\s+([\d.]+)',
        content, re.DOTALL
    )

    for name, w, h in modules:
        cell = Cell(name=name, width=float(w), height=float(h))
        netlist.add_cell(cell)

    design.netlist = netlist
    return design


def parse_mcnc_blocks(blocks_path: str, nets_path: str) -> Design:
    """Convenience function to parse MCNC benchmark files."""
    parser = BenchmarkParser()
    return parser.parse(blocks_path, nets_path)
