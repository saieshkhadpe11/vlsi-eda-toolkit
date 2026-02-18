"""
Unit tests for the VLSI EDA Toolkit.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np

from core.cell import Cell, CellType, Pin, PinDirection
from core.net import Net
from core.netlist import Netlist
from core.floorplan import Floorplan
from core.design import Design
from eda_parser.benchmark_parser import BenchmarkParser


class TestCell:
    """Tests for Cell class."""

    def test_cell_creation(self):
        cell = Cell("test", width=100, height=50)
        assert cell.name == "test"
        assert cell.width == 100
        assert cell.height == 50
        assert cell.area == 5000

    def test_cell_center(self):
        cell = Cell("c1", width=100, height=50, x=10, y=20)
        cx, cy = cell.center
        assert cx == 60.0
        assert cy == 45.0

    def test_cell_overlap(self):
        c1 = Cell("a", width=100, height=100, x=0, y=0)
        c2 = Cell("b", width=100, height=100, x=50, y=50)
        assert c1.overlaps(c2)
        assert c1.overlap_area(c2) == 2500.0

    def test_no_overlap(self):
        c1 = Cell("a", width=50, height=50, x=0, y=0)
        c2 = Cell("b", width=50, height=50, x=100, y=100)
        assert not c1.overlaps(c2)
        assert c1.overlap_area(c2) == 0.0

    def test_cell_move(self):
        cell = Cell("c1", width=50, height=50, x=0, y=0)
        cell.move(10, 20)
        assert cell.x == 10
        assert cell.y == 20

    def test_fixed_cell_no_move(self):
        cell = Cell("c1", width=50, height=50, x=0, y=0, fixed=True)
        cell.move(10, 20)
        assert cell.x == 0
        assert cell.y == 0

    def test_soft_macro_reshape(self):
        cell = Cell("soft", width=100, height=100,
                     cell_type=CellType.SOFT_MACRO)
        original_area = cell.area
        cell.reshape(2.0)
        assert abs(cell.area - original_area) < 1.0


class TestNet:
    """Tests for Net class."""

    def test_net_hpwl(self):
        net = Net("n1")
        net.add_pin("c1", "out")
        net.add_pin("c2", "in")

        positions = {"c1": (0, 0), "c2": (100, 200)}
        hpwl = net.hpwl(positions)
        assert hpwl == 300.0

    def test_net_hpwl_weighted(self):
        net = Net("n1", weight=2.0)
        net.add_pin("c1", "out")
        net.add_pin("c2", "in")

        positions = {"c1": (0, 0), "c2": (100, 200)}
        hpwl = net.hpwl(positions)
        assert hpwl == 600.0

    def test_net_degree(self):
        net = Net("n1")
        net.add_pin("c1", "out")
        net.add_pin("c2", "in")
        net.add_pin("c3", "in")
        assert net.degree == 3


class TestNetlist:
    """Tests for Netlist class."""

    def test_total_hpwl(self):
        netlist = Netlist("test")
        c1 = Cell("c1", width=10, height=10, x=0, y=0)
        c2 = Cell("c2", width=10, height=10, x=100, y=100)
        netlist.add_cell(c1)
        netlist.add_cell(c2)

        net = Net("n1")
        net.add_pin("c1", "out")
        net.add_pin("c2", "in")
        netlist.add_net(net)

        hpwl = netlist.total_hpwl()
        assert hpwl > 0

    def test_connectivity(self):
        netlist = Netlist("test")
        netlist.add_cell(Cell("c1", 10, 10))
        netlist.add_cell(Cell("c2", 10, 10))

        net = Net("n1")
        net.add_pin("c1", "out")
        net.add_pin("c2", "in")
        netlist.add_net(net)

        connected = netlist.get_connected_cells("c1")
        assert "c2" in connected


class TestFloorplan:
    """Tests for Floorplan class."""

    def test_utilization(self):
        netlist = Netlist("test")
        netlist.add_cell(Cell("c1", width=100, height=100))
        fp = Floorplan(chip_width=200, chip_height=200, netlist=netlist)
        assert abs(fp.utilization - 0.25) < 0.01

    def test_boundary_violations(self):
        netlist = Netlist("test")
        cell = Cell("c1", width=100, height=100, x=-50, y=0)
        netlist.add_cell(cell)
        fp = Floorplan(chip_width=200, chip_height=200, netlist=netlist)
        violations = fp.boundary_violations()
        assert len(violations) > 0


class TestBenchmarkParser:
    """Tests for BenchmarkParser."""

    def test_random_generation(self):
        parser = BenchmarkParser()
        design = parser.generate_random(
            num_cells=10, num_nets=15, seed=42
        )
        assert design.netlist.num_cells > 10  # cells + pads
        assert design.netlist.num_nets == 15

    def test_json_parsing(self):
        """Test JSON benchmark parsing."""
        benchmark_path = os.path.join(
            os.path.dirname(__file__), '..', 'benchmarks', 'example_8block.json'
        )
        if os.path.exists(benchmark_path):
            parser = BenchmarkParser()
            design = parser.from_json(benchmark_path)
            assert design.netlist.num_cells == 12  # 8 blocks + 4 pads
            assert design.netlist.num_nets == 14


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
