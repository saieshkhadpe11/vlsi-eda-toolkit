"""
Static Timing Analysis (STA) Engine.

Implements a graph-based static timing analyzer that computes:
  - Arrival times (AT) at all nodes
  - Required times (RT) at all nodes
  - Slack at each node (RT - AT)
  - Critical paths (most negative slack)

The timing graph is a DAG where:
  - Nodes represent cell pins
  - Edges represent either cell delays (internal) or wire delays (interconnect)

Delay model:
  - Cell delay: configurable per cell type
  - Wire delay: proportional to HPWL (Elmore delay approximation)

This provides a simplified but structurally accurate STA suitable for
floorplan-level timing estimation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import networkx as nx

from core.netlist import Netlist
from core.design import Design


@dataclass
class TimingNode:
    """Node in the timing graph."""
    name: str                    # "cell_name/pin_name"
    cell_name: str
    pin_name: str
    arrival_time: float = 0.0   # Latest arrival time
    required_time: float = float('inf')  # Earliest required time
    is_input: bool = False       # Primary input
    is_output: bool = False      # Primary output

    @property
    def slack(self) -> float:
        """Timing slack (positive = met, negative = violated)."""
        return self.required_time - self.arrival_time


@dataclass
class TimingEdge:
    """Edge in the timing graph (delay element)."""
    source: str
    sink: str
    delay: float
    edge_type: str = "wire"  # "wire" or "cell"


@dataclass
class TimingPath:
    """A timing path from input to output."""
    nodes: list[str] = field(default_factory=list)
    delays: list[float] = field(default_factory=list)
    total_delay: float = 0.0
    slack: float = 0.0

    @property
    def num_stages(self) -> int:
        return len(self.delays)


@dataclass
class TimingResult:
    """Result of the STA analysis."""
    wns: float = 0.0                    # Worst Negative Slack
    tns: float = 0.0                    # Total Negative Slack
    num_violations: int = 0             # Number of timing violations
    clock_period: float = 0.0           # Target clock period
    critical_path: Optional[TimingPath] = None
    all_paths: list[TimingPath] = field(default_factory=list)
    node_slacks: dict[str, float] = field(default_factory=dict)
    runtime_seconds: float = 0.0

    @property
    def is_timing_clean(self) -> bool:
        return self.wns >= 0

    def summary(self) -> str:
        status = "✓ PASS" if self.is_timing_clean else "✗ FAIL"
        lines = [
            f"┌──────────────────────────────────────────┐",
            f"│  Static Timing Analysis Results          │",
            f"├──────────────────────────────────────────┤",
            f"│  Status:      {status}" +
            " " * max(0, 24 - len(status)) + "│",
            f"│  Clock:       {self.clock_period:.2f} ns" +
            " " * max(0, 22 - len(f"{self.clock_period:.2f} ns")) + "│",
            f"│  WNS:         {self.wns:.3f} ns" +
            " " * max(0, 22 - len(f"{self.wns:.3f} ns")) + "│",
            f"│  TNS:         {self.tns:.3f} ns" +
            " " * max(0, 22 - len(f"{self.tns:.3f} ns")) + "│",
            f"│  Violations:  {self.num_violations}" +
            " " * max(0, 24 - len(f"{self.num_violations}")) + "│",
            f"│  Paths:       {len(self.all_paths)}" +
            " " * max(0, 24 - len(f"{len(self.all_paths)}")) + "│",
            f"└──────────────────────────────────────────┘",
        ]
        return "\n".join(lines)


class STAEngine:
    """
    Graph-based Static Timing Analysis engine.

    Builds a timing DAG from the netlist connectivity and computes
    arrival times, required times, and slack at each node.

    Usage:
        sta = STAEngine(design, clock_period=10.0)
        result = sta.analyze()
        print(result.summary())
    """

    def __init__(self, design: Design, clock_period: float = 10.0,
                 wire_delay_factor: float = 0.01,
                 cell_delay_map: dict[str, float] | None = None):
        """
        Args:
            design: The Design object with placed netlist.
            clock_period: Target clock period in ns.
            wire_delay_factor: Delay per unit wirelength (ns/μm).
            cell_delay_map: Custom per-cell delays. If None, uses area-based model.
        """
        self.design = design
        self.netlist = design.netlist
        self.clock_period = clock_period
        self.wire_delay_factor = wire_delay_factor
        self.cell_delay_map = cell_delay_map or {}

        # Build timing graph
        self.graph = nx.DiGraph()
        self.nodes: dict[str, TimingNode] = {}
        self.edges: list[TimingEdge] = []

    def analyze(self, verbose: bool = True) -> TimingResult:
        """
        Run full STA: build graph → forward propagation → backward propagation.

        Returns:
            TimingResult with WNS, TNS, and critical paths.
        """
        import time
        start = time.time()

        self._build_timing_graph()
        self._forward_propagation()
        self._backward_propagation()
        result = self._extract_results()

        elapsed = time.time() - start
        result.runtime_seconds = elapsed
        result.clock_period = self.clock_period

        if verbose:
            print(result.summary())

        self.design.snapshot_metrics("sta", elapsed)
        return result

    def _build_timing_graph(self) -> None:
        """
        Build the timing DAG from netlist connectivity.

        Each cell gets an input and output node. Nets create edges
        between output nodes and input nodes of connected cells.
        """
        positions = self.netlist.get_cell_positions()

        # Create nodes for each cell (simplified: one input, one output per cell)
        for cell_name, cell in self.netlist.cells.items():
            in_node = f"{cell_name}/in"
            out_node = f"{cell_name}/out"

            self.nodes[in_node] = TimingNode(
                name=in_node, cell_name=cell_name, pin_name="in",
                is_input=(cell.fixed),  # I/O pads are primary inputs
            )
            self.nodes[out_node] = TimingNode(
                name=out_node, cell_name=cell_name, pin_name="out",
                is_output=(cell.fixed),
            )

            # Internal cell delay
            cell_delay = self._get_cell_delay(cell_name)
            self.graph.add_edge(in_node, out_node, delay=cell_delay)
            self.edges.append(TimingEdge(in_node, out_node, cell_delay, "cell"))

        # Create edges from nets
        for net in self.netlist.nets.values():
            cell_names = list(net.get_cell_names())
            if len(cell_names) < 2:
                continue

            # Assume first cell is driver, rest are sinks (simplified)
            driver = cell_names[0]
            driver_node = f"{driver}/out"

            for sink in cell_names[1:]:
                sink_node = f"{sink}/in"

                # Wire delay based on Manhattan distance
                if driver in positions and sink in positions:
                    dx = abs(positions[driver][0] - positions[sink][0])
                    dy = abs(positions[driver][1] - positions[sink][1])
                    wire_delay = (dx + dy) * self.wire_delay_factor
                else:
                    wire_delay = 0.1  # Default

                self.graph.add_edge(driver_node, sink_node, delay=wire_delay)
                self.edges.append(TimingEdge(driver_node, sink_node, wire_delay, "wire"))

    def _get_cell_delay(self, cell_name: str) -> float:
        """
        Get delay for a cell.

        Uses custom map if available, otherwise estimates from cell area
        (larger cells ≈ more complex logic ≈ higher delay).
        """
        if cell_name in self.cell_delay_map:
            return self.cell_delay_map[cell_name]

        cell = self.netlist.cells.get(cell_name)
        if cell is None:
            return 0.1

        # Area-based delay model: delay ∝ sqrt(area)
        delay = 0.05 * np.sqrt(cell.area / 100.0)
        return max(0.01, min(delay, 5.0))  # Clamp to reasonable range

    def _forward_propagation(self) -> None:
        """
        Forward pass: compute arrival times.

        Traverses the DAG in topological order, computing the latest
        arrival time at each node.
        """
        # Set primary input arrival times to 0
        for node in self.nodes.values():
            if node.is_input:
                node.arrival_time = 0.0
            else:
                node.arrival_time = 0.0

        # Topological traversal
        try:
            topo_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            # Graph has cycles — break them for analysis
            topo_order = list(self.nodes.keys())

        for node_name in topo_order:
            if node_name not in self.nodes:
                continue

            node = self.nodes[node_name]

            # Arrival time = max over all incoming edges of (source AT + delay)
            for pred in self.graph.predecessors(node_name):
                if pred in self.nodes:
                    edge_delay = self.graph[pred][node_name].get('delay', 0)
                    new_at = self.nodes[pred].arrival_time + edge_delay
                    node.arrival_time = max(node.arrival_time, new_at)

    def _backward_propagation(self) -> None:
        """
        Backward pass: compute required times.

        Traverses the DAG in reverse topological order, computing the
        earliest required time at each node.
        """
        # Set primary output required times to clock period
        for node in self.nodes.values():
            if node.is_output:
                node.required_time = self.clock_period
            else:
                node.required_time = float('inf')

        # Reverse topological traversal
        try:
            topo_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            topo_order = list(self.nodes.keys())

        for node_name in reversed(topo_order):
            if node_name not in self.nodes:
                continue

            node = self.nodes[node_name]

            # Required time = min over all outgoing edges of (sink RT - delay)
            for succ in self.graph.successors(node_name):
                if succ in self.nodes:
                    edge_delay = self.graph[node_name][succ].get('delay', 0)
                    new_rt = self.nodes[succ].required_time - edge_delay
                    node.required_time = min(node.required_time, new_rt)

    def _extract_results(self) -> TimingResult:
        """Extract timing metrics from analyzed graph."""
        result = TimingResult()
        result.clock_period = self.clock_period

        # Compute slack for each node
        wns = float('inf')
        tns = 0.0
        violations = 0

        for node_name, node in self.nodes.items():
            if node.required_time == float('inf'):
                continue

            slack = node.slack
            result.node_slacks[node_name] = slack

            if slack < wns:
                wns = slack

            if slack < 0:
                tns += slack
                violations += 1

        result.wns = wns if wns != float('inf') else 0.0
        result.tns = tns
        result.num_violations = violations

        # Find critical path
        critical_path = self._trace_critical_path()
        if critical_path:
            result.critical_path = critical_path
            result.all_paths = [critical_path]

        return result

    def _trace_critical_path(self) -> Optional[TimingPath]:
        """Trace the most critical path through the design."""
        # Find the node with worst slack
        worst_node = None
        worst_slack = float('inf')

        for name, node in self.nodes.items():
            if node.required_time < float('inf') and node.slack < worst_slack:
                worst_slack = node.slack
                worst_node = name

        if worst_node is None:
            return None

        # Trace backward from worst node to a primary input
        path_nodes = [worst_node]
        path_delays = []
        current = worst_node

        visited = set()
        while current and current not in visited:
            visited.add(current)
            preds = list(self.graph.predecessors(current))
            if not preds:
                break

            # Follow the predecessor with the latest arrival time
            best_pred = max(preds, key=lambda p: self.nodes[p].arrival_time
                          if p in self.nodes else 0)
            delay = self.graph[best_pred][current].get('delay', 0)
            path_nodes.insert(0, best_pred)
            path_delays.insert(0, delay)
            current = best_pred

        path = TimingPath(
            nodes=path_nodes,
            delays=path_delays,
            total_delay=sum(path_delays),
            slack=worst_slack,
        )
        return path
