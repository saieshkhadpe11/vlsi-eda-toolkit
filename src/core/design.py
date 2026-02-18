"""
Design: top-level container for the entire VLSI design flow.

Aggregates netlist, floorplan, routing, and timing information into a
single design object that flows through the EDA pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from .netlist import Netlist
from .floorplan import Floorplan


@dataclass
class DesignMetrics:
    """Snapshot of design quality metrics at a point in the flow."""
    timestamp: str = ""
    stage: str = ""
    total_hpwl: float = 0.0
    total_overlap: float = 0.0
    utilization: float = 0.0
    quality_score: float = 0.0
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "stage": self.stage,
            "total_hpwl": self.total_hpwl,
            "total_overlap": self.total_overlap,
            "utilization": self.utilization,
            "quality_score": self.quality_score,
            "runtime_seconds": self.runtime_seconds,
        }


@dataclass
class Design:
    """
    Top-level design container for the VLSI EDA flow.

    Holds all design data and tracks metrics through each stage
    of the physical design flow (parsing → floorplanning → routing → timing).

    Attributes:
        name: Design name.
        netlist: The design netlist.
        floorplan: The current floorplan solution.
        metrics_history: Record of metrics at each stage.
    """
    name: str = "untitled"
    netlist: Optional[Netlist] = None
    floorplan: Optional[Floorplan] = None
    metrics_history: list[DesignMetrics] = field(default_factory=list)

    def create_floorplan(self, chip_width: float, chip_height: float) -> Floorplan:
        """Create a floorplan with the given chip dimensions."""
        self.floorplan = Floorplan(
            chip_width=chip_width,
            chip_height=chip_height,
            netlist=self.netlist,
        )
        return self.floorplan

    def snapshot_metrics(self, stage: str, runtime: float = 0.0) -> DesignMetrics:
        """Take a snapshot of current design metrics."""
        metrics = DesignMetrics(
            timestamp=datetime.now().isoformat(),
            stage=stage,
            total_hpwl=self.floorplan.total_hpwl() if self.floorplan else 0.0,
            total_overlap=self.floorplan.total_overlap_area() if self.floorplan else 0.0,
            utilization=self.floorplan.utilization if self.floorplan else 0.0,
            quality_score=self.floorplan.quality_score() if self.floorplan else 0.0,
            runtime_seconds=runtime,
        )
        self.metrics_history.append(metrics)
        return metrics

    def summary(self) -> str:
        """Full design summary."""
        parts = [f"═══ Design: {self.name} ═══"]
        if self.netlist:
            parts.append(self.netlist.summary())
        if self.floorplan:
            parts.append(self.floorplan.summary())
        if self.metrics_history:
            parts.append("\n── Metrics History ──")
            for m in self.metrics_history:
                parts.append(f"  [{m.stage}] HPWL={m.total_hpwl:.1f}, "
                             f"Overlap={m.total_overlap:.1f}, "
                             f"Util={m.utilization:.1%}, "
                             f"Time={m.runtime_seconds:.2f}s")
        return "\n".join(parts)

    def __repr__(self) -> str:
        return f"Design('{self.name}')"
