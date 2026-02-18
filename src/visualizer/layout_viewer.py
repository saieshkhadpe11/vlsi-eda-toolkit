"""
Layout Viewer: Professional VLSI floorplan and routing visualization.

Generates publication-quality plots of:
  - Floorplan layout with cell labels and dimensions
  - Thermal heatmaps overlaid on floorplan
  - Routing congestion maps
  - Net connectivity visualization
  - Optimization convergence plots
  - Timing slack distribution
  - Multi-panel design summary dashboards
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

from core.cell import Cell, CellType
from core.netlist import Netlist
from core.floorplan import Floorplan
from core.design import Design


# ── Color Palettes ────────────────────────────────────────────────────

CELL_COLORS = {
    CellType.HARD_MACRO: '#4A90D9',
    CellType.SOFT_MACRO: '#7B68EE',
    CellType.STANDARD_CELL: '#50C878',
    CellType.IO_PAD: '#FF6B6B',
    CellType.FIXED: '#FFD93D',
}

THERMAL_CMAP = LinearSegmentedColormap.from_list(
    'thermal', ['#1a1a2e', '#16213e', '#0f3460', '#e94560', '#ff6b6b', '#ffd93d']
)

DARK_BG = '#0d1117'
DARK_GRID = '#21262d'
DARK_TEXT = '#c9d1d9'
ACCENT = '#58a6ff'


class LayoutViewer:
    """
    Professional VLSI layout visualization engine.

    Generates high-quality plots suitable for papers,
    presentations, and portfolio display.
    """

    def __init__(self, design: Design, style: str = 'dark'):
        self.design = design
        self.floorplan = design.floorplan
        self.netlist = design.netlist
        self.style = style

        if style == 'dark':
            plt.rcParams.update({
                'figure.facecolor': DARK_BG,
                'axes.facecolor': DARK_BG,
                'axes.edgecolor': DARK_GRID,
                'text.color': DARK_TEXT,
                'xtick.color': DARK_TEXT,
                'ytick.color': DARK_TEXT,
                'axes.labelcolor': DARK_TEXT,
                'font.family': 'sans-serif',
                'font.size': 10,
            })

    def plot_floorplan(self, save_path: str = 'floorplan.png',
                       show_labels: bool = True,
                       show_nets: bool = False,
                       figsize: tuple = (12, 10),
                       dpi: int = 150) -> None:
        """Plot the floorplan layout with colored cells."""
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        # Chip outline
        chip_rect = patches.Rectangle(
            (0, 0), self.floorplan.chip_width, self.floorplan.chip_height,
            linewidth=2, edgecolor=ACCENT, facecolor='none',
            linestyle='--', alpha=0.7
        )
        ax.add_patch(chip_rect)

        # Draw cells
        for cell in self.netlist.cells.values():
            color = CELL_COLORS.get(cell.cell_type, '#4A90D9')
            alpha = 0.85 if not cell.fixed else 0.6

            rect = patches.FancyBboxPatch(
                (cell.x, cell.y), cell.width, cell.height,
                boxstyle="round,pad=1",
                linewidth=1.5, edgecolor='white',
                facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)

            if show_labels and cell.area > 0:
                font_size = max(5, min(9, cell.width / 15))
                ax.text(
                    cell.x + cell.width / 2, cell.y + cell.height / 2,
                    cell.name, ha='center', va='center',
                    fontsize=font_size, color='white', fontweight='bold',
                    alpha=0.9
                )

        # Draw nets
        if show_nets:
            self._draw_nets(ax)

        # Legend
        legend_elements = [
            patches.Patch(facecolor=c, label=t.name.replace('_', ' ').title())
            for t, c in CELL_COLORS.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                  fontsize=8, framealpha=0.8,
                  facecolor=DARK_BG, edgecolor=DARK_GRID)

        # Formatting
        margin = max(self.floorplan.chip_width, self.floorplan.chip_height) * 0.05
        ax.set_xlim(-margin, self.floorplan.chip_width + margin)
        ax.set_ylim(-margin, self.floorplan.chip_height + margin)
        ax.set_aspect('equal')
        ax.set_xlabel('X (μm)', fontsize=11)
        ax.set_ylabel('Y (μm)', fontsize=11)
        ax.set_title('VLSI Floorplan Layout', fontsize=14, fontweight='bold',
                      color=ACCENT, pad=15)
        ax.grid(True, alpha=0.1, color=DARK_GRID)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved floorplan → {save_path}")

    def plot_thermal_map(self, save_path: str = 'thermal.png',
                         grid_resolution: int = 50,
                         figsize: tuple = (12, 10),
                         dpi: int = 150) -> None:
        """Plot thermal heatmap based on cell power densities."""
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        # Create thermal grid
        thermal = np.zeros((grid_resolution, grid_resolution))

        for cell in self.netlist.cells.values():
            cx, cy = cell.center
            gi = int(cx / self.floorplan.chip_width * (grid_resolution - 1))
            gj = int(cy / self.floorplan.chip_height * (grid_resolution - 1))
            gi = np.clip(gi, 0, grid_resolution - 1)
            gj = np.clip(gj, 0, grid_resolution - 1)

            # Gaussian heat spread
            power = cell.total_power
            sigma = max(cell.width, cell.height) / self.floorplan.chip_width * grid_resolution
            sigma = max(sigma, 1.5)

            for di in range(-int(3 * sigma), int(3 * sigma) + 1):
                for dj in range(-int(3 * sigma), int(3 * sigma) + 1):
                    ni, nj = gi + di, gj + dj
                    if 0 <= ni < grid_resolution and 0 <= nj < grid_resolution:
                        dist2 = di ** 2 + dj ** 2
                        thermal[nj][ni] += power * np.exp(-dist2 / (2 * sigma ** 2))

        im = ax.imshow(
            thermal, extent=[0, self.floorplan.chip_width, 0, self.floorplan.chip_height],
            origin='lower', cmap=THERMAL_CMAP, interpolation='bicubic',
            aspect='equal'
        )

        # Overlay cell outlines
        for cell in self.netlist.cells.values():
            rect = patches.Rectangle(
                (cell.x, cell.y), cell.width, cell.height,
                linewidth=0.8, edgecolor='white', facecolor='none', alpha=0.4
            )
            ax.add_patch(rect)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Power Density (W/μm²)', fontsize=10)

        ax.set_xlabel('X (μm)', fontsize=11)
        ax.set_ylabel('Y (μm)', fontsize=11)
        ax.set_title('Thermal Heatmap', fontsize=14, fontweight='bold',
                      color='#ff6b6b', pad=15)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved thermal map → {save_path}")

    def plot_congestion(self, congestion_map: np.ndarray,
                        save_path: str = 'congestion.png',
                        figsize: tuple = (12, 10),
                        dpi: int = 150) -> None:
        """Plot routing congestion heatmap."""
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        cmap = LinearSegmentedColormap.from_list(
            'congestion', ['#1a1a2e', '#2d6a4f', '#52b788', '#ffd93d', '#e94560']
        )

        im = ax.imshow(
            congestion_map,
            extent=[0, self.floorplan.chip_width, 0, self.floorplan.chip_height],
            origin='lower', cmap=cmap, interpolation='nearest',
            aspect='equal', vmin=0, vmax=max(1.5, congestion_map.max())
        )

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Congestion Ratio', fontsize=10)

        ax.set_xlabel('X (μm)', fontsize=11)
        ax.set_ylabel('Y (μm)', fontsize=11)
        ax.set_title('Routing Congestion Map', fontsize=14, fontweight='bold',
                      color='#52b788', pad=15)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved congestion map → {save_path}")

    def plot_convergence(self, cost_history: list[float],
                         save_path: str = 'convergence.png',
                         title: str = 'Optimization Convergence',
                         temperature_history: list[float] | None = None,
                         figsize: tuple = (12, 5),
                         dpi: int = 150) -> None:
        """Plot optimization convergence curve."""
        fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        # Cost curve
        x = range(len(cost_history))
        ax1.plot(x, cost_history, color=ACCENT, linewidth=1.5,
                 alpha=0.9, label='Cost')

        # Running minimum
        running_min = np.minimum.accumulate(cost_history)
        ax1.plot(x, running_min, color='#ff6b6b', linewidth=2,
                 linestyle='--', label='Best Cost', alpha=0.8)

        ax1.set_xlabel('Iteration (×moves_per_temp)', fontsize=11)
        ax1.set_ylabel('Cost', fontsize=11, color=ACCENT)
        ax1.tick_params(axis='y', labelcolor=ACCENT)

        # Temperature curve (secondary axis)
        if temperature_history:
            ax2 = ax1.twinx()
            ax2.plot(range(len(temperature_history)), temperature_history,
                     color='#ffd93d', linewidth=1, alpha=0.5, label='Temperature')
            ax2.set_ylabel('Temperature', fontsize=11, color='#ffd93d')
            ax2.tick_params(axis='y', labelcolor='#ffd93d')
            ax2.set_yscale('log')

        ax1.set_title(title, fontsize=14, fontweight='bold', color=ACCENT, pad=15)
        ax1.legend(loc='upper right', fontsize=9,
                   facecolor=DARK_BG, edgecolor=DARK_GRID)
        ax1.grid(True, alpha=0.1)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved convergence plot → {save_path}")

    def plot_dashboard(self, save_path: str = 'dashboard.png',
                       cost_history: list[float] | None = None,
                       congestion_map: np.ndarray | None = None,
                       figsize: tuple = (20, 14),
                       dpi: int = 150) -> None:
        """Generate a multi-panel design summary dashboard."""
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

        # Panel 1: Floorplan
        ax1 = fig.add_subplot(gs[0, 0])
        self._draw_floorplan_on_ax(ax1)
        ax1.set_title('Floorplan', fontsize=12, fontweight='bold', color=ACCENT)

        # Panel 2: Thermal
        ax2 = fig.add_subplot(gs[0, 1])
        self._draw_thermal_on_ax(ax2)
        ax2.set_title('Thermal Map', fontsize=12, fontweight='bold', color='#ff6b6b')

        # Panel 3: Metrics Text
        ax3 = fig.add_subplot(gs[0, 2])
        self._draw_metrics_text(ax3)
        ax3.set_title('Design Metrics', fontsize=12, fontweight='bold', color=ACCENT)

        # Panel 4: Convergence
        ax4 = fig.add_subplot(gs[1, 0])
        if cost_history:
            ax4.plot(cost_history, color=ACCENT, linewidth=1)
            running_min = np.minimum.accumulate(cost_history)
            ax4.plot(running_min, color='#ff6b6b', linewidth=1.5, linestyle='--')
            ax4.set_title('Convergence', fontsize=12, fontweight='bold', color=ACCENT)
            ax4.grid(True, alpha=0.1)
        else:
            ax4.text(0.5, 0.5, 'No optimization data', ha='center', va='center',
                     fontsize=11, color=DARK_TEXT)
            ax4.set_title('Convergence', fontsize=12, fontweight='bold', color=ACCENT)

        # Panel 5: Congestion
        ax5 = fig.add_subplot(gs[1, 1])
        if congestion_map is not None:
            im = ax5.imshow(congestion_map, origin='lower', cmap='YlOrRd',
                            interpolation='nearest', aspect='equal')
            plt.colorbar(im, ax=ax5, shrink=0.8)
            ax5.set_title('Congestion', fontsize=12, fontweight='bold', color='#52b788')
        else:
            ax5.text(0.5, 0.5, 'No routing data', ha='center', va='center',
                     fontsize=11, color=DARK_TEXT)
            ax5.set_title('Congestion', fontsize=12, fontweight='bold', color='#52b788')

        # Panel 6: Cell Area Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        areas = [c.area for c in self.netlist.cells.values() if c.area > 0]
        if areas:
            ax6.hist(areas, bins=15, color=ACCENT, alpha=0.7, edgecolor='white')
            ax6.set_xlabel('Cell Area (μm²)', fontsize=9)
            ax6.set_ylabel('Count', fontsize=9)
        ax6.set_title('Area Distribution', fontsize=12, fontweight='bold', color=ACCENT)
        ax6.grid(True, alpha=0.1)

        fig.suptitle(f'VLSI Design Dashboard — {self.design.name}',
                     fontsize=16, fontweight='bold', color=ACCENT, y=0.98)

        plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved dashboard → {save_path}")

    # ── Helper Methods ────────────────────────────────────────────────

    def _draw_nets(self, ax) -> None:
        """Draw net connections as lines between cell centers."""
        lines = []
        for net in self.netlist.nets.values():
            cell_names = list(net.get_cell_names())
            for i in range(len(cell_names) - 1):
                c1 = self.netlist.cells.get(cell_names[i])
                c2 = self.netlist.cells.get(cell_names[i + 1])
                if c1 and c2:
                    lines.append([c1.center, c2.center])

        if lines:
            lc = LineCollection(lines, colors=ACCENT, linewidths=0.5, alpha=0.3)
            ax.add_collection(lc)

    def _draw_floorplan_on_ax(self, ax) -> None:
        """Draw floorplan on a given axes (for dashboard)."""
        chip_rect = patches.Rectangle(
            (0, 0), self.floorplan.chip_width, self.floorplan.chip_height,
            linewidth=1.5, edgecolor=ACCENT, facecolor='none', linestyle='--', alpha=0.5
        )
        ax.add_patch(chip_rect)

        for cell in self.netlist.cells.values():
            color = CELL_COLORS.get(cell.cell_type, '#4A90D9')
            rect = patches.Rectangle(
                (cell.x, cell.y), cell.width, cell.height,
                linewidth=0.8, edgecolor='white', facecolor=color, alpha=0.75
            )
            ax.add_patch(rect)

        ax.set_xlim(-10, self.floorplan.chip_width + 10)
        ax.set_ylim(-10, self.floorplan.chip_height + 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.05)

    def _draw_thermal_on_ax(self, ax, resolution: int = 30) -> None:
        """Draw thermal map on a given axes."""
        thermal = np.zeros((resolution, resolution))
        for cell in self.netlist.cells.values():
            cx, cy = cell.center
            gi = int(cx / self.floorplan.chip_width * (resolution - 1))
            gj = int(cy / self.floorplan.chip_height * (resolution - 1))
            gi = np.clip(gi, 0, resolution - 1)
            gj = np.clip(gj, 0, resolution - 1)
            power = cell.total_power
            sigma = max(1.5, max(cell.width, cell.height) /
                        self.floorplan.chip_width * resolution)
            for di in range(-int(2 * sigma), int(2 * sigma) + 1):
                for dj in range(-int(2 * sigma), int(2 * sigma) + 1):
                    ni, nj = gi + di, gj + dj
                    if 0 <= ni < resolution and 0 <= nj < resolution:
                        thermal[nj][ni] += power * np.exp(
                            -(di**2 + dj**2) / (2 * sigma**2))

        ax.imshow(thermal, origin='lower', cmap=THERMAL_CMAP,
                  interpolation='bicubic', aspect='equal')

    def _draw_metrics_text(self, ax) -> None:
        """Draw design metrics as formatted text."""
        ax.axis('off')
        fp = self.floorplan
        nl = self.netlist

        metrics = [
            ("Chip Size", f"{fp.chip_width:.0f} × {fp.chip_height:.0f} μm"),
            ("Total Cells", f"{nl.num_cells}"),
            ("Total Nets", f"{nl.num_nets}"),
            ("Cell Area", f"{nl.total_cell_area:.0f} μm²"),
            ("Utilization", f"{fp.utilization * 100:.1f}%"),
            ("Total HPWL", f"{fp.total_hpwl():.1f} μm"),
            ("Overlap Area", f"{fp.total_overlap_area():.1f} μm²"),
            ("Total Power", f"{nl.total_power:.4f} W"),
            ("Quality Score", f"{fp.quality_score():.4f}"),
        ]

        y_pos = 0.92
        for label, value in metrics:
            ax.text(0.05, y_pos, label + ":", fontsize=10, fontweight='bold',
                    color=ACCENT, transform=ax.transAxes)
            ax.text(0.6, y_pos, value, fontsize=10, color=DARK_TEXT,
                    transform=ax.transAxes)
            y_pos -= 0.1
