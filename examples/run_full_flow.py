"""
VLSI EDA Toolkit — Full Design Flow Example
=============================================

This script demonstrates the complete physical design flow:
  1. Parse/Generate benchmark
  2. Run Simulated Annealing floorplanner
  3. Run PIAB-FP floorplanner (comparison)
  4. Global routing
  5. Static timing analysis
  6. Visualization (floorplan, thermal, congestion, dashboard)

Usage:
    cd vlsi-eda-toolkit
    python examples/run_full_flow.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import Design
from eda_parser.benchmark_parser import BenchmarkParser
from floorplanner.simulated_annealing import SimulatedAnnealingFloorplanner, SAConfig
from floorplanner.piab_fp import PIABFloorplanner, PIABConfig
from floorplanner.cost import CostWeights
from router.global_router import GlobalRouter
from timing.sta import STAEngine
from visualizer.layout_viewer import LayoutViewer


def main():
    print("=" * 60)
    print("  VLSI EDA Toolkit - Full Design Flow")
    print("=" * 60)

    # ── Step 1: Load Benchmark ────────────────────────────────────
    print("\n[PARSE] Step 1: Loading benchmark...")
    parser = BenchmarkParser()

    benchmark_path = os.path.join(
        os.path.dirname(__file__), '..', 'benchmarks', 'example_8block.json'
    )

    if os.path.exists(benchmark_path):
        design = parser.from_json(benchmark_path)
        print(f"   Loaded: {benchmark_path}")
    else:
        print("   Benchmark not found - generating random design...")
        design = parser.generate_random(
            num_cells=25, num_nets=40,
            chip_size=1000.0, seed=42
        )

    print(design.netlist.summary())

    # ── Step 2: Simulated Annealing ───────────────────────────────
    print("\n[SA] Step 2: Simulated Annealing Floorplanner...")
    sa_config = SAConfig(
        initial_temperature=500.0,
        cooling_rate=0.995,
        min_temperature=0.1,
        moves_per_temperature=50,
        max_iterations=100000,
        seed=42,
        verbose=True,
        log_interval=2000,
    )

    sa = SimulatedAnnealingFloorplanner(design, config=sa_config)
    sa_result = sa.run()

    # Save SA results for visualization
    sa_cost_history = sa_result.cost_history
    sa_temp_history = sa_result.temperature_history

    # ── Step 3: Visualization (SA result) ─────────────────────────
    print("\n[VIZ] Step 3: Generating SA visualizations...")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)

    viewer = LayoutViewer(design)
    viewer.plot_floorplan(
        save_path=os.path.join(output_dir, 'sa_floorplan.png'),
        show_labels=True, show_nets=True
    )
    viewer.plot_thermal_map(
        save_path=os.path.join(output_dir, 'sa_thermal.png')
    )
    viewer.plot_convergence(
        sa_cost_history,
        save_path=os.path.join(output_dir, 'sa_convergence.png'),
        title='Simulated Annealing Convergence',
        temperature_history=sa_temp_history
    )

    # ── Step 4: PIAB-FP (for comparison) ──────────────────────────
    print("\n[PIAB] Step 4: PIAB-FP Physics-Inspired Floorplanner...")

    # Create a fresh design for PIAB-FP comparison
    if os.path.exists(benchmark_path):
        design_piab = parser.from_json(benchmark_path)
    else:
        design_piab = parser.generate_random(
            num_cells=25, num_nets=40,
            chip_size=1000.0, seed=42
        )

    piab_config = PIABConfig(
        max_iterations=1500,
        dt=0.3,
        damping=0.88,
        seed=42,
        verbose=True,
        log_interval=200,
    )

    piab = PIABFloorplanner(design_piab, config=piab_config)
    piab_result = piab.run()

    # PIAB visualizations
    viewer_piab = LayoutViewer(design_piab)
    viewer_piab.plot_floorplan(
        save_path=os.path.join(output_dir, 'piab_floorplan.png'),
        show_labels=True, show_nets=True
    )
    viewer_piab.plot_convergence(
        piab_result.cost_history,
        save_path=os.path.join(output_dir, 'piab_convergence.png'),
        title='PIAB-FP Convergence'
    )

    # ── Step 5: Global Routing ────────────────────────────────────
    print("\n[ROUTE] Step 5: Global Routing (on SA result)...")
    router = GlobalRouter(
        design, grid_cols=25, grid_rows=25,
        default_capacity=8, congestion_penalty=5.0,
        max_rip_up_iterations=3
    )
    routing_result = router.route(verbose=True)

    # Congestion visualization
    cmap = router.routing_grid.congestion_map()
    viewer.plot_congestion(
        cmap,
        save_path=os.path.join(output_dir, 'congestion.png')
    )

    # ── Step 6: Static Timing Analysis ────────────────────────────
    print("\n[STA] Step 6: Static Timing Analysis...")
    sta = STAEngine(design, clock_period=10.0, wire_delay_factor=0.01)
    timing_result = sta.analyze(verbose=True)

    # ── Step 7: Dashboard ─────────────────────────────────────────
    print("\n[DASH] Step 7: Generating Design Dashboard...")
    viewer.plot_dashboard(
        save_path=os.path.join(output_dir, 'dashboard.png'),
        cost_history=sa_cost_history,
        congestion_map=cmap
    )

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FLOW COMPLETE - RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  SA Best Cost:     {sa_result.best_cost:.4f}")
    print(f"  PIAB-FP Cost:     {piab_result.final_cost:.4f}")
    print(f"  Routing Success:  {routing_result.routed_nets}/{routing_result.total_nets}")
    print(f"  Routing WL:       {routing_result.total_wirelength:.1f}")
    print(f"  Timing WNS:       {timing_result.wns:.3f} ns")
    print(f"  Timing TNS:       {timing_result.tns:.3f} ns")
    print(f"\n  Output saved to: {os.path.abspath(output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
