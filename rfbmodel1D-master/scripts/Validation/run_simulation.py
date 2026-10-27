"""
run_simulation.py
=================
Executes the full 1-D HVRFB model from the command line, mirroring the
workflow in notebooks/complete_cell/debug_time_integration.ipynb.

Usage
-----
    python scripts/run_simulation.py [--tf TF] [--soc SOC] [--rtol RTOL]
                                     [--out OUT] [--no-plot]

Outputs
-------
    <OUT>.npz  – compressed NumPy archive with tsol, ysol, and meta-data.

Example
-------
python scripts/run_simulation.py --tf 0.1 --soc 0.95 --phi_el 1 --no-plot --out results/results_t0.1
python scripts/run_simulation.py --tf 0.1 --soc 0.95 --phi_el 1 --out results/results_t0.1      
"""

import argparse
import time
from pathlib import Path
import numpy as np
from solve_dae.integrate import solve_dae
from rfbmodel1D.parameters.params import SimulationParams, Vars
from rfbmodel1D.cells.cell import Cell 

def parse_args():
    p = argparse.ArgumentParser(description="Run the 1-D HVRFB full-cell model.")
    p.add_argument("--tf",    type=float, default=0.1,    help="Final integration time [s] (default: 0.1)")
    p.add_argument("--soc",   type=float, default=0.95,   help="State of charge [0-1] (default: 0.95)")
    p.add_argument("--rtol",  type=float, default=1e-4,   help="Relative tolerance (default: 1e-4)")
    p.add_argument("--out",   type=str,   default="results/results", help="Output file stem (default: results/results)")
    p.add_argument("--no-plot", action="store_true",      help="Skip matplotlib plots")
    p.add_argument("--phi_el", type=float, default=1.1,   help="Fixed electronic potential in CC cathode (default: 1.1 V)")
    return p.parse_args()


# ------------------------------------------------------------------------------
# Build absolute tolerances (mirrors notebook cell)
# ------------------------------------------------------------------------------

def build_atol(cell):
    Nc = cell.mesh.cathode_mesh.n_x
    Nm = cell.mesh.membrane_mesh.n_x
    conc          = 1e-1
    potential     = 1e-3
    lambda_       = 1e-2
    overpotential = 1e-4
    saturation    = 1e-3
    volume        = 1e-6
    gas_conc      = 1e-3
    atol_tank      = np.array([conc, conc, conc, conc, volume],                  dtype=float)
    atol_cath_node = np.array([conc, conc, conc, conc, potential, potential],     dtype=float)
    atol_mem_node  = np.array([lambda_, potential],                               dtype=float)
    atol_anode     = np.array([gas_conc, gas_conc, saturation,
                               overpotential, gas_conc, gas_conc, saturation],    dtype=float)
    return np.concatenate([
        atol_tank,
        np.tile(atol_cath_node, Nc),
        np.tile(atol_mem_node,  Nm),
        atol_anode,
    ])


# ------------------------------------------------------------------------------
# Terminal event: stop once cathode tank SOC drops below SOC_MIN
# ------------------------------------------------------------------------------

SOC_MIN    = 0.02  # stop at 2 % SOC — below this concentration starvation dominates
LAMBDA_MIN = 2.2   # stop when membrane λ drops to this value (κ near-singular below here)

def soc_cutoff_event(t, y, yp):
    """Terminal event: fires (zero crossing, decreasing) when tank SOC < SOC_MIN."""
    c_V4 = y[0]   # tank V4  [mol/m³]
    c_V5 = y[1]   # tank V5  [mol/m³]
    soc  = c_V5 / (c_V4 + c_V5 + 1e-10)
    return soc - SOC_MIN

soc_cutoff_event.terminal  = True
soc_cutoff_event.direction = -1   # only trigger while SOC is decreasing


def make_lambda_cutoff_event(cell):
    """
    Factory that returns a terminal event firing when min(λ_membrane) < LAMBDA_MIN.
    Membrane state layout: each node stores [λ, Φ_io], so λ values are at
    even offsets within the membrane block.
    """
    n_tk  = cell.layout.n_vars_tank
    Nc    = cell.mesh.cathode_mesh.n_x
    n_ca  = cell.layout.n_vars_cathode
    Nm    = cell.mesh.membrane_mesh.n_x
    off_m = n_tk + Nc * n_ca          # index of first membrane state variable
    # λ is the first variable of each membrane node -> stride 2
    lam_idx = np.arange(off_m, off_m + Nm * 2, 2)

    def lambda_cutoff_event(t, y, yp):
        """Terminal event: fires when minimum membrane λ drops below LAMBDA_MIN."""
        return np.min(y[lam_idx]) - LAMBDA_MIN

    lambda_cutoff_event.terminal  = True
    lambda_cutoff_event.direction = -1
    return lambda_cutoff_event


# ------------------------------------------------------------------------------
# Setup helpers
# ------------------------------------------------------------------------------

def build_cell(soc, phi_el):
    sim_params = SimulationParams
    cell = Cell(params=sim_params)
    cell.modify_BC(Vars.phi_el, "cathode", "left", "dirichlet", phi_el)

    V_tank   = 0.0000005                  # [m³]
    c_T_V4   = (1 - soc) * 1000           # [mol/m³]
    c_T_V5   = soc * 1000
    c_T_H    = 4963
    c_T_HSO4 = c_T_H * 3 / 5

    c_0 = np.array([c_T_V4, c_T_V5, c_T_H, c_T_HSO4, V_tank])
    y0, yp0 = cell.initialize_dae_state(c_0, xtol=1e-6)
    return cell, y0, yp0

# ------------------------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------------------------

def print_summary(sol, tsol, ysol, cell):
    Nc   = cell.mesh.cathode_mesh.n_x
    Nm   = cell.mesh.membrane_mesh.n_x
    n_ca = cell.layout.n_vars_cathode
    n_me = cell.layout.n_vars_membrane
    n_tk = cell.layout.n_vars_tank
    n_an = cell.layout.n_vars_anode

    off_ca  = n_tk
    off_mem = off_ca  + Nc * n_ca
    off_an  = off_mem + Nm * n_me

    y_final     = ysol[:, -1]
    y_ca_final  = y_final[off_ca:off_mem].reshape(Nc, n_ca)
    y_mem_final = y_final[off_mem:off_an].reshape(Nm, n_me)
    y_an_final  = y_final[off_an:off_an + n_an]

    c_V4_f = ysol[0, -1];  c_V5_f = ysol[1, -1]
    soc_f  = c_V5_f / (c_V4_f + c_V5_f + 1e-10)
    print(f"\n✓ Integration completed in {sol._elapsed:.2f} s")
    print(f"  Success : {sol.success}")
    print(f"  Message : {sol.message}")
    if sol.t_events is not None:
        if len(sol.t_events[0]) > 0:
            print(f"  Stopped : SOC cutoff event (SOC={soc_f:.3f} < {SOC_MIN})")
        if len(sol.t_events[1]) > 0:
            print(f"  Stopped : membrane drying event (λ_min < {LAMBDA_MIN})")
    print(f"  t_final : {tsol[-1]:.6g} s")
    print(f"  Steps   : {len(tsol)}")
    print(f"  nfev    : {sol.nfev}  |  njev: {sol.njev}  |  nlu: {sol.nlu}")

    labels_an = ["c_vw_acl", "c_H2_acl", "s_acl", "eta_acl",
                 "c_vw_agdl", "c_H2_agdl", "s_agdl"]
    print("\nAnode final state:")
    for name, val in zip(labels_an, y_an_final):
        print(f"  {name:12s} = {val:.6e}")

    print(f"\nMembrane λ (first / last node): "
          f"{y_mem_final[0, 0]:.6e}  /  {y_mem_final[-1, 0]:.6e}")

    c_H_min  = np.min(y_ca_final[:, 2])
    lam_min  = np.min(y_mem_final[:, 0])
    print(f"\nPositivity: min(c_H)={c_H_min:.3e}  min(λ)={lam_min:.3e}  "
          f"OK={c_H_min > 0 and lam_min > 0}")

    return y_mem_final, Nm


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    args = parse_args()
    out_path = Path(args.out)
    if out_path.suffix != ".npz":
        out_path = out_path.with_suffix(".npz")
    output_dir = out_path.parent if str(out_path.parent) else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building cell and initialising DAE state …")
    cell, y0, yp0 = build_cell(args.soc, args.phi_el)
    atol = build_atol(cell)

    lambda_cutoff_event = make_lambda_cutoff_event(cell)

    print(f"Running BDF integrator  t=[0, {args.tf}] s …")
    t0 = time.time()
    sol = solve_dae(
        cell.dae_residual_full_cell,
        (0.0, args.tf),
        y0, yp0,
        method="BDF",
        atol=atol,
        rtol=args.rtol,
        events=[soc_cutoff_event, lambda_cutoff_event],
    )
    sol._elapsed = time.time() - t0

    ysol = sol.y 
    tsol = sol.t

    y_mem_final, Nm = print_summary(sol, tsol, ysol, cell)

    # Save results
    np.savez_compressed(out_path, tsol=tsol, ysol=ysol,
                        soc=args.soc, tf=args.tf, rtol=args.rtol,
                        phi_el=args.phi_el)
    print(f"\nResults saved to: {out_path}")

    if not args.no_plot:
        import subprocess
        import sys
        run_all_plots = Path(__file__).parent / "run_all_plots.py"
        subprocess.run([
            sys.executable,
            str(run_all_plots),
            "--result-name", out_path.stem,
            "--results-dir", str(output_dir),
            "--output-dir", str(output_dir / "plot_results")
        ], check=True)
    # Save final values
    np.save('results/kappa_io_value.npy', cell.membrane.kappa_io)

if __name__ == "__main__":
    main()