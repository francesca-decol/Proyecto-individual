import argparse
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
from pathlib import Path
import numpy as np
from solve_dae.integrate import solve_dae
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from rfbmodel1D.parameters.params import SimulationParams, Vars
from rfbmodel1D.cells.cell import Cell
from scripts.functions.calculate_current_density import calculate_current_density

"""
Simulate E-cell experiments using the 1-D HVRFB model, keeping constant current density and tank concentrations.
python scripts/Validation/grid_convergence_membrane.py
python scripts/Validation/grid_convergence_membrane.py --load results/validation/grid_convergence_membrane.npz
"""
# ------------------------------------------------------------------------------
# Simulated scenario
# ------------------------------------------------------------------------------
soc   = 0.5
T     = 32   # °C
Q_v   = 45   # ml/min
# ------------------------------------------------------------------------------
# Tolerances / solver settings
# ------------------------------------------------------------------------------
RTOL    = 1e-6
SOC_MIN = 0.02

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
    atol_tank      = np.array([conc, conc, conc, conc, volume],              dtype=float)
    atol_cath_node = np.array([conc, conc, conc, conc, potential, potential], dtype=float)
    atol_mem_node  = np.array([lambda_, potential],                           dtype=float)
    atol_anode     = np.array([gas_conc, gas_conc, saturation,
                               overpotential, gas_conc, gas_conc, saturation], dtype=float)
    atol_coupling  = np.array([conc], dtype=float)
    return np.concatenate([
        atol_tank,
        np.tile(atol_cath_node, Nc),
        np.tile(atol_mem_node,  Nm),
        atol_anode,
        atol_coupling,
    ])

def soc_cutoff_event(t, y, yp):
    c_V4 = y[0]
    c_V5 = y[1]
    soc  = c_V5 / (c_V4 + c_V5 + 1e-10)
    return soc - SOC_MIN
soc_cutoff_event.terminal  = True  # type: ignore
soc_cutoff_event.direction = -1    # type: ignore

def build_cell(soc, j_appl, T, Q_v, N_c, N_m):
    sim_params = SimulationParams
    SimulationParams.CathodeParams.Q_total = Q_v * 1e-6 / 60 * 1000
    SimulationParams.CellParams.T          = T + 273.15
    SimulationParams.MeshParams.NM = N_m
    SimulationParams.MeshParams.NC = N_c
    cell         = Cell(params=sim_params)
    cell.bc_mode = "galvanostatic"
    cell.modify_BC(Vars.phi_el, "cathode", "left", "galvanostatic", j_appl)
    V_tank   = 50000
    c_T_V4   = (1 - soc) * 1000
    c_T_V5   = soc * 1000
    c_T_H    = 4963
    c_T_HSO4 = c_T_H * 3 / 5
    c_0 = np.array([c_T_V4, c_T_V5, c_T_H, c_T_HSO4, V_tank])
    y0, yp0 = cell.initialize_dae_state(c_0, xtol=1e-6)
    return cell, y0, yp0

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------
def run_simulations(N_c, j_appl, tf, N_m, save_path):
    x_mem_all       = []
    phi_io_mem_all  = []
    lam_mem_all     = []
    for nodes in N_m:
        print(f"Simulating N_m = {nodes} ...")
        cell, y0, yp0 = build_cell(soc, j_appl, T, Q_v, N_c, nodes)
        atol = build_atol(cell)
        t0  = time.time()
        sol = solve_dae(
            cell.dae_residual_full_cell,
            (0.0, tf),
            y0, yp0,
            method="BDF",
            atol=atol,
            rtol=RTOL,
            events=soc_cutoff_event,
        )
        print(f"  done in {time.time() - t0:.1f} s")
        y_final    = sol.y[:, -1]
        segs       = cell.split_state(y_final)
        y_mem      = segs["membrane"] 
        lam_m      = y_mem[:, 0]
        phi_io_mem = y_mem[:, 1]
        # Nodes center
        dx_m  = cell.mesh.membrane_mesh.dx * cell.membrane.thickness # Nm
        edges_m  = np.concatenate([[0.0], np.cumsum(dx_m)])   # Nm+1
        x_center_m  = 0.5 * (edges_m[:-1]  + edges_m[1:])    # Nm cell centres
        x_mem_all.append(x_center_m)
        phi_io_mem_all.append(phi_io_mem)
        lam_mem_all.append(lam_m)
    L_mem = SimulationParams.MembraneParams.thickness
    # Pad arrays to a common length so they can be stored in one npz
    max_mem = max(a.size for a in x_mem_all)
    def pad(arrays, length):
        return np.array([np.pad(a, (0, length - a.size), constant_values=np.nan)
                         for a in arrays])
    data = dict(
        N_m        = np.array(N_m),
        j_appl     = np.array(j_appl),
        L_mem      = np.array(L_mem),
        x_mem      = pad(x_mem_all,    max_mem),
        phi_io_mem = pad(phi_io_mem_all, max_mem),
        lam_mem    = pad(lam_mem_all,    max_mem),
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        N_m=data["N_m"], j_appl=data["j_appl"], L_mem=data["L_mem"],
        x_mem=data["x_mem"], phi_io_mem=data["phi_io_mem"], lam_mem=data["lam_mem"],
    )
    print(f"Results saved to {save_path}")
    return data

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------
def plot_results(data, out_dir="results/validation"):
    N_m        = data["N_m"].tolist()
    phi_io_mem = data["phi_io_mem"]
    lam_mem    = data["lam_mem"]
    x_mem      = data["x_mem"]
    L_mem    = float(data["L_mem"])
    colors = matplotlib.colormaps["plasma"](np.linspace(0.1, 0.85, len(N_m)))

    def draw_curves(ax, idx, NM, c, variable ,label=True):
        x_n_mem  = data["x_mem"][idx]  / L_mem
        lbl = fr"$N_m={NM}$" if label else "_"
        ax.plot(x_n_mem,  data[variable][idx],  "-",  color=c, lw=1.8, label=lbl)

    def add_legends(ax, loc1="center left", loc2="center right"):
        mesh_h, mesh_l = ax.get_legend_handles_labels()
        leg1 = ax.legend(handles=mesh_h, labels=mesh_l,
                         fontsize=9, frameon=True, loc=loc1,
                         title=r"$N_m$", title_fontsize=9)
        ax.add_artist(leg1)
        leg1.get_frame().set_linewidth(0.8)

    def style_ax(ax, variable, xlim=(0, 1)):
        ax.set_xlabel("Normalised position  $x / L$", fontsize=12)
        if variable == "phi_io_mem":
            ax.set_ylabel(r"Membrane ionic potential (V)", fontsize=12)
        else:
            ax.set_ylabel(r"Membrane water content", fontsize=12)
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.25, linestyle=":")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # -- Plot 1: Water content in membrane ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    for idx, NM in enumerate(N_m):
        draw_curves(ax, idx, NM, colors[idx], variable = "lam_mem")
    style_ax(ax, variable="lam_mem")
    add_legends(ax)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/mem_water_content.png", dpi=600, bbox_inches="tight")
    plt.close()
    # -- Plot 2: Ionic potential in membrane ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    x_hi = 0.05
    x_lo = 0.0
    phi_lo = -0.108
    phi_hi = -0.104
    for idx, NM in enumerate(N_m):
        draw_curves(ax, idx, NM, colors[idx], variable = "phi_io_mem")
    style_ax(ax, variable="phi_io_mem")
    add_legends(ax)
    ax4_in = ax.inset_axes((0.66, 0.1, 0.25, 0.25))
    for idx, NM in enumerate(N_m):
        draw_curves(ax4_in, idx, NM, colors[idx], variable = "phi_io_mem", label=False)
    ax4_in.set_xlim(x_lo, x_hi)     
    ax4_in.set_ylim(phi_lo, phi_hi)
    ax4_in.tick_params(labelsize=8)
    ax4_in.grid(True, alpha=0.25, linestyle=":")
    ax.indicate_inset_zoom(ax4_in, edgecolor="0.4", linestyle="dotted")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/mem_ionic_potential.png", dpi=600, bbox_inches="tight")
    plt.close()
# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", metavar="NPZ", default=None,
                        help="Skip simulation and load results from an existing .npz file")
    parser.add_argument("--save", metavar="NPZ", default="results/validation/grid_convergence_membrane.npz",
                        help="Path to save simulation results (default: results/validation/grid_convergence_membrane.npz)")
    parser.add_argument("--out-dir", default="results/validation",
                        help="Directory for output figures")
    args = parser.parse_args()

    if args.load:
        print(f"Loading results from {args.load}")
        data = dict(np.load(args.load))
    else:
        N_c    = 150
        j_appl = -1500
        tf     = 500
        N_m    = [80, 160, 320, 640]
        data = run_simulations(N_c, j_appl, tf, N_m, save_path=args.save)

    plot_results(data, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
