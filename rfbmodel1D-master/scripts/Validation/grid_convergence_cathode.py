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
python scripts/Validation/grid_convergence_cathode.py
python scripts/Validation/grid_convergence_cathode.py --load results/validation/grid_convergence_cathode.npz
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
    x_ca_all     = []
    x_mem_all    = []
    i_el_ca_all  = []
    i_io_ca_all  = []
    i_io_mem_all = []

    for nodes in N_c:
        print(f"Simulating N_c = {nodes} ...")
        cell, y0, yp0 = build_cell(soc, j_appl, T, Q_v, nodes, N_m)
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
        y_final = sol.y[:, -1]
        result  = calculate_current_density(cell, y_final)
        x_ca_all.append(result["x_ca"])
        x_mem_all.append(result["x_mem"])
        i_el_ca_all.append(result["i_el_ca"])
        i_io_ca_all.append(result["i_io_ca"])
        i_io_mem_all.append(result["i_io_mem"])

    L_ca  = SimulationParams.CathodeParams.thickness
    L_mem = SimulationParams.MembraneParams.thickness

    # Pad arrays to a common length so they can be stored in one npz
    max_ca  = max(a.size for a in x_ca_all)
    max_mem = max(a.size for a in x_mem_all)

    def pad(arrays, length):
        return np.array([np.pad(a, (0, length - a.size), constant_values=np.nan)
                         for a in arrays])

    data = dict(
        N_c        = np.array(N_c),
        j_appl     = np.array(j_appl),
        L_ca       = np.array(L_ca),
        L_mem      = np.array(L_mem),
        x_ca       = pad(x_ca_all,     max_ca),
        x_mem      = pad(x_mem_all,    max_mem),
        i_el_ca    = pad(i_el_ca_all,  max_ca),
        i_io_ca    = pad(i_io_ca_all,  max_ca),
        i_io_mem   = pad(i_io_mem_all, max_mem),
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        N_c=data["N_c"], j_appl=data["j_appl"], L_ca=data["L_ca"], L_mem=data["L_mem"],
        x_ca=data["x_ca"], x_mem=data["x_mem"],
        i_el_ca=data["i_el_ca"], i_io_ca=data["i_io_ca"], i_io_mem=data["i_io_mem"],
    )
    print(f"Results saved to {save_path}")
    return data

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------
def plot_results(data, out_dir="results/validation"):
    N_c      = data["N_c"].tolist()
    j_appl   = float(data["j_appl"])
    L_ca     = float(data["L_ca"])
    L_mem    = float(data["L_mem"])
    L_tot    = L_ca + L_mem
    x_iface  = L_ca / L_tot
    zoom_half = 0.05
    x_lo = x_iface - zoom_half
    x_hi = min(x_iface + zoom_half, 1.0)

    colors = matplotlib.colormaps["plasma"](np.linspace(0.1, 0.85, len(N_c)))

    style_handles = [
        mlines.Line2D([], [], color="0.3", ls="-",  lw=1.8, label=r"$i_\mathrm{el}$ (cathode)"),
        mlines.Line2D([], [], color="0.3", ls="--", lw=1.8, label=r"$i_\mathrm{io}$ (cathode)"),
        mlines.Line2D([], [], color="0.3", ls=":",  lw=1.8, label=r"$i_\mathrm{io}$ (membrane)"),
    ]

    def draw_curves(ax, idx, NC, c, label=True):
        x_n_ca  = data["x_ca"][idx]  / L_tot
        x_n_mem = data["x_mem"][idx] / L_tot
        lbl = fr"$N_c={NC}$" if label else "_"
        ax.plot(x_n_ca,  data["i_el_ca"][idx],  "-",  color=c, lw=1.8, label=lbl)
        ax.plot(x_n_ca,  data["i_io_ca"][idx],  "--", color=c, lw=1.8, label="_")
        ax.plot(x_n_mem, data["i_io_mem"][idx], ":",  color=c, lw=1.8, label="_")

    def add_legends(ax, loc1="center left", loc2="center right"):
        mesh_h, mesh_l = ax.get_legend_handles_labels()
        leg1 = ax.legend(handles=mesh_h, labels=mesh_l,
                         fontsize=9, frameon=True, loc=loc1,
                         title=r"$N_c$", title_fontsize=9)
        leg2 = ax.legend(handles=style_handles,
                         fontsize=9, frameon=True, loc=loc2)
        ax.add_artist(leg1)
        for leg in (leg1, leg2):
            leg.get_frame().set_linewidth(0.8)

    def style_ax(ax, xlim=(0, 1)):
        ax.set_xlabel("Normalised position  $x / L$", fontsize=12)
        ax.set_ylabel(r"Current density / A m$^{-2}$", fontsize=12)
        ax.set_xlim(*xlim)
        ax.axvline(x=x_iface, color="0.5", linestyle="--", lw=1.0)
        ax.grid(True, alpha=0.25, linestyle=":")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # -- Plot 1: full domain ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    for idx, NC in enumerate(N_c):
        draw_curves(ax, idx, NC, colors[idx])
    style_ax(ax)
    add_legends(ax)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/current_mesh_global.png", dpi=600, bbox_inches="tight")
    plt.close()

    # -- Plot 2: zoomed interface ----------------------------------------------
    _, ax2 = plt.subplots(figsize=(6, 6))
    for idx, NC in enumerate(N_c):
        draw_curves(ax2, idx, NC, colors[idx])
    style_ax(ax2, xlim=(x_lo, x_hi))
    ax2.set_ylim(j_appl * 1.02, j_appl * 0.98)
    add_legends(ax2)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/current_mesh_interface.png", dpi=600, bbox_inches="tight")
    plt.close()

    # -- Plot 3: full domain with inset zoom -----------------------------------
    fig3, ax4 = plt.subplots(figsize=(6, 6))
    for idx, NC in enumerate(N_c):
        draw_curves(ax4, idx, NC, colors[idx])
    style_ax(ax4)
    add_legends(ax4, loc1="upper left", loc2="upper center")
    ax4_in = ax4.inset_axes((0.66, 0.38, 0.25, 0.25))
    for idx, NC in enumerate(N_c):
        x_n_ca  = data["x_ca"][idx]  / L_tot
        x_n_mem = data["x_mem"][idx] / L_tot
        ax4_in.plot(x_n_ca,  data["i_el_ca"][idx],  "-",  color=colors[idx], lw=1.5)
        ax4_in.plot(x_n_ca,  data["i_io_ca"][idx],  "--", color=colors[idx], lw=1.5)
        ax4_in.plot(x_n_mem, data["i_io_mem"][idx], ":",  color=colors[idx], lw=1.5)
    ax4_in.axvline(x=x_iface, color="0.5", linestyle="--", lw=1.0)
    ax4_in.set_xlim(x_lo, x_hi)
    ax4_in.set_ylim(j_appl * 1.03, j_appl * 0.97)
    ax4_in.tick_params(labelsize=8)
    ax4_in.grid(True, alpha=0.25, linestyle=":")
    ax4.indicate_inset_zoom(ax4_in, edgecolor="0.4", linestyle="dotted")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/current_mesh_inset.png", dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Figures saved to {out_dir}/")
    

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", metavar="NPZ", default=None,
                        help="Skip simulation and load results from an existing .npz file")
    parser.add_argument("--save", metavar="NPZ", default="results/validation/grid_convergence_cathode.npz",
                        help="Path to save simulation results (default: results/validation/grid_convergence_cathode.npz)")
    parser.add_argument("--out-dir", default="results/validation",
                        help="Directory for output figures")
    args = parser.parse_args()

    if args.load:
        print(f"Loading results from {args.load}")
        data = dict(np.load(args.load))
    else:
        N_c    = [80, 160, 320, 640]
        j_appl = -1500
        tf     = 500
        N_m    = 150
        data = run_simulations(N_c, j_appl, tf, N_m, save_path=args.save)

    plot_results(data, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
