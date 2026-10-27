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

"""Check conservation of current density (charge) and tank capacity (mass) over time.
python scripts/Validation/check_conservation.py
python scripts/Validation/check_conservation.py --load results/validation/conservation.npz
"""
# ------------------------------------------------------------------------------
# Simulated scenario
# ------------------------------------------------------------------------------
soc_list   = [0.25, 0.75]
j_list     = [-250, -500, -1000]
T          = 32    # °C
Q_v        = 45    # ml/min
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
    V_tank   = 0.5 / 1000
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
def run_simulations(soc_list, j_list, N_c, N_m, tf, save_path):
    x_ca_all     = []
    x_mem_all    = []
    i_el_ca_all  = []
    i_io_ca_all  = []
    i_io_mem_all = []
    t_all        = []
    socs_conc    = []
    socs_faraday = []
    run_soc      = []
    run_j        = []
    etas         = []

    for SOC in soc_list:
        for j in j_list:
            print(f"Simulating SOC={SOC}, j={j} A/m² ...")
            cell, y0, yp0 = build_cell(SOC, j, T, Q_v, N_c, N_m)
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

            # -- Current density at final time ---------------------------------
            y_final = sol.y[:, -1]
            result  = calculate_current_density(cell, y_final)
            x_ca_all.append(result["x_ca"])
            x_mem_all.append(result["x_mem"])
            i_el_ca_all.append(result["i_el_ca"])
            i_io_ca_all.append(result["i_io_ca"])
            i_io_mem_all.append(result["i_io_mem"])
            run_soc.append(SOC)
            run_j.append(j)
            etas.append(cell.cathode.eta) # type: ignore
            # -- SOC time series -----------------------------------------------
            eps    = cell.cathode.porosity
            A_ca   = cell.cathode.height * cell.cathode.width
            Nc     = cell.mesh.cathode_mesh.n_x
            n_tank = cell.layout.n_vars_tank
            n_ca   = cell.layout.n_vars_cathode
            Nt     = len(sol.t)
            off_ca = n_tank
            dx_ca  = cell.mesh.cathode_mesh.dx * cell.cathode.thickness
            V_cells = A_ca * eps * dx_ca

            n_V5_tank  = sol.y[1, :] * sol.y[4, :]
            c_V5_ca    = sol.y[off_ca:off_ca + Nc*n_ca, :].reshape(Nc, n_ca, Nt)[:, 1, :]
            n_V5_ca    = np.einsum("i,it->t", V_cells, c_V5_ca)
            n_V5_total = n_V5_tank + n_V5_ca

            n_V4_tank  = sol.y[0, :] * sol.y[4, :]
            c_V4_ca    = sol.y[off_ca:off_ca + Nc*n_ca, :].reshape(Nc, n_ca, Nt)[:, 0, :]
            n_V4_ca    = np.einsum("i,it->t", V_cells, c_V4_ca)
            n_V4_total = n_V4_tank + n_V4_ca
            n_V_total  = n_V5_total + n_V4_total

            I_appl       = j * A_ca
            n_V5_Faraday = n_V5_total[0] + I_appl * sol.t / cell.F
            SOC_model    = n_V5_total   / n_V_total * 100
            SOC_Faraday  = n_V5_Faraday / n_V_total * 100

            t_all.append(sol.t)
            socs_conc.append(SOC_model)
            socs_faraday.append(SOC_Faraday)

    L_ca  = SimulationParams.CathodeParams.thickness
    L_mem = SimulationParams.MembraneParams.thickness

    max_ca  = max(a.size for a in x_ca_all)
    max_mem = max(a.size for a in x_mem_all)
    max_t   = max(a.size for a in t_all)

    def pad(arrays, length):
        return np.array([np.pad(a, (0, length - a.size), constant_values=np.nan)
                         for a in arrays])

    data = dict(
        L_ca         = np.array(L_ca),
        L_mem        = np.array(L_mem),
        x_ca         = pad(x_ca_all,     max_ca),
        x_mem        = pad(x_mem_all,    max_mem),
        i_el_ca      = pad(i_el_ca_all,  max_ca),
        i_io_ca      = pad(i_io_ca_all,  max_ca),
        i_io_mem     = pad(i_io_mem_all, max_mem),
        run_soc      = np.array(run_soc),
        run_j        = np.array(run_j),
        t_all        = pad(t_all,        max_t),
        socs_conc    = pad(socs_conc,    max_t),
        socs_faraday = pad(socs_faraday, max_t),
        etas         = np.array(etas),
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        L_ca=data["L_ca"], L_mem=data["L_mem"],
        x_ca=data["x_ca"], x_mem=data["x_mem"],
        i_el_ca=data["i_el_ca"], i_io_ca=data["i_io_ca"], i_io_mem=data["i_io_mem"],
        run_soc=data["run_soc"], run_j=data["run_j"],
        t_all=data["t_all"], socs_conc=data["socs_conc"], socs_faraday=data["socs_faraday"], etas=data["etas"],
    )
    print(f"Results saved to {save_path}")
    return data

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------
def plot_results(data, out_dir="results/validation"):
    L_ca    = float(data["L_ca"])
    L_mem   = float(data["L_mem"])
    L_tot   = L_ca + L_mem
    x_iface = L_ca / L_tot
    run_soc = data["run_soc"]
    run_j   = data["run_j"]

    j_vals   = sorted(set(run_j.tolist()))
    soc_vals = sorted(set(run_soc.tolist()))          # [0.1, 0.9]
    j_colors = dict(zip(j_vals, matplotlib.colormaps["plasma"](np.linspace(0.1, 0.85, len(j_vals)))))
    soc_ls   = {soc_vals[0]: "--", soc_vals[1]: "-"}  # low SOC -> dashed, high SOC -> solid

    j_handles = [
        mlines.Line2D([], [], color=j_colors[j], ls="-", lw=1.8,
                      label=fr"$j={j}$ A m$^{{-2}}$")
        for j in j_vals
    ]
    soc_handles = [
        mlines.Line2D([], [], color="0.3", ls=soc_ls[s], lw=1.8,
                      label=fr"SOC$_0$ = {s:.1f}")
        for s in soc_vals
    ]

    def add_legends(ax, loc1="upper left", loc2="upper right"):
        leg1 = ax.legend(handles=j_handles,   fontsize=9, frameon=True, loc=loc1)
        leg2 = ax.legend(handles=soc_handles, fontsize=9, frameon=True, loc=loc2)
        ax.add_artist(leg1)
        for leg in (leg1, leg2):
            leg.get_frame().set_linewidth(0.8)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # -- Plot 1: current density at final time ---------------------------------
    _, ax = plt.subplots(figsize=(6, 6))
    for idx in range(len(run_j)):
        c       = j_colors[run_j[idx]]
        ls      = soc_ls[run_soc[idx]]
        x_n_ca  = data["x_ca"][idx]  / L_tot
        x_n_mem = data["x_mem"][idx] / L_tot
        ax.plot(x_n_ca,  data["i_el_ca"][idx],  ls, color=c, lw=1.8)
        ax.plot(x_n_ca,  data["i_io_ca"][idx],  ls, color=c, lw=1.0, alpha=0.5)
        ax.plot(x_n_mem, data["i_io_mem"][idx], ls, color=c, lw=1.0, alpha=0.3)
    ax.axvline(x=x_iface, color="0.5", linestyle="--", lw=1.0)
    ax.set_xlabel("Normalised position  $x / L$", fontsize=12)
    ax.set_ylabel(r"Current density / A m$^{-2}$", fontsize=12)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.25, linestyle=":")
    add_legends(ax, loc1="center right", loc2="center left")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/conservation_current.png", dpi=600, bbox_inches="tight")
    plt.close()

    # -- Plot 2: SOC over time -------------------------------------------------
    type_handles = [
        mlines.Line2D([], [], color="0.3", lw=1.8, ls="--",  label="Concentration (tank+cathode)"),
        mlines.Line2D([], [], color="0.3", lw=0.8, ls="-",  label="Faraday"),
    ]
    _, ax2 = plt.subplots(figsize=(6, 6))
    for idx in range(len(run_j)):
        c  = j_colors[run_j[idx]]
        ls = soc_ls[run_soc[idx]]
        t  = data["t_all"][idx]
        ax2.plot(t, data["socs_conc"][idx], color=c, lw=1.8, linestyle = "--")
        ax2.plot(t, data["socs_faraday"][idx], color=c, lw=0.8)
    ax2.set_xlabel("Time / s", fontsize=12)
    ax2.set_ylabel("SOC / %", fontsize=12)
    ax2.grid(True, alpha=0.25, linestyle=":")
    leg_j2   = ax2.legend(handles=j_handles,    fontsize=9, frameon=True, loc=2)
    leg_soc2 = ax2.legend(handles=soc_handles,  fontsize=9, frameon=True, loc=1)
    leg_typ  = ax2.legend(handles=type_handles, fontsize=9, frameon=True, loc= "upper right")
    ax2.add_artist(leg_j2)
    ax2.set_ylim(0,100)
    for leg in (leg_j2, leg_typ, leg_soc2):
        leg.get_frame().set_linewidth(0.8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/conservation_soc.png", dpi=600, bbox_inches="tight")
    plt.close()

    # -- Plot 3: overpotential over cathode thickness at final time -----------
    _, ax3 = plt.subplots(figsize=(6, 6))
    for idx in range(len(run_j)):
        c           = j_colors[run_j[idx]]
        ls          = soc_ls[run_soc[idx]]
        x_ca        = data["x_ca"][idx]
        x_ca_center = 0.5 * (x_ca[:-1] + x_ca[1:]) / L_ca
        ax3.plot(x_ca_center, data["etas"][idx], ls, color=c, lw=1.8)
    ax3.set_xlabel(r"Normalised position  $x / L_\mathrm{ca}$", fontsize=12)
    ax3.set_ylabel(r"Overpotential $\eta$ / V", fontsize=12)
    ax3.set_xlim(0, 1)
    ax3.grid(True, alpha=0.25, linestyle=":")
    add_legends(ax3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/conservation_eta.png", dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Figures saved to {out_dir}/")
    plt.show()

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", metavar="NPZ", default=None,
                        help="Skip simulation and load results from an existing .npz file")
    parser.add_argument("--save", metavar="NPZ", default="results/validation/conservation.npz",
                        help="Path to save simulation results")
    parser.add_argument("--out-dir", default="results/validation",
                        help="Directory for output figures")
    args = parser.parse_args()

    if args.load:
        print(f"Loading results from {args.load}")
        data = dict(np.load(args.load))
    else:
        N_c = 150
        N_m = 150
        tf  = 2 * 3600
        data = run_simulations(soc_list, j_list, N_c, N_m, tf, save_path=args.save)

    plot_results(data, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
