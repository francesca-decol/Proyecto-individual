"""
Fits interfacial mass-transfer coefficients (k_vw_left, k_vw_right) and
equilibrium water contents (lam_eq_left, lam_eq_right) to the transient
DRY_VS sorption data (dry N₂ left, saturated vapour right, Hwang 2013).
Diffusivity D(λ) is fixed from the prior desorption fit.

Usage:  python scripts/Experimental/water_fit_sorption.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import dual_annealing, minimize
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from rfbmodel1D.parameters.params import D_params, k_params as k_params_ref
from scripts.functions.membrane_model import N_FVM, membrane_ode, cost_k, update_params_file

# -- Geometry -------------------------------------------------------------------
L       = 120e-6
DX      = L / N_FVM
X_NODES = (np.arange(N_FVM) + 0.5) * DX
# -- Input  -------------------------------------------------------------------
input = "liquid" # "sorption" or "desorption" or "liquid"
s_f   = 1    # Scalling factor
if input == "sorption":
    DATA_PATH = Path(__file__).parent.parent.parent.parent / "data/experimental/DRY_VS_sorption_Hwang.csv"
    left_condition  = dict(state = "vapor", mode = "desorption")
    right_condition = dict(state = "vapor", mode = "sorption")
    a_left          = 0
    a_right         = 1
    lam_0           = 2
    _title = "Fitting sorption data (dry N₂ / vapour)"
elif input == "desorption":
    DATA_PATH = Path(__file__).parent.parent.parent.parent / "data/experimental/DRY_VS_desorption_Hwang.csv"
    left_condition  = dict(state = "vapor", mode = "desorption")
    right_condition = dict(state = "vapor", mode = "sorption")
    a_left          = 0
    a_right         = 1
    lam_0           = 14
    _title = "Fitting desorption data (dry N₂ / vapour)"
elif input == "liquid":
    DATA_PATH = Path(__file__).parent.parent.parent.parent / "data/experimental/DRY_LS_desorption_Hwang.csv"
    left_condition  = dict(state = "vapor", mode = "desorption")
    right_condition = dict(state = "liquid", mode = "sorption")
    a_left          = 0
    a_right         = 3
    lam_0           = 14
    _title = "Fitting desorption data (dry N₂ / liquid)"

K_NAMES = {0: "k_vw_sorp", 1: "k_vw_desorp", 2: "k_lw_sorp", 3: "k_lw_desorp"}
K_IDX   = {("vapor",  "sorption"):    0, ("vapor",  "desorption"):  1, ("liquid", "sorption"):    2, ("liquid", "desorption"):  3}

def _active_k_indices(left_cond, right_cond):
    """Return sorted list of k_params indices that are actually used as BCs."""
    idxs = set()
    for cond in (left_cond, right_cond):
        if cond["state"] == "vapor":   # liquid BCs bypass k entirely (use D_face)
            idxs.add(K_IDX[(cond["state"], cond["mode"])])
    return sorted(idxs)

def main():
    # Extract data
    df    = pd.read_csv(DATA_PATH)
    t_all = sorted(df["time_seconds"].unique())
    t0    = t_all[0]
    t_fit = [t for t in t_all if t > t0]
    t_fit = t_fit[-1:] #
    print(f"IC at t={t0} s  |  fitting to {t_fit} s")
    lam0 = np.ones(N_FVM) * lam_0
    x_lists, lam_lists = [], []
    for t in t_fit:
        sub = df[df["time_seconds"] == t].sort_values("normalized_thickness")  # type: ignore
        x_lists.append(sub["normalized_thickness"].values * L)
        lam_lists.append(sub["water_content"].values)

    active = _active_k_indices(left_condition, right_condition)
    print(f"Active k indices to fit: {[K_NAMES[i] for i in active]}")
    x0 = [k_params_ref[i] for i in active]
    lo = [1e-11] * len(active)
    hi = [1e-2]  * len(active)

    def cost_k_scalar(x):
        k_full = list(k_params_ref)
        for idx, val in zip(active, x):
            k_full[idx] = val
        residuals = cost_k(k_full, D_params, t_fit, x_lists, lam_lists, lam0, t0, a_left, a_right, left_condition, right_condition, DX)
        return np.sum(residuals**2)

    best = [np.inf]
    def callback(x, *_):
        f = cost_k_scalar(x)
        if f < best[0]:
            best[0] = f
            print(f"  improved -> f={f:.4e}  x={np.array2string(np.array(x), precision=2)}")

    result = minimize(
        cost_k_scalar,
        bounds=list(zip(lo, hi)),
        x0=x0,
        callback=callback,
    )
    print(result.message)

    k_params = list(k_params_ref)
    for idx, val in zip(active, result.x):
        k_params[idx] = val
    print("\nOptimal parameters:")
    for idx, val in zip(active, result.x):
        print(f"  {K_NAMES[idx]:<20} = {val:.4e}  m/s")
    n_res = sum(len(x) for x in x_lists)
    rmse  = np.sqrt(result.fun / n_res)
    print(f" RMSE(λ)              = {rmse:.4f}")
    print(" APE(λ)               = {:.2f} %".format(100 * rmse / np.mean([np.mean(lam) for lam in lam_lists])))
    update_params_file("k_params", k_params)
    sol = solve_ivp(
        membrane_ode, (t0, t_fit[-1]), lam0,
        args=(a_left, a_right, left_condition, right_condition, DX, D_params, k_params),
        method="BDF", t_eval=t_fit,  # type: ignore
    )
    n_t  = len(t_fit)
    cols = min(n_t, 4)
    rows = (n_t + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows),
                             sharex=True, sharey=True, squeeze=False)
    fig.suptitle(
         _title
    )

    for i, t in enumerate(t_fit):
        ax = axes[i // cols][i % cols]
        ax.scatter(x_lists[i] / L, lam_lists[i], color="k", s=15, zorder=5, label="Exp")
        if sol.success:
            ax.plot(X_NODES / L, sol.y[:, i], color="tab:blue", label="Fit")
        ax.set_title(f"t = {t} s")
        ax.set_xlabel("x / –")
        if i % cols == 0:
            ax.set_ylabel("λ")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n_t, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    plt.tight_layout()
    out_fig = Path(__file__).parent.parent.parent.parent / "results" / "experimental" / (Path(__file__).stem + ".png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved -> {out_fig}")


if __name__ == "__main__":
    main()