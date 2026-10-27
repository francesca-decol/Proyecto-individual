"""
Fits the 8-parameter D(λ) model to transient desorption data.
Initial condition: λ=14 uniform (vapor-equilibrated membrane, t=0).
Fit targets: all measured time points.
Usage:  python scripts/Experimental/water_fit_diffusion.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from scripts.functions.membrane_model import N_FVM, membrane_ode, cost_D, cost_all, update_params_file
from rfbmodel1D.parameters.params import k_params

# Input
DATA_PATH = Path(__file__).parent.parent.parent.parent / "data/experimental/DRY_LS_desorption_Hwang.csv"
left_condition  = dict(state = "vapor", mode = "desorption")
right_condition = dict(state = "liquid", mode = "sorption")
a_left          = 0
a_right         = 3
L               = 119e-6 # m
lam0            = np.ones(N_FVM) * 14 # Initial condition: vapour-equilibrated membrane 
# Main
DX      = L / N_FVM
X_NODES = (np.arange(N_FVM) + 0.5) * DX
def main():
    df    = pd.read_csv(DATA_PATH)
    t_all = sorted(df["time_seconds"].unique())[-1:]
    t0    = 0
    t_fit = list(t_all)
    x_lists, lam_lists = [], []
    for t in t_fit:
        mask = df[df["time_seconds"] == t].sort_values("normalized_thickness")  # type: ignore
        x_lists.append(mask["normalized_thickness"].values * L)
        lam_lists.append(mask["water_content"].values)
    #     a        b         D_plat    lam1   w1     lam2   w2     k
    x0 = [2e-11, 6e-11,  1.25e-10,     7.0,   1.5,   18.0,  1.0,   0.35]
    lo = [1e-11, 1e-11,  1e-11,        3.0,   0.3,   15,    0.2,   0.05]
    hi = [1e-9, 5e-10,   2e-10,        12.0,  4.0,   20.5,  3.0,   1.5 ]

    def cost_D_scalar(x):
        residuals = cost_D(x, k_params, t_fit, x_lists, lam_lists, lam0, t0, a_left, a_right, left_condition, right_condition, DX)
        return np.sum(residuals**2)

    result = minimize(
        cost_D_scalar,
        x0=x0,
        bounds=list(zip(lo, hi)))
    
    print(result.message)
    p_opt = result.x
    print(p_opt)
    names = ["a", "b", "D_plat", "lam1", "w1", "lam2", "w2", "k"]
    print("\nOptimal parameters:")
    for name, val in zip(names, p_opt):
        print(f"  {name:8s} = {val:.4e}")
    residuals = cost_D(p_opt, k_params, t_fit, x_lists, lam_lists, lam0, t0, a_left, a_right, left_condition, right_condition, DX)
    lam_exp_all = np.concatenate(lam_lists)
    ape = 100 * np.mean(np.abs(residuals / lam_exp_all))
    print(f"  RMSE(λ) = {np.sqrt(np.mean(residuals**2)):.4f}")
    print(f"  APE(λ)  = {ape:.2f} %")
    update_params_file("D_params", p_opt)

    sol = solve_ivp(membrane_ode, (t0, t_fit[-1]), lam0,
                    args=(a_left, a_right, left_condition, right_condition, DX, p_opt[:8], k_params),
                    method="BDF", t_eval=t_fit)

    fig, ax = plt.subplots(1, len(t_fit), figsize=(4 * len(t_fit), 4), sharex=True, sharey=True)

    
    ax.scatter(x_lists[0] / L, lam_lists[0], color="k", s=10, label="Exp")
    ax.plot(X_NODES / L, sol.y[:, 0], color="tab:blue", label="Fit")
    ax.set_title(f"t = {t_fit[0]} s")
    ax.set_xlabel("x / –")
    if 0 == 0:
        ax.set_ylabel("λ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_fig = Path(__file__).parent / "fit_transient_diffusivity.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"Figure saved -> {out_fig}")


if __name__ == "__main__":
    main()