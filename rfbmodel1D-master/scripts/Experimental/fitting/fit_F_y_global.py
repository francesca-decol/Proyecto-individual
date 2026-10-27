"""
Global fit for cathode F_y (activity coefficient in the Nernst OCV).

Finds the single F_y that minimises the E_cell error across all tests in a
global fit JSON, using ONLY the lowest current density point per test (one
charge + one discharge) — so each F_y evaluation needs very few DAE solves.

All other parameters (k_ca, j0_an, membrane params) are loaded from existing
fit files and held fixed.

Usage
-----
  python scripts/Experimental/fitting/fit_F_y_global.py
  python scripts/Experimental/fitting/fit_F_y_global.py --bounds 5 100
  python scripts/Experimental/fitting/fit_F_y_global.py --global-json "results\test fitting\global\global_k_ca_fit_20260821_104300.json"
  python scripts/Experimental/fitting/fit_F_y_global.py --global-json "results\test fitting\global\global_k_ca_fit_20260822_120312.json"
"""
import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize_scalar
from solve_dae.integrate import solve_dae

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from simulate_HFR import build_cell, build_atol, soc_cutoff_event       # noqa: E402
from rfbmodel1D.parameters.params import (                                # noqa: E402
    cathodeparams, anodeparams, membraneparams, cellparams)
import pandas as pd


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--global-json", type=str, default=None,
                   help="Path to the global k_ca fit JSON (default: latest in results/test fitting/global/)")
    p.add_argument("--bounds", type=float, nargs=2, default=[10, 60],
                   help="Search bounds for F_y (default: 10 60)")
    p.add_argument("--xatol", type=float, default=0.05,
                   help="Absolute tolerance for the F_y search (default: 0.05)")
    p.add_argument("--maxiter", type=int, default=30,
                   help="Maximum iterations for the F_y search (default: 30)")
    return p.parse_args()


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
TF           = 1800
RTOL         = 1e-5
_CHECK_EVERY = 20_000
_MIN_DT      = 0.5
_E_PENALTY   = 10.0   # Large penalty voltage for failed simulations

_ROOT    = Path(__file__).parent.parent.parent.parent
_CSV     = _ROOT / "data" / "experimental" / "E_cell.csv"
_RES     = _ROOT / "results" / "test fitting"
_RES_GLB = _RES / "global"


def _discover_global_json() -> Path:
    """Find the latest global_k_ca_fit_*.json."""
    candidates = sorted(_RES_GLB.glob("global_k_ca_fit_*.json"))
    if not candidates:
        raise FileNotFoundError("No global_k_ca_fit_*.json found in " + str(_RES_GLB))
    return candidates[-1]


def _discover_hfr_json(test_no: int) -> Path:
    """Find the latest HFR fit JSON for a given test number."""
    candidates = sorted((_RES / f"test {test_no}").glob(f"HFR_fit_test{test_no}_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No HFR fit JSON found for test {test_no} in "
            f"results/test fitting/test {test_no}/. Run fit_HFR_v2.py first."
        )
    return candidates[-1]


# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------
def _load_test_data(test_no: int, global_fit: dict, all_data: pd.DataFrame) -> dict:
    """Return conditions, lowest-j points, and fitted params for one test."""
    d   = all_data[all_data["test_no"] == test_no]
    soc = float(d["SOC"].iloc[0])
    T   = float(d["T(°C)"].iloc[0])
    Q_v = float(d["Q_v (ml/min)"].iloc[0])

    # Lowest |j| charge point
    charge = d[d["mode"] == "charge"].sort_values("j_appl (A/m2)", key=abs)
    j_ch   = float(charge["j_appl (A/m2)"].iloc[0])
    E_ch   = float(charge["value (V)"].iloc[0])

    # Lowest |j| discharge point
    discharge = d[d["mode"] == "discharge"].sort_values("j_appl (A/m2)", key=abs)
    j_dis  = float(discharge["j_appl (A/m2)"].iloc[0])
    E_dis  = float(discharge["value (V)"].iloc[0])

    # j0_an for this test from global fit
    j0_an = global_fit["params"]["j0_an"][str(test_no)]

    # HFR membrane params
    with open(_discover_hfr_json(test_no)) as f:
        hfr = json.load(f)
    _p, _s = hfr["params"], hfr["settings"]
    hfr_params = dict(
        k_sorp_vw   = _p["k_vw_sorp"],
        k_desorp_vw = _p["S_desorp"] * _p["k_vw_sorp"],
        k_sorp_lw   = _p["k_lw_sorp"],
        k_desorp_lw = _p["S_desorp"] * _p["k_lw_sorp"],
        k_ca_lw     = _p["k_ca"],
        sf_D        = _s["sf_D"],
        vanadium_uptake = _s["vanadium_uptake"],
    )

    return dict(
        test_no=test_no, soc=soc, T=T, Q_v=Q_v,
        j_charge=j_ch, E_exp_charge=E_ch,
        j_discharge=j_dis, E_exp_discharge=E_dis,
        j0_an=j0_an, hfr=hfr_params,
    )


# ------------------------------------------------------------------------------
# DAE forward simulation
# ------------------------------------------------------------------------------
def _apply_params(cfg: dict, k_ca: float, F_y: float) -> None:
    """Set all parameters for this test into the global singletons."""
    # HFR membrane params
    h = cfg["hfr"]
    membraneparams.k_sorp_vw   = h["k_sorp_vw"]
    membraneparams.k_desorp_vw = h["k_desorp_vw"]
    membraneparams.k_sorp_lw   = h["k_sorp_lw"]
    membraneparams.k_desorp_lw = h["k_desorp_lw"]
    membraneparams.k_ca_lw     = h["k_ca_lw"]
    membraneparams.sf_D        = h["sf_D"]
    cellparams.vanadium_uptake = h["vanadium_uptake"]
    # Kinetics
    cathodeparams.reaction_rate_constant = k_ca
    anodeparams.j_0 = cfg["j0_an"]
    # Activity coefficient
    cathodeparams.F_y = F_y


def _run_E_cell(j_appl: float, cfg: dict) -> float:
    """Steady-state E_cell for one operating point. Returns penalty on failure."""
    cell, y0, yp0 = build_cell(cfg["soc"], j_appl, cfg["T"], cfg["Q_v"])
    atol         = build_atol(cell)
    jac_sparsity = cell.compute_jac_sparsity(y0, yp0)

    _state = {"calls": 0, "last_t": None}

    def _residual(t, y, yp):
        _state["calls"] += 1
        if _state["calls"] % _CHECK_EVERY == 0:
            if _state["last_t"] is not None and (t - _state["last_t"]) < _MIN_DT:
                raise RuntimeError(f"stuck@t={t:.4f}")
            _state["last_t"] = t
        return cell.dae_residual_full_cell(t, y, yp)

    try:
        sol = solve_dae(
            _residual,
            (0.0, TF), y0, yp0,
            method="BDF", atol=atol, rtol=RTOL,
            events=soc_cutoff_event,
            jac_sparsity=jac_sparsity,
        )
    except Exception as e:
        print(f"    -> solve failed j={j_appl:+.0f} test {cfg['test_no']}: {e}", flush=True)
        return _E_PENALTY

    if sol.t[-1] < 0.999 * TF:
        print(f"    -> early stop j={j_appl:+.0f} test {cfg['test_no']} t={sol.t[-1]:.1f}s", flush=True)
        return _E_PENALTY

    y_f       = sol.y[:, -1]
    Nc        = cell.mesh.cathode_mesh.n_x
    n_vars_ca = cell.layout.n_vars_cathode
    off_ca    = cell.layout.n_vars_tank
    Phi_el    = y_f[off_ca:off_ca + Nc*n_vars_ca].reshape(Nc, n_vars_ca)[:, 4]
    dx        = cell.mesh.cathode_mesh.dx[0] * cell.cathode.thickness
    return float(Phi_el[0] + j_appl * (dx / 2) / cell.cathode.eff_conductivity)


# ------------------------------------------------------------------------------
# Objective
# ------------------------------------------------------------------------------
def _objective(F_y: float, k_ca: float, configs: list) -> float:
    """MSE across all tests at lowest-j points for a given F_y."""
    errors_sq = []
    for cfg in configs:
        _apply_params(cfg, k_ca, F_y)
        # Charge point
        E_sim_ch = _run_E_cell(cfg["j_charge"], cfg)
        errors_sq.append((E_sim_ch - cfg["E_exp_charge"]) ** 2)
        # Discharge point
        E_sim_dis = _run_E_cell(-cfg["j_discharge"], cfg)
        errors_sq.append((E_sim_dis - cfg["E_exp_discharge"]) ** 2)

    mse = float(np.mean(errors_sq))
    print(f"  F_y = {F_y:.4f}  MSE = {mse:.6f}", flush=True)
    return mse


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    args = parse_args()

    # Load global fit
    if args.global_json:
        global_json_path = Path(args.global_json)
    else:
        global_json_path = _discover_global_json()
    print(f"Global fit: {global_json_path.name}", flush=True)

    with open(global_json_path) as f:
        global_fit = json.load(f)

    k_ca  = global_fit["params"]["k_ca"]
    tests = global_fit["tests"]
    print(f"k_ca = {k_ca:.4e}", flush=True)
    print(f"Tests: {tests}", flush=True)

    # Load experimental data
    all_data = pd.read_csv(_CSV)
    configs  = [_load_test_data(t, global_fit, all_data) for t in tests]

    print(f"\nFitting F_y over {len(configs)} tests, lowest |j| charge + discharge per test")
    print(f"Search bounds: [{args.bounds[0]}, {args.bounds[1]}]")
    print(f"Current default: F_y = {cathodeparams.F_y}\n", flush=True)

    # Print what we're simulating
    for cfg in configs:
        print(f"  test {cfg['test_no']:2d}: SOC={cfg['soc']:.1f}  T={cfg['T']}°C  Q_v={cfg['Q_v']}  "
              f"j_ch={cfg['j_charge']:+.1f}  j_dis={cfg['j_discharge']:+.1f}  "
              f"j0_an={cfg['j0_an']:.1f}", flush=True)
    print(flush=True)

    t0 = time.time()
    result = minimize_scalar(
        _objective,
        bounds=args.bounds,
        args=(k_ca, configs),
        method="bounded",
        options={"xatol": args.xatol, "maxiter": args.maxiter},
    )

    elapsed = time.time() - t0
    F_y_opt = float(result.x)
    mse_opt = float(result.fun)

    print(f"\n{'='*60}")
    print(f"  Optimal F_y = {F_y_opt:.4f}")
    print(f"  MSE         = {mse_opt:.6f}")
    print(f"  RMSE        = {np.sqrt(mse_opt):.4f} V")
    print(f"  Elapsed     = {elapsed:.0f} s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    # Save result
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "parameter": "F_y",
        "F_y_optimal": F_y_opt,
        "mse": mse_opt,
        "rmse": float(np.sqrt(mse_opt)),
        "bounds": args.bounds,
        "tests": tests,
        "k_ca": k_ca,
        "global_fit_source": global_json_path.name,
        "elapsed_s": elapsed,
        "n_func_evals": int(result.nfev),
        "per_test": {},
    }

    # Re-run at optimal to get per-test errors
    _apply_params(configs[0], k_ca, F_y_opt)
    for cfg in configs:
        _apply_params(cfg, k_ca, F_y_opt)
        E_sim_ch  = _run_E_cell(cfg["j_charge"], cfg)
        E_sim_dis = _run_E_cell(-cfg["j_discharge"], cfg)
        out["per_test"][str(cfg["test_no"])] = {
            "E_exp_charge": cfg["E_exp_charge"],
            "E_sim_charge": E_sim_ch,
            "error_charge": E_sim_ch - cfg["E_exp_charge"],
            "E_exp_discharge": cfg["E_exp_discharge"],
            "E_sim_discharge": E_sim_dis,
            "error_discharge": E_sim_dis - cfg["E_exp_discharge"],
        }

    out_path = _RES_GLB / f"F_y_fit_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

