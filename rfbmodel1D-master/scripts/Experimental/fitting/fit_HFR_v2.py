"""
Fit membrane sorption/desorption kinetics by minimising the MSE
between simulated and experimental HFR.

Differences from fit_HFR.py:
  - Single S_desorp shared by both VW and LW channels (was S_vw_desorp + S_lw_desorp).
  - R_extra is a free fitting parameter in linear space (0 … HFR at lowest |j|).

Usage:
    python scripts/Experimental/fitting/fit_HFR_v2.py --test 5
    python scripts/Experimental/fitting/fit_HFR_v2.py --test 5 --workers 8 --maxiter 10 --popsize 20
    python scripts/Experimental/fitting/fit_HFR_v2.py --test 25 --workers 14
    python scripts/Experimental/fitting/fit_HFR_v2.py --test 1 --workers 7 --point-weights "-647.3988439:0.1, -797.6878613:0.1, -959.5375723:0.1,-1121.387283:0.1"
    python scripts/Experimental/fitting/fit_HFR_v2.py --test 2 --workers 7 --point-weights "-1073.170732:0.1,-1121.95122:0.1"
"""
import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import differential_evolution
from solve_dae.integrate import solve_dae
import pandas as pd
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _matplotlib_ok = True
except Exception:
    _matplotlib_ok = False
    print("[WARNING] matplotlib unavailable — plots will be skipped.", flush=True)

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from simulate_HFR import build_cell, build_atol, soc_cutoff_event  # noqa: E402
from rfbmodel1D.parameters.params import cellparams, membraneparams             # noqa: E402


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test",    type=float, default=5)
    p.add_argument("--workers", type=int,   default=-1,
                   help="Parallel workers for DE (-1 = all cores, 1 = serial).")
    p.add_argument("--maxiter", type=int,   default=10)
    p.add_argument("--popsize", type=int,   default=15)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--point-weights", type=str, default=None,
                   help="Comma-separated 'j:weight' overrides for per-point weights in "
                        "the MSE, matched against j_appl (A/m2) using the same sign "
                        "convention as the fit (charge positive, discharge negative). "
                        "E.g. \"750:0,-500:0.1\" excludes j=750 (charge) and down-weights "
                        "j=-500 (discharge) to 10%%. Points not listed default to weight 1.0. "
                        "Use this to de-emphasise points the model can't reproduce instead "
                        "of deleting them from the CSV.")
    return p.parse_args()


STRIDE_charge    = 1
STRIDE_discharge = 1
TF    = 1800
RTOL  = 1e-5

# Solver-stuck guard
_CHECK_EVERY      = 20_000
_MIN_DT           = 0.1
_MAX_WALL_S       = 360.0   # per-solve wall-clock budget [s]
_LAM_DRY          = 2.05    # flag membrane as dried when min(λ) falls below this
_HFR_PENALTY      = 1e1
_MSE_PENALTY      = 1e6


# ------------------------------------------------------------------------------
# Load experimental data (module-level so workers see it after import)
# ------------------------------------------------------------------------------
args = parse_args()
csv  = Path(__file__).parent.parent.parent.parent / "data/experimental/HFR.csv"
data = pd.read_csv(csv)
data = data[data["test_no"] == args.test]
soc  = data["SOC"].iloc[0]
T    = data["T(°C)"].iloc[0]
Q_v  = data["Q_v (ml/min)"].iloc[0]

j_charge    = data[data["mode"] == "charge"]["j_appl (A/m2)"][::STRIDE_charge]
j_discharge = data[data["mode"] == "discharge"]["j_appl (A/m2)"][::STRIDE_discharge]
HFR_exp     = np.concatenate([
    data[data["mode"] == "charge"]["value (ohm cm2)"][::STRIDE_charge],
    data[data["mode"] == "discharge"]["value (ohm cm2)"][::STRIDE_discharge],
])

# R_extra upper bound: experimental HFR at the lowest applied |current density|
j_all_abs   = np.concatenate([np.abs(j_charge), np.abs(j_discharge)])
idx_min_j   = int(np.argmin(j_all_abs))
R_EXTRA_MAX = float(HFR_exp[idx_min_j])   # Ω·cm²

# Per-point MSE weights, in the same (charge..., -discharge...) order as
# objective()'s j_all — lets specific points be down-weighted or fully
# excluded (weight 0) via --point-weights instead of edited out of the CSV.
def _parse_point_weights(spec: str | None) -> dict:
    if not spec:
        return {}
    overrides = {}
    for pair in spec.split(","):
        j_str, w_str = pair.split(":")
        overrides[float(j_str)] = float(w_str)
    return overrides

_j_all_signed      = np.array([float(j) for j in j_charge] + [-float(j) for j in j_discharge])
_point_weight_overrides = _parse_point_weights(args.point_weights)
POINT_WEIGHTS = np.ones_like(_j_all_signed)
for _j_val, _w in _point_weight_overrides.items():
    _idx = np.where(np.isclose(_j_all_signed, _j_val, atol=1.0))[0]
    if len(_idx) == 0:
        print(f"[WARNING] --point-weights: no point found matching j={_j_val}", flush=True)
    else:
        POINT_WEIGHTS[_idx] = _w
        print(f"  point weight override: j={_j_val:+.0f} A/m² -> weight={_w}", flush=True)


# ------------------------------------------------------------------------------
# Single-run HFR calculator with stuck-solver guard
# ------------------------------------------------------------------------------
def _compute_HFR(j_appl: float, r_extra_ohmcm2: float) -> float:
    cell, y0, yp0 = build_cell(soc, j_appl, T, Q_v)
    atol          = build_atol(cell)
    jac_sparsity  = cell.compute_jac_sparsity(y0, yp0)

    _state = {"calls": 0, "last_t": None, "t_wall": time.time()}

    def _residual(t, y, yp):
        _state["calls"] += 1
        if _state["calls"] % _CHECK_EVERY == 0:
            if time.time() - _state["t_wall"] > _MAX_WALL_S:
                raise RuntimeError(f"wall-timeout at t={t:.2f}")
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
    except RuntimeError as e:
        if "stuck" in str(e) or "wall-timeout" in str(e):
            print(f"  -> solver aborted ({e}), returning penalty {_HFR_PENALTY}", flush=True)
            return _HFR_PENALTY
        raise
    except Exception as e:
        print(f"  -> solver failed ({type(e).__name__}: {e}), "
              f"returning penalty {_HFR_PENALTY}", flush=True)
        return _HFR_PENALTY
    if sol.t[-1] < 0.999 *TF:
        print(f"  -> solver stopped early at t={sol.t[-1]:.2f} s (event?), "
              f"returning penalty {_HFR_PENALTY}", flush=True)
        return _HFR_PENALTY

    lam_final = cell.split_state(sol.y[:, -1])["membrane"][:, 0]
    if np.min(lam_final) < _LAM_DRY:
        print(f"  -> membrane dried (lam_min={np.min(lam_final):.3f}, j={j_appl:+.0f})", flush=True)

    cell.dae_residual_full_cell(sol.t[-1], sol.y[:, -1], sol.yp[:, -1])

    kappa_io_ca = cell.cathode.kappa_io
    sigma_ca    = np.ones_like(kappa_io_ca) * cell.cathode.eff_conductivity
    dx_ca       = cell.mesh.cathode_mesh.dx * cell.cathode.thickness
    R_cathode   = sum(dx_ca / (sigma_ca + kappa_io_ca))

    kappa_io_m = cell.membrane.kappa_io
    dx_m       = cell.mesh.membrane_mesh.dx * cell.membrane.thickness
    R_membrane = sum(dx_m / kappa_io_m)

    R_ACL  = cell.anode.thickness  / cell.anode.eff_conductivity
    R_AGDL = cell.agdl.thickness   / cell.agdl.eff_conductivity
    R_BP   = cell.thickness_BP / cell.sigma_BP
    R_CC   = cell.thickness_CC / cell.sigma_CC

    r_extra_ohm_m2 = r_extra_ohmcm2 / 1e4   # Ω·cm² -> Ω·m²
    return (R_cathode + R_membrane + R_ACL + R_AGDL + R_BP + R_CC + r_extra_ohm_m2) * 1e4


# ------------------------------------------------------------------------------
# Extended single-run HFR + lambda profile calculator (diagnostic use only)
# ------------------------------------------------------------------------------
def _compute_HFR_and_lambda(j_appl: float, r_extra_ohmcm2: float):
    """Like _compute_HFR but also returns (lam_profile, x_mem) at final timestep."""
    cell, y0, yp0 = build_cell(soc, j_appl, T, Q_v)
    atol          = build_atol(cell)
    jac_sparsity  = cell.compute_jac_sparsity(y0, yp0)

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
        print(f"  -> solver failed ({type(e).__name__}: {e}), skipping point", flush=True)
        return _HFR_PENALTY, None, None
    
    if sol.t[-1] < 0.999 * TF:
        print(f"  -> solver stopped early at t={sol.t[-1]:.2f} s (event?), skipping point", flush=True)
        return _HFR_PENALTY, None, None

    split       = cell.split_state(sol.y[:, -1])
    lam_profile = split["membrane"][:, 0]
    x_mem       = cell.mesh.membrane_mesh.x_grid

    cell.dae_residual_full_cell(sol.t[-1], sol.y[:, -1], sol.yp[:, -1])

    kappa_io_ca = cell.cathode.kappa_io
    sigma_ca    = np.ones_like(kappa_io_ca) * cell.cathode.eff_conductivity
    dx_ca       = cell.mesh.cathode_mesh.dx * cell.cathode.thickness
    R_cathode   = sum(dx_ca / (sigma_ca + kappa_io_ca))

    kappa_io_m = cell.membrane.kappa_io
    dx_m       = cell.mesh.membrane_mesh.dx * cell.membrane.thickness
    R_membrane = sum(dx_m / kappa_io_m)

    R_ACL  = cell.anode.thickness  / cell.anode.eff_conductivity
    R_AGDL = cell.agdl.thickness   / cell.agdl.eff_conductivity
    R_BP   = cell.thickness_BP / cell.sigma_BP
    R_CC   = cell.thickness_CC / cell.sigma_CC

    r_extra_ohm_m2 = r_extra_ohmcm2 / 1e4
    hfr = (R_cathode + R_membrane + R_ACL + R_AGDL + R_BP + R_CC + r_extra_ohm_m2) * 1e4

    return hfr, lam_profile, x_mem


# ------------------------------------------------------------------------------
# Objective (module-level so multiprocessing workers can import it)
# ------------------------------------------------------------------------------
def objective(k_vw_sorp, k_lw_sorp, S_desorp, k_ca) -> tuple:
    """
    Physical-space objective.  S_desorp is shared by both VW and LW channels.
    R_extra is analytically eliminated (variable projection):
      R_extra* = clip(mean(HFR_exp − HFR_sim_no_extra), 0, R_EXTRA_MAX)
    Returns (mse, r_extra_opt_ohmcm2).
    """
    membraneparams.k_sorp_vw   = k_vw_sorp
    membraneparams.k_desorp_vw = S_desorp * k_vw_sorp
    membraneparams.k_sorp_lw   = k_lw_sorp
    membraneparams.k_desorp_lw = S_desorp * k_lw_sorp
    membraneparams.k_ca_lw     = k_ca
    cellparams.vanadium_uptake = False
    membraneparams.sf_D        = 1

    t0 = time.time()
    try:
        j_all   = [float(j) for j in j_charge] + [-float(j) for j in j_discharge]
        HFR_sim = np.array([_compute_HFR(j, 0.0) for j in j_all])
    except Exception as e:
        print(f"  -> objective failed ({type(e).__name__}: {e}), "
              f"returning {_MSE_PENALTY}", flush=True)
        return _MSE_PENALTY, 0.0

    # Closed-form optimal offset, constrained to [0, R_EXTRA_MAX] — weighted
    # so down-weighted/excluded points don't bias the offset either.
    r_extra_opt = float(np.clip(
        np.average(HFR_exp - HFR_sim, weights=POINT_WEIGHTS), 0.0, R_EXTRA_MAX))
    HFR_sim_total = HFR_sim + r_extra_opt

    nc = len(j_charge)
    for i, j in enumerate(j_charge):
        w = POINT_WEIGHTS[i]
        tag = "  [excluded]" if w == 0 else (f"  [w={w:g}]" if w != 1 else "")
        print(f"  charge    j={j:+7.0f}  HFR={HFR_sim_total[i]:.4f} ohm.cm2{tag}", flush=True)
    for i, j in enumerate(j_discharge):
        w = POINT_WEIGHTS[nc+i]
        tag = "  [excluded]" if w == 0 else (f"  [w={w:g}]" if w != 1 else "")
        print(f"  discharge j={-j:+7.0f}  HFR={HFR_sim_total[nc+i]:.4f} ohm.cm2{tag}", flush=True)
    print(f"  k_vw_sorp={k_vw_sorp:.3e}  k_lw_sorp={k_lw_sorp:.3e}  "
          f"S_desorp={S_desorp:.3e}  k_ca={k_ca:.3e}  R_extra*={r_extra_opt:.4f} ohm.cm2",
          flush=True)

    mse = float(np.average((HFR_sim_total - HFR_exp) ** 2, weights=POINT_WEIGHTS))
    print(f"  MSE = {mse:.6f}  ({time.time()-t0:.1f}s)", flush=True)
    return mse, r_extra_opt


def objective_log(log_params) -> float:
    """DE searches in log10 space (4 kinetic params only); R_extra solved analytically."""
    k_vw_sorp, k_lw_sorp, S_desorp, k_ca = 10.0 ** np.asarray(log_params)
    mse, _ = objective(k_vw_sorp, k_lw_sorp, S_desorp, k_ca)
    return mse


# ------------------------------------------------------------------------------
# Fit
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Fitting — R_extra analytically eliminated in [0, {R_EXTRA_MAX:.4f}] ohm.cm2  "
          f"test {int(args.test)}", flush=True)
    print(f"  workers={args.workers}  maxiter={args.maxiter}  "
          f"popsize={args.popsize}  seed={args.seed}", flush=True)

    log_bounds = [
        (np.log10(1e-6), np.log10(1e-3)),   # log k_vw_sorp
        (np.log10(1e-5), np.log10(9e-3)),   # log k_lw_sorp
        (np.log10(1),    np.log10(10)),     # log S_desorp (single, shared VW+LW)
        (np.log10(1e-5), np.log10(9e-3)),   # log k_ca
    ]
    _names_k = ["k_vw_sorp", "k_lw_sorp", "S_desorp", "k_ca"]

    _best = {"x": None, "fun": np.inf, "nit": 0, "nfev": 0}

    def _callback(intermediate_result):
        gen = _best["nit"] + 1
        fun = intermediate_result.fun
        x   = intermediate_result.x
        if fun < _best["fun"]:
            _best["x"]   = x.copy()
            _best["fun"] = fun
        _best["nit"]  = gen
        _best["nfev"] = intermediate_result.nfev
        k_lin   = 10.0 ** x
        elapsed = time.time() - t_start
        print(f"\n{'-'*60}", flush=True)
        print(f"  Test {int(args.test)} | Generation {gen:3d} | best MSE={fun:.6f}"
              f"  RMSE={fun**0.5:.4f} ohm.cm2  |  {elapsed/3600:.2f} h elapsed", flush=True)
        print(f"  Re-evaluating best individual:", flush=True)
        _, r_extra_gen = objective(*k_lin)
        _best["r_extra"] = r_extra_gen
        print(f"{'-'*40}", flush=True)
        for name, v in zip(_names_k, k_lin):
            print(f"    {name:<15} {v:.4e}", flush=True)
        print(f"    {'R_extra':<15} {r_extra_gen:.4e} ohm.cm2", flush=True)
        print(f"  RTOL={RTOL}  sf_D={membraneparams.sf_D}  "
              f"vanadium_uptake={cellparams.vanadium_uptake}", flush=True)
        print(f"{'-'*60}", flush=True)

    t_start = time.time()
    try:
        result = differential_evolution(
            objective_log,
            bounds        = log_bounds,
            maxiter       = args.maxiter,
            popsize       = args.popsize,
            mutation      = (0.5, 1.0),
            recombination = 0.7,
            tol           = 1e-3,
            rng           = args.seed,
            workers       = args.workers,
            updating      = 'deferred',
            polish        = False,
            init          = 'sobol',
            disp          = True,
            callback      = _callback,
        )
    except KeyboardInterrupt:
        print("\n[interrupted] keeping best solution found so far.", flush=True)
        from types import SimpleNamespace
        result = SimpleNamespace(
            x       = _best["x"],
            fun     = _best["fun"],
            nit     = _best["nit"],
            nfev    = _best["nfev"],
            success = False,
            message = "interrupted by user",
        )
    t_elapsed = time.time() - t_start

    # Recover optimal R_extra for the best kinetic parameters
    k_best = 10.0 ** np.asarray(result.x)
    _, r_extra_best = objective(*k_best)

    print(f"\n{'='*60}")
    print(f"Test       = {int(args.test)}")
    print(f"Elapsed    = {t_elapsed/60:.1f} min  ({t_elapsed:.0f} s)")
    print(f"NFE        = {result.nfev}   (generations: {result.nit})")
    print(f"Converged  = {result.success}   ({result.message})")
    print(f"MSE        = {result.fun:.6f}")
    print(f"{'-'*60}")
    for name, v in zip(_names_k, k_best):
        print(f"{name:<22} {v:.4e}")
    print(f"{'R_extra (ohm.cm2)':<22} {r_extra_best:.4e}")

    out_dir  = Path(__file__).parent.parent.parent.parent / "results" / "test fitting" / f"test {int(args.test)}"
    out_dir.mkdir(exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"HFR_fit_test{int(args.test)}_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "test":      int(args.test),
            "conditions": {
                "soc": float(soc),
                "T_C": float(T),
                "Q_v_ml_min": float(Q_v),
            },
            "elapsed_s": t_elapsed,
            "nfev":      int(result.nfev),
            "nit":       int(result.nit),
            "success":   bool(result.success),
            "message":   str(result.message),
            "mse":       float(result.fun),
            "params":    {n: float(v) for n, v in zip(_names_k, k_best)}
                         | {"R_extra (Ω·cm²)": r_extra_best},
            "log_bounds": {n: list(b) for n, b in zip(_names_k, log_bounds)},
            "settings": {
                "maxiter":            args.maxiter,
                "popsize":            args.popsize,
                "seed":               args.seed,
                "workers":            args.workers,
                "stride_charge":      STRIDE_charge,
                "stride_discharge":   STRIDE_discharge,
                "tf":                 TF,
                "rtol":               RTOL,
                "sf_D":               membraneparams.sf_D,
                "vanadium_uptake":    cellparams.vanadium_uptake,
                "r_extra_max_ohmcm2": R_EXTRA_MAX,
                "point_weights": {f"{j:+.0f}": float(w)
                                   for j, w in zip(_j_all_signed, POINT_WEIGHTS) if w != 1},
            },
        }, f, indent=2)
    print(f"\nResult saved to {out_path}")

    # -- Diagnostic plots ------------------------------------------------------
    if not _matplotlib_ok:
        print("\n[WARNING] Plots skipped (matplotlib unavailable).")
    else:
        print("\nRunning final diagnostic pass for plots...", flush=True)
        nc = len(j_charge)
        nd = len(j_discharge)
        j_c = np.asarray(list(j_charge),    dtype=float)
        j_d = np.asarray(list(j_discharge), dtype=float)
        j_all_plot = list(j_c) + [-float(j) for j in j_d]

        HFR_plot   = []
        lam_charge = []
        lam_disc   = []
        x_mem_ref  = None

        for i, j in enumerate(j_all_plot):
            hfr, lam, x_mem = _compute_HFR_and_lambda(j, r_extra_best)
            HFR_plot.append(hfr)
            if x_mem is not None:
                x_mem_ref = x_mem
            if i < nc:
                lam_charge.append(lam)
                print(f"  charge    j={abs(j):7.0f}  HFR={hfr:.4f} ohm.cm2", flush=True)
            else:
                lam_disc.append(lam)
                print(f"  discharge j={abs(j):7.0f}  HFR={hfr:.4f} ohm.cm2", flush=True)

        HFR_plot = np.array(HFR_plot)

        # HFR figure
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(j_c, HFR_plot[:nc],  'b-o',  lw=2, ms=6, label='Charge (sim)')
        ax.plot(j_d, HFR_plot[nc:],  'r-o',  lw=2, ms=6, label='Discharge (sim)')
        ax.plot(j_c, HFR_exp[:nc],   'b^--', lw=1.5, ms=7, mfc='none', label='Charge (exp)')
        ax.plot(j_d, HFR_exp[nc:],   'r^--', lw=1.5, ms=7, mfc='none', label='Discharge (exp)')
        ax.set_xlabel('Applied current density  [A/m²]')
        ax.set_ylabel('HFR  [Ω·cm²]')
        ax.set_title(
            f'Test {int(args.test)}  |  SOC={soc}  T={T}°C  Q_v={Q_v} ml/min\n'
            f'k_vw={k_best[0]:.2e}  S_des={k_best[2]:.2e}  '
            f'k_lw={k_best[1]:.2e}  k_ca={k_best[3]:.2e}\n'
            f'R_extra={r_extra_best:.4f} Ω·cm²  MSE={result.fun:.4f}'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        hfr_plot_path = out_dir / f"HFR_fit_test{int(args.test)}_{stamp}_hfr.png"
        fig.savefig(hfr_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"HFR plot saved to {hfr_plot_path}")

        # Lambda profile figure
        if x_mem_ref is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

            colors_c = plt.cm.Blues(np.linspace(0.4, 0.95, max(nc, 1))) # type: ignore
            for lam, j, c in zip(lam_charge, j_c, colors_c):
                if lam is not None:
                    axes[0].plot(x_mem_ref, lam, color=c, lw=2, label=f'{j:.0f} A/m²')
            axes[0].set_xlabel('Normalised membrane position  (cathode -> anode)')
            axes[0].set_ylabel('Water content  λ')
            axes[0].set_title('Charge')
            axes[0].legend(fontsize=7, loc='best')
            axes[0].grid(True, alpha=0.3)

            colors_d = plt.cm.Reds(np.linspace(0.4, 0.95, max(nd, 1))) # type: ignore
            for lam, j, c in zip(lam_disc, j_d, colors_d):
                if lam is not None:
                    axes[1].plot(x_mem_ref, lam, color=c, lw=2, label=f'{j:.0f} A/m²')
            axes[1].set_xlabel('Normalised membrane position  (cathode -> anode)')
            axes[1].set_title('Discharge')
            axes[1].legend(fontsize=7, loc='best')
            axes[1].grid(True, alpha=0.3)

            fig.suptitle(
                f'λ profile at t_final  |  Test {int(args.test)}  SOC={soc}\n'
                f'k_vw={k_best[0]:.2e}  S_des={k_best[2]:.2e}  '
                f'k_lw={k_best[1]:.2e}  k_ca={k_best[3]:.2e}  '
                f'R_extra={r_extra_best:.4f} Ω·cm²',
                fontsize=11
            )
            fig.tight_layout()
            lam_plot_path = out_dir / f"HFR_fit_test{int(args.test)}_{stamp}_lambda.png"
            fig.savefig(lam_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Lambda plot saved to {lam_plot_path}")
