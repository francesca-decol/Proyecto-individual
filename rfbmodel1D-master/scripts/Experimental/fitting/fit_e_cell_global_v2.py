"""
Global fit for the cathode reaction rate constant and anode exchange current — v2.

Same physical model and objective as fit_e_cell_global.py, but a different
search STRATEGY for Stage 1. The original script ran a single joint 4-D
differential_evolution over [k_ca, j0_an_test1, j0_an_test2, j0_an_test3],
which needs a population of popsize*4 individuals per generation and pays
for 66 DAE simulations on every one of them — the joint search wastes most
of its budget exploring the j0_an directions, even though the objective is
EXACTLY separable in them once k_ca is fixed (each test's MSE only depends
on its own j0_an):

    MSE(k_ca, j0_1, j0_2, j0_3) = [SSE_1(k_ca, j0_1) + SSE_2(k_ca, j0_2)
                                    + SSE_3(k_ca, j0_3)] / N_total

so for a *given* k_ca, minimising jointly over (j0_1, j0_2, j0_3) is
mathematically identical to minimising each SSE_i independently over j0_i.

v2 exploits this:
  - k_ca is searched with a coarse-to-fine PARALLEL GRID (not DE): a wide
    log-spaced scan, then a narrower scan around the best point. Grid
    points are embarrassingly parallel (unlike DE generations, which are
    serial), so this maps far better onto a multi-core machine.
  - For every k_ca candidate, each test's j0_an is optimised independently
    via scipy.optimize.minimize_scalar (bounded Brent) — a smooth 1-D
    bounded problem that needs ~10-20 evaluations instead of DE's implicit
    population*generations exploration, and the 3 tests are mathematically
    independent so there is no need to search them jointly.
  - A final tight-tolerance per-test polish (loose grid tolerance -> tight)
    gives the reported j0_an values.

Stage 2 (k_ca fixed, single test) collapses to exactly the inner problem
above — no DE, no grid, just one bounded 1-D search.

Usage
-----
  python scripts/Experimental/fitting/fit_e_cell_global_v2.py --stage 1 --workers 10 --tests 1,4,7,10,13,18,21,24,27;
  python scripts/Experimental/fitting/fit_e_cell_global_v2.py --stage 1 --workers 10 --tests 2,5,8,11,16,19,22,25,28,14,15;
  python scripts/Experimental/fitting/fit_e_cell_global_v2.py --stage 1 --workers 10 --tests 3,6,9,12,17,20,23,26,29;

  python scripts/Experimental/fitting/fit_e_cell_global_v2.py --stage 2 --test 5 --workers 7

  # Override the k_ca used in Stage 2 explicitly
  python scripts/Experimental/fitting/fit_e_cell_global_v2.py --stage 2 --test 5 --k-ca 4e-7

  # Tune grid resolution / inner search tightness
  python scripts/Experimental/fitting/fit_e_cell_global_v2.py --stage 1 --grid-points 24 --refine-points 12
"""
import sys
import os
import json
import argparse
import time
import multiprocessing as mp
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize_scalar
from solve_dae.integrate import solve_dae
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from simulate_HFR import build_cell, build_atol, soc_cutoff_event       # noqa: E402
from rfbmodel1D.parameters.params import (                                # noqa: E402
    cathodeparams, anodeparams, membraneparams, cellparams)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",   type=int,   default=1, choices=[1, 2],
                   help="1 = fit shared k_ca; 2 = fit j0_an for a single test")
    p.add_argument("--test",    type=float, default=None,
                   help="Test number for Stage 2 (5, 16, or 25)")
    p.add_argument("--tests",   type=str,   default=None,
                   help="Comma-separated test numbers for Stage 1 (e.g. '5,16,25'). "
                        "Overrides the hardcoded TESTS_T32 default.")
    p.add_argument("--k-ca",   type=float, default=None,
                   help="Fixed k_ca [m/s] for Stage 2. If omitted, loads from the "
                        "latest Stage 1 result in results/test fitting/global/")
    p.add_argument("--workers", type=int,   default=-1,
                   help="Parallel workers for the k_ca grid rounds (-1 = all cores)")
    p.add_argument("--grid-points",   type=int,   default=16,
                   help="Round-1 coarse grid size over k_ca")
    p.add_argument("--refine-points", type=int,   default=8,
                   help="Round-2 fine grid size over k_ca, centred on the round-1 best")
    p.add_argument("--refine-span",   type=float, default=1.5,
                   help="Round-2 half-width, in units of the round-1 grid spacing")
    p.add_argument("--inner-xatol",   type=float, default=0.03,
                   help="log10 tolerance for the per-test j0_an search during grid rounds")
    p.add_argument("--inner-maxiter", type=int,   default=20,
                   help="Max evaluations for the per-test j0_an search during grid rounds")
    p.add_argument("--polish-xatol",   type=float, default=1e-3,
                   help="log10 tolerance for the final per-test j0_an refine")
    p.add_argument("--polish-maxiter", type=int,   default=40,
                   help="Max evaluations for the final per-test j0_an refine")
    p.add_argument("--context",  type=str,   default=None,
                   help="Free-form note on parameter changes for this run relative to "
                        "other fits (e.g. \"SSA=2e-4, alpha=0.5\"). Saved into the output "
                        "JSON so results stay traceable to the params.py values used.")
    p.add_argument("--point-weights", type=str, default=None,
                   help="Comma-separated 'test:j:weight' overrides for per-point weights "
                        "in each test's MSE, matched against j_appl (A/m2) using the same "
                        "sign convention as the fit (charge positive, discharge negative). "
                        "E.g. \"1:-650.25:0.1,1:-807.88:0.1,2:-1123.15:0.1\" down-weights "
                        "specific discharge points to 10%% for tests 1 and 2 — same "
                        "mechanism as fit_HFR_v2.py's --point-weights, for points the model "
                        "can't reproduce. Points not listed default to weight 1.0.")
    return p.parse_args()


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
TESTS_T32    = [5, 16, 25]          # all tests at T = 32 °C
STRIDE       = 1
TF           = 1800
RTOL         = 1e-5
_CHECK_EVERY = 20_000
_MIN_DT      = 0.5
_E_PENALTY   = 1e3
_MSE_PENALTY = 1e6

_ROOT    = Path(__file__).parent.parent.parent.parent
_CSV     = _ROOT / "data/experimental/E_cell.csv"
_RES     = _ROOT / "results" / "test fitting"
_RES_GLB = _RES / "global"

def _discover_hfr_json(test_no: int) -> Path:
    """Find the latest HFR fit JSON for a given test number."""
    candidates = sorted((_RES / f"test {test_no}").glob(f"HFR_fit_test{test_no}_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No HFR fit JSON found for test {test_no} in "
            f"results/test fitting/test {test_no}/. Run fit_HFR_v2.py first."
        )
    return candidates[-1]

# Log-space bounds. k_ca bounds are the old v1 range (1e-8, 9e-6) divided by
# ~5000: the c_H_plus**(2-2*alpha) term added to i_0_ca inflates the effective
# reaction rate by that factor relative to when the old bounds/fits were made,
# so a k_ca fit under the old bounds is now ~5000x too aggressive and causes
# widespread DAE-initialization failures at harsher operating points (see
# test 1: SOC=10%, low Q_v — fails at every point across the old range).
_LOG_BOUNDS_K_CA  = (np.log10(2e-12), np.log10(1.8e-9))  # [m/s]
_LOG_BOUNDS_J0_AN = (np.log10(9e2),   np.log10(9e4))     # [A/m²]


# ------------------------------------------------------------------------------
# Load experimental data and HFR params at module level so that
# multiprocessing workers (which re-import this module) pick them up.
# ------------------------------------------------------------------------------
args      = parse_args()
_all_data = pd.read_csv(_CSV)


def _parse_point_weights(spec: str | None) -> dict:
    """Parse '--point-weights' into {test_no: {j_appl: weight}}."""
    if not spec:
        return {}
    overrides: dict = {}
    for triple in spec.split(","):
        t_str, j_str, w_str = triple.split(":")
        overrides.setdefault(int(t_str), {})[float(j_str)] = float(w_str)
    return overrides


_POINT_WEIGHT_OVERRIDES = _parse_point_weights(args.point_weights)


def _load_test_config(test_no: int) -> dict:
    """Return data + HFR params for one test."""
    d   = _all_data[_all_data["test_no"] == test_no]
    soc = float(d["SOC"].iloc[0])
    T   = float(d["T(°C)"].iloc[0])
    Q_v = float(d["Q_v (ml/min)"].iloc[0])

    j_ch  = d[d["mode"] == "charge"]["j_appl (A/m2)"][::STRIDE].values.astype(float)
    j_dis = d[d["mode"] == "discharge"]["j_appl (A/m2)"][::STRIDE].values.astype(float)
    E_exp = np.concatenate([
        d[d["mode"] == "charge"]["value (V)"][::STRIDE].values,
        d[d["mode"] == "discharge"]["value (V)"][::STRIDE].values,
    ]).astype(float)

    # HFR membrane params
    with open(_discover_hfr_json(test_no)) as f:
        hfr = json.load(f)
    _p, _s = hfr["params"], hfr["settings"]

    # Per-point MSE weights, in the same (charge..., -discharge...) order as
    # _run_test's j_all — lets specific points be down-weighted or fully
    # excluded (weight 0), instead of edited out of the CSV.
    #
    # Auto-loaded from this test's own HFR fit (fit_HFR_v2.py's
    # --point-weights, saved into settings.point_weights there): HFR.csv and
    # E_cell.csv log slightly different current values for the same nominal
    # setpoint, so matching is by NEAREST point, not exact — the same
    # membrane-drying / non-reproducible points flagged during HFR fitting
    # get the same down-weighting here, with no need to re-specify them.
    # --point-weights on the CLI is applied on top and wins on conflicts.
    j_all_signed = np.concatenate([j_ch, -j_dis])
    weights = np.ones_like(j_all_signed)

    _hfr_point_weights = _s.get("point_weights", {})
    _NEAREST_MATCH_TOL = 15.0  # [A/m2] HFR.csv vs E_cell.csv grid mismatch is a few A/m2 in practice
    for j_str, w in _hfr_point_weights.items():
        j_hfr = float(j_str)
        idx = int(np.argmin(np.abs(j_all_signed - j_hfr)))
        diff = abs(j_all_signed[idx] - j_hfr)
        if diff > _NEAREST_MATCH_TOL:
            print(f"[WARNING] test {test_no}: HFR point_weights j={j_hfr:+.0f} has no "
                  f"close E_cell match (nearest is {j_all_signed[idx]:+.0f}, "
                  f"diff={diff:.1f} A/m² > {_NEAREST_MATCH_TOL}), skipping", flush=True)
        else:
            weights[idx] = w
            print(f"  point weight (from HFR fit): test {test_no}  "
                  f"j={j_all_signed[idx]:+.0f} A/m² (HFR j={j_hfr:+.0f})  -> weight={w}",
                  flush=True)

    for j_val, w in _POINT_WEIGHT_OVERRIDES.get(test_no, {}).items():
        idx = np.where(np.isclose(j_all_signed, j_val, atol=1.0))[0]
        if len(idx) == 0:
            print(f"[WARNING] --point-weights: test {test_no} has no point "
                  f"matching j={j_val}", flush=True)
        else:
            weights[idx] = w
            print(f"  point weight override (CLI): test {test_no}  j={j_val:+.0f} A/m² "
                  f"-> weight={w}", flush=True)

    hfr_params = dict(
        k_sorp_vw   = _p["k_vw_sorp"],
        k_desorp_vw = _p["S_desorp"] * _p["k_vw_sorp"],
        k_sorp_lw   = _p["k_lw_sorp"],
        k_desorp_lw = _p["S_desorp"] * _p["k_lw_sorp"],
        k_ca_lw     = _p["k_ca"],
        sf_D        = _s["sf_D"],
        vanadium_uptake = _s["vanadium_uptake"],
    )

    return dict(test_no=test_no, soc=soc, T=T, Q_v=Q_v,
                j_charge=j_ch, j_discharge=j_dis, E_exp=E_exp,
                weights=weights, j_all_signed=j_all_signed, hfr=hfr_params)


# Load configs needed for this run
if args.stage == 1:
    _tests_active = (
        [int(t.strip()) for t in args.tests.split(",")]
        if args.tests else TESTS_T32
    )
    _test_configs = [_load_test_config(t) for t in _tests_active]
else:
    if args.test is None:
        raise SystemExit("--test is required for Stage 2")
    _test_configs = [_load_test_config(int(args.test))]

_n_points_per_test = {
    int(cfg["test_no"]): len(cfg["j_charge"]) + len(cfg["j_discharge"])
    for cfg in _test_configs
}


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _apply_hfr(cfg: dict) -> None:
    """Write this test's HFR membrane params into the global singletons."""
    h = cfg["hfr"]
    membraneparams.k_sorp_vw   = h["k_sorp_vw"]
    membraneparams.k_desorp_vw = h["k_desorp_vw"]
    membraneparams.k_sorp_lw   = h["k_sorp_lw"]
    membraneparams.k_desorp_lw = h["k_desorp_lw"]
    membraneparams.k_ca_lw     = h["k_ca_lw"]
    membraneparams.sf_D        = h["sf_D"]
    cellparams.vanadium_uptake = h["vanadium_uptake"]

def _run_E_cell(j_appl: float, cfg: dict) -> float:
    """Steady-state E_cell for one operating point. Returns penalty on failure."""
    _apply_hfr(cfg)
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
    except RuntimeError as e:
        if "stuck" in str(e):
            print(f"  -> solver stuck ({e}), penalty", flush=True)
            return _E_PENALTY
        raise
    except Exception as e:
        print(f"  -> solver failed ({type(e).__name__}: {e}), penalty", flush=True)
        return _E_PENALTY

    if sol.t[-1] < 0.999 * TF:
        print(f"  -> early stop t={sol.t[-1]:.1f} s, penalty", flush=True)
        return _E_PENALTY

    y_f       = sol.y[:, -1]
    Nc        = cell.mesh.cathode_mesh.n_x
    n_vars_ca = cell.layout.n_vars_cathode
    off_ca    = cell.layout.n_vars_tank
    Phi_el    = y_f[off_ca:off_ca + Nc*n_vars_ca].reshape(Nc, n_vars_ca)[:, 4]
    dx        = cell.mesh.cathode_mesh.dx[0] * cell.cathode.thickness
    return float(Phi_el[0] + j_appl * (dx / 2) / cell.cathode.eff_conductivity)


def _run_test(cfg: dict) -> tuple[list[float], np.ndarray]:
    """Run all charge+discharge points for one test config. Returns (E_sim, E_exp)."""
    j_all = list(cfg["j_charge"]) + [-j for j in cfg["j_discharge"]]
    E_sim = [_run_E_cell(j, cfg) for j in j_all]
    return E_sim, cfg["E_exp"]


# ------------------------------------------------------------------------------
# Per-test j0_an profiling — the "inner" problem.
# Given the currently-set k_ca, find the j0_an that minimises this test's
# own MSE. Independent across tests (see module docstring).
# ------------------------------------------------------------------------------
def _fit_j0_an_for_test(cfg: dict, xatol: float, maxiter: int) -> tuple[float, float]:
    """Return (best log10 j0_an, MSE at that j0_an) for one test at the current k_ca."""
    def _obj(lp_j0: float) -> float:
        anodeparams.j_0 = 10.0 ** float(lp_j0)
        try:
            E_sim, E_exp = _run_test(cfg)
        except Exception as e:
            print(f"  -> test {cfg['test_no']} raised {e}, penalty", flush=True)
            return _MSE_PENALTY
        return float(np.average((np.array(E_sim) - E_exp) ** 2, weights=cfg["weights"]))

    res = minimize_scalar(
        _obj, bounds=_LOG_BOUNDS_J0_AN, method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )
    return float(res.x), float(res.fun)


def _objective_outer(log_k_ca: float, xatol: float, maxiter: int) -> tuple[float, dict]:
    """Given log_k_ca, independently profile out each test's j0_an.

    Returns (combined_mse, per_test) where combined_mse is defined exactly as
    in the v1 joint objective (mean over ALL points pooled across tests) —
    mathematically identical to the joint minimum over (j0_1, j0_2, j0_3)
    because the objective is additively separable in them (see module
    docstring): minimising each test's own MSE independently reproduces the
    same j0_an values a joint search would converge to.
    """
    cathodeparams.reaction_rate_constant = 10.0 ** log_k_ca
    total_sse, total_n = 0.0, 0
    per_test: dict[int, dict] = {}
    for cfg in _test_configs:
        lp_opt, mse_opt = _fit_j0_an_for_test(cfg, xatol, maxiter)
        tn = int(cfg["test_no"])
        n  = _n_points_per_test[tn]
        total_sse += mse_opt * n
        total_n   += n
        per_test[tn] = {"log_j0_an": lp_opt, "mse": mse_opt}
    return total_sse / total_n, per_test


def _grid_worker(log_k_ca: float) -> tuple[float, float, dict, float]:
    """Top-level (picklable) wrapper for multiprocessing.Pool.map."""
    t0 = time.time()
    mse, per_test = _objective_outer(log_k_ca, args.inner_xatol, args.inner_maxiter)
    return log_k_ca, mse, per_test, time.time() - t0


def _run_grid_round(candidates: list, n_workers: int) -> list:
    """Evaluate a list of log10(k_ca) candidates, in parallel if n_workers > 1."""
    if n_workers <= 1:
        return [_grid_worker(lk) for lk in candidates]
    with mp.Pool(processes=n_workers) as pool:
        return pool.map(_grid_worker, candidates)


def _fmt_j0(per_test: dict) -> str:
    return ", ".join(f"t{tn}:{10.0**v['log_j0_an']:.2e}" for tn, v in sorted(per_test.items()))


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    # -- Stage 1 --------------------------------------------------------------
    if args.stage == 1:
        print("Stage 1 (v2) — coarse-to-fine grid over k_ca, profiled j0_an per test",
              flush=True)
        if args.context:
            print(f"  context: {args.context}", flush=True)
        for cfg in _test_configs:
            print(f"  Test {int(cfg['test_no'])}: SOC={cfg['soc']}  T={cfg['T']}°C  "
                  f"Q={cfg['Q_v']} ml/min", flush=True)
        print(f"  workers={n_workers}  grid_points={args.grid_points}  "
              f"refine_points={args.refine_points}  "
              f"inner_xatol={args.inner_xatol}  polish_xatol={args.polish_xatol}",
              flush=True)

        lo, hi = _LOG_BOUNDS_K_CA
        t_start = time.time()

        # -- Round 1 — coarse global scan (fully parallel) ------------------
        grid1 = list(np.linspace(lo, hi, args.grid_points))
        print(f"\n{'-'*60}\nRound 1 — coarse scan over "
              f"k_ca in [{10**lo:.2e}, {10**hi:.2e}] m/s ({len(grid1)} points)\n{'-'*60}",
              flush=True)
        results1 = _run_grid_round(grid1, n_workers)
        for lk, mse, per_test, el in sorted(results1, key=lambda r: r[0]):
            print(f"  k_ca={10.0**lk:.4e}  MSE={mse:.6f}  RMSE={mse**0.5:.4f} V  "
                  f"j0_an=[{_fmt_j0(per_test)}]  ({el:.1f}s)", flush=True)
        best1 = min(results1, key=lambda r: r[1])
        print(f"Round 1 best: k_ca={10.0**best1[0]:.4e}  MSE={best1[1]:.6f}  "
              f"RMSE={best1[1]**0.5:.4f} V", flush=True)

        # -- Round 2 — fine local scan around the round-1 best --------------
        step1 = (hi - lo) / max(len(grid1) - 1, 1)
        lo2 = max(lo, best1[0] - args.refine_span * step1)
        hi2 = min(hi, best1[0] + args.refine_span * step1)
        grid2 = list(np.linspace(lo2, hi2, args.refine_points))
        print(f"\n{'-'*60}\nRound 2 — refine scan over "
              f"k_ca in [{10**lo2:.2e}, {10**hi2:.2e}] m/s ({len(grid2)} points)\n{'-'*60}",
              flush=True)
        results2 = _run_grid_round(grid2, n_workers)
        for lk, mse, per_test, el in sorted(results2, key=lambda r: r[0]):
            print(f"  k_ca={10.0**lk:.4e}  MSE={mse:.6f}  RMSE={mse**0.5:.4f} V  "
                  f"j0_an=[{_fmt_j0(per_test)}]  ({el:.1f}s)", flush=True)

        all_results  = results1 + results2
        best         = min(all_results, key=lambda r: r[1])
        k_ca_fit     = 10.0 ** best[0]
        elapsed_scan = time.time() - t_start
        print(f"\nBest of both rounds: k_ca={k_ca_fit:.4e}  MSE={best[1]:.6f}  "
              f"RMSE={best[1]**0.5:.4f} V  ({elapsed_scan/60:.1f} min so far)", flush=True)

        # -- Final polish — tight per-test j0_an refine at the winning k_ca --
        print(f"\n{'-'*60}\nFinal polish at k_ca={k_ca_fit:.4e}\n{'-'*60}", flush=True)
        cathodeparams.reaction_rate_constant = k_ca_fit
        j0_an_fit, mse_per_test = {}, {}
        for cfg in _test_configs:
            tn = int(cfg["test_no"])
            lp_opt, mse_opt = _fit_j0_an_for_test(cfg, args.polish_xatol, args.polish_maxiter)
            j0_an_fit[tn]    = 10.0 ** lp_opt
            mse_per_test[tn] = mse_opt
            print(f"  test {tn}: j0_an={j0_an_fit[tn]:.4e}  "
                  f"MSE={mse_opt:.6f}  RMSE={mse_opt**0.5:.4f} V", flush=True)

        n_total = sum(_n_points_per_test[tn] for tn in j0_an_fit)
        combined_mse = sum(mse_per_test[tn] * _n_points_per_test[tn]
                            for tn in j0_an_fit) / n_total
        elapsed = time.time() - t_start

        print(f"\n{'='*60}")
        print(f"Stage 1 (v2) complete  |  elapsed {elapsed/60:.1f} min")
        print(f"Combined MSE={combined_mse:.6f}  RMSE={combined_mse**0.5:.4f} V")
        print(f"  k_ca = {k_ca_fit:.4e}  m/s")
        for tn, j0 in j0_an_fit.items():
            print(f"  j0_an t{tn} = {j0:.4e}  A/m²")

        _RES_GLB.mkdir(parents=True, exist_ok=True)
        out_path = _RES_GLB / f"global_k_ca_fit_{stamp}.json"
        with open(out_path, "w") as f:
            json.dump({
                "stage":     1,
                "algorithm": "grid_refine_v2",
                "context":   args.context,
                "tests":     [int(cfg["test_no"]) for cfg in _test_configs],
                "elapsed_s": elapsed,
                "mse":       combined_mse,
                "rmse":      combined_mse ** 0.5,
                "params":    {"k_ca": k_ca_fit, "j0_an": j0_an_fit},
                "mse_per_test": mse_per_test,
                "log_bounds": {"k_ca": list(_LOG_BOUNDS_K_CA), "j0_an": list(_LOG_BOUNDS_J0_AN)},
                "scan": [
                    {"log_k_ca": lk, "k_ca": 10.0**lk, "mse": mse}
                    for lk, mse, _, _ in sorted(all_results, key=lambda r: r[0])
                ],
                "settings": {
                    "workers": n_workers, "grid_points": args.grid_points,
                    "refine_points": args.refine_points, "refine_span": args.refine_span,
                    "inner_xatol": args.inner_xatol, "inner_maxiter": args.inner_maxiter,
                    "polish_xatol": args.polish_xatol, "polish_maxiter": args.polish_maxiter,
                    "stride": STRIDE, "tf": TF, "rtol": RTOL,
                    "point_weights": {
                        str(int(cfg["test_no"])): {
                            f"{j:+.0f}": float(w) for j, w in
                            zip(cfg["j_all_signed"], cfg["weights"]) if w != 1
                        }
                        for cfg in _test_configs
                    },
                },
            }, f, indent=2)
        print(f"Result saved -> {out_path}")

        # -- Diagnostic: MSE(k_ca) profile ------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        r1_sorted = sorted(results1, key=lambda r: r[0])
        r2_sorted = sorted(results2, key=lambda r: r[0])
        ax.plot([10**lk for lk, *_ in r1_sorted], [m for _, m, *_ in r1_sorted],
                'o-', color='tab:blue', label='Round 1 (coarse)')
        ax.plot([10**lk for lk, *_ in r2_sorted], [m for _, m, *_ in r2_sorted],
                'o-', color='tab:orange', label='Round 2 (refine)')
        ax.axvline(k_ca_fit, color='k', ls='--', lw=1, label=f'best k_ca={k_ca_fit:.3e}')
        ax.set_xscale('log')
        ax.set_xlabel('k_ca  [m/s]')
        ax.set_ylabel('MSE  [V²]  (j0_an profiled out per test)')
        ax.set_title('Stage 1 (v2)  —  profiled MSE(k_ca)')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        fig.tight_layout()
        profile_path = _RES_GLB / f"global_k_ca_{stamp}_mse_profile.png"
        fig.savefig(profile_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"MSE(k_ca) profile plot saved -> {profile_path}")

        # -- Diagnostic: E_cell vs j for all tests ----------------------------
        print("\nRunning final diagnostic pass for plot...", flush=True)
        n_tests = len(_test_configs)
        fig, axes = plt.subplots(1, n_tests, figsize=(5*n_tests, 5), sharey=False)
        if n_tests == 1:
            axes = [axes]
        for ax, cfg in zip(axes, _test_configs):
            tn = int(cfg["test_no"])
            anodeparams.j_0 = j0_an_fit[tn]
            j_c  = cfg["j_charge"]
            j_d  = cfg["j_discharge"]
            E_exp_c = cfg["E_exp"][:len(j_c)]
            E_exp_d = cfg["E_exp"][len(j_c):]
            j_all  = list(j_c) + [-j for j in j_d]
            E_plot = [_run_E_cell(j, cfg) for j in j_all]
            nc = len(j_c)
            ax.plot(j_c, E_plot[:nc],   'b-o',  lw=2, ms=5, label='Charge (sim)')
            ax.plot(j_d, E_plot[nc:],   'r-o',  lw=2, ms=5, label='Discharge (sim)')
            ax.plot(j_c, E_exp_c,       'b^--', lw=1.5, ms=6, mfc='none', label='Charge (exp)')
            ax.plot(j_d, E_exp_d,       'r^--', lw=1.5, ms=6, mfc='none', label='Discharge (exp)')
            ax.set_xlabel('j  [A/m²]')
            ax.set_ylabel('E_cell  [V]')
            ax.set_title(f"Test {tn}  SOC={cfg['soc']}  T={cfg['T']}°C\n"
                         f"j0_an={j0_an_fit[tn]:.3e}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Stage 1 (v2) global fit  |  k_ca={k_ca_fit:.4e}  "
                     f"MSE={combined_mse:.6f}  RMSE={combined_mse**0.5:.4f} V")
        fig.tight_layout()
        plot_path = _RES_GLB / f"global_k_ca_{stamp}_e_vs_i.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved -> {plot_path}")

    # -- Stage 2 --------------------------------------------------------------
    else:
        cfg = _test_configs[0]
        tn  = int(cfg["test_no"])

        # Resolve k_ca
        k_ca = args.k_ca
        if k_ca is None:
            latest_s1 = sorted(_RES_GLB.glob("global_k_ca_fit_*.json"))
            if not latest_s1:
                raise SystemExit(
                    "No Stage 1 result found in results/test fitting/global/. "
                    "Run Stage 1 first or pass --k-ca VALUE.")
            with open(latest_s1[-1]) as f:
                s1 = json.load(f)
            k_ca = s1["params"]["k_ca"]
            print(f"Loaded k_ca={k_ca:.4e} from {latest_s1[-1].name}", flush=True)

        cathodeparams.reaction_rate_constant = k_ca
        print(f"Stage 2 (v2) — fitting j0_an for test {tn}  (k_ca={k_ca:.4e} fixed)",
              flush=True)
        if args.context:
            print(f"  context: {args.context}", flush=True)
        print(f"  SOC={cfg['soc']}  T={cfg['T']}°C  Q={cfg['Q_v']} ml/min", flush=True)

        t_start = time.time()
        lp_opt, mse_opt = _fit_j0_an_for_test(cfg, args.polish_xatol, args.polish_maxiter)
        j0_an_fit = 10.0 ** lp_opt
        elapsed = time.time() - t_start

        print(f"\n{'='*60}")
        print(f"Stage 2 (v2)  Test {tn}  |  elapsed {elapsed:.1f} s")
        print(f"MSE={mse_opt:.6f}  RMSE={mse_opt**0.5:.4f} V")
        print(f"  k_ca  = {k_ca:.4e}  (fixed)")
        print(f"  j0_an = {j0_an_fit:.4e}")

        test_dir = _RES / f"test {tn}"
        test_dir.mkdir(parents=True, exist_ok=True)
        out_path = test_dir / f"global_j0an_test{tn}_{stamp}.json"
        with open(out_path, "w") as f:
            json.dump({
                "stage":     2,
                "algorithm": "bounded_profile_v2",
                "context":   args.context,
                "test":      tn,
                "elapsed_s": elapsed,
                "mse":       mse_opt,
                "rmse":      mse_opt ** 0.5,
                "params":    {"k_ca": k_ca, "j0_an": j0_an_fit},
                "log_bounds": [list(_LOG_BOUNDS_J0_AN)],
                "settings":  {
                    "polish_xatol": args.polish_xatol, "polish_maxiter": args.polish_maxiter,
                    "stride": STRIDE, "tf": TF, "rtol": RTOL,
                    "k_ca_fixed": k_ca,
                    "point_weights": {
                        f"{j:+.0f}": float(w) for j, w in
                        zip(cfg["j_all_signed"], cfg["weights"]) if w != 1
                    },
                },
            }, f, indent=2)
        print(f"Result saved -> {out_path}")

        # -- Diagnostic plot ---------------------------------------------------
        print("\nRunning final diagnostic pass for plot...", flush=True)
        anodeparams.j_0 = j0_an_fit
        j_c, j_d   = cfg["j_charge"], cfg["j_discharge"]
        E_exp_c    = cfg["E_exp"][:len(j_c)]
        E_exp_d    = cfg["E_exp"][len(j_c):]
        j_all_plot = list(j_c) + [-j for j in j_d]
        E_plot     = [_run_E_cell(j, cfg) for j in j_all_plot]
        nc         = len(j_c)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(j_c, E_plot[:nc],  'b-o',  lw=2, ms=6, label='Charge (sim)')
        ax.plot(j_d, E_plot[nc:],  'r-o',  lw=2, ms=6, label='Discharge (sim)')
        ax.plot(j_c, E_exp_c,      'b^--', lw=1.5, ms=7, mfc='none', label='Charge (exp)')
        ax.plot(j_d, E_exp_d,      'r^--', lw=1.5, ms=7, mfc='none', label='Discharge (exp)')
        ax.set_xlabel('Applied current density  [A/m²]')
        ax.set_ylabel('E_cell  [V]')
        ax.set_title(
            f'Test {tn}  |  SOC={cfg["soc"]}  T={cfg["T"]}°C  Q={cfg["Q_v"]} ml/min\n'
            f'k_ca={k_ca:.4e} (fixed)   j0_an={j0_an_fit:.4e}\n'
            f'MSE={mse_opt:.6f}  RMSE={mse_opt**0.5:.4f} V'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = test_dir / f"global_j0an_test{tn}_{stamp}_e_vs_i.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved -> {plot_path}")
