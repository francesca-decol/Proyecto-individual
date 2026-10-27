"""
Re-simulate HFR(j) for every fitted test using its saved HFR fit parameters
and write the results to a single tidy CSV for later plotting.

This does NOT re-fit anything — it only re-runs the forward DAE simulation
at each test's already-fitted parameters (from fit_HFR_v2.py's
HFR_fit_test{N}_*.json) to reconstruct the HFR(j) curve.

Output
------
  results/test fitting/hfr_by_condition/hfr_simulated_all.csv
      Long format, same schema as data/experimental/HFR.csv plus a
      HFR_sim (ohm cm2) column:
      test_no, SOC, T(°C), Q_v (ml/min), mode, j_appl (A/m2),
      HFR_sim (ohm cm2), HFR_exp (ohm cm2)

  results/test fitting/test N/hfr_resim_testN.json
      Per-test cache, re-used automatically on the next run unless a newer
      fit exists for that test, or --force is passed.

Once the CSV exists, use plot_hfr_by_condition.py to make figures from it —
that script has no simulation dependencies (pandas + matplotlib only), so
it's safe to edit and re-run instantly without touching the DAE solver.

Usage
-----
  python scripts/Experimental/analysis/run_hfr_all_tests.py --workers 8
  python scripts/Experimental/analysis/run_hfr_all_tests.py --tests 1 2 3 --force
"""
import sys
import os
import json
import argparse
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from solve_dae.integrate import solve_dae

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from simulate_HFR import build_cell, build_atol, soc_cutoff_event   # noqa: E402
from rfbmodel1D.parameters.params import cellparams, membraneparams  # noqa: E402


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests",   type=int, nargs="+", default=None,
                   help="Restrict to these test numbers (default: all in HFR.csv)")
    p.add_argument("--workers", type=int, default=-1,
                   help="Parallel workers across tests (-1 = all cores)")
    p.add_argument("--force",   action="store_true",
                   help="Ignore cache, re-simulate every point")
    return p.parse_args()


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
TF           = 1800
RTOL         = 1e-5
_CHECK_EVERY = 20_000
_MIN_DT      = 0.5

_ROOT    = Path(__file__).parent.parent.parent.parent
_HFR_CSV = _ROOT / "data" / "experimental" / "HFR.csv"
_RES     = _ROOT / "results" / "test fitting"
_OUT_DIR = _RES / "hfr_by_condition"
_OUT_CSV = _OUT_DIR / "hfr_simulated_all.csv"


def _discover_fit_json(test_no: int) -> Path | None:
    """Latest HFR fit JSON for a test number, or None if it hasn't been fitted."""
    folder = _RES / f"test {test_no}"
    candidates = sorted(folder.glob(f"HFR_fit_test{test_no}_*.json"))
    return candidates[-1] if candidates else None


def _cache_path(test_no: int) -> Path:
    return _RES / f"test {test_no}" / f"hfr_resim_test{test_no}.json"


# ------------------------------------------------------------------------------
# Forward HFR simulation at fixed (already-fitted) parameters
# ------------------------------------------------------------------------------
def _compute_HFR(j_appl: float, soc: float, T: float, Q_v: float,
                  r_extra_ohmcm2: float) -> float | None:
    """Steady-state HFR for one operating point. Returns None on solver failure."""
    cell, y0, yp0 = build_cell(soc, j_appl, T, Q_v)
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
        print(f"    -> solve failed j={j_appl:+.0f}: {e}", flush=True)
        return None

    if sol.t[-1] < 0.999 * TF:
        print(f"    -> early stop j={j_appl:+.0f}  t={sol.t[-1]:.1f}s", flush=True)
        return None

    cell.dae_residual_full_cell(sol.t[-1], sol.y[:, -1], sol.yp[:, -1])

    kappa_io_ca = cell.cathode.kappa_io
    sigma_ca    = np.ones_like(kappa_io_ca) * cell.cathode.eff_conductivity
    dx_ca       = cell.mesh.cathode_mesh.dx * cell.cathode.thickness
    R_cathode   = sum(dx_ca / (sigma_ca + kappa_io_ca))

    kappa_io_m  = cell.membrane.kappa_io
    dx_m        = cell.mesh.membrane_mesh.dx * cell.membrane.thickness
    R_membrane  = sum(dx_m / kappa_io_m)

    R_ACL  = cell.anode.thickness  / cell.anode.eff_conductivity
    R_AGDL = cell.agdl.thickness   / cell.agdl.eff_conductivity
    R_BP   = cell.thickness_BP / cell.sigma_BP
    R_CC   = cell.thickness_CC / cell.sigma_CC

    r_extra_ohm_m2 = r_extra_ohmcm2 / 1e4
    return float((R_cathode + R_membrane + R_ACL + R_AGDL + R_BP + R_CC + r_extra_ohm_m2) * 1e4)


def _resim_test(test_no: int, fit_path: Path) -> dict:
    """Re-simulate HFR(j) for one test using its fitted parameters."""
    with open(fit_path) as f:
        fit = json.load(f)
    p, s, c = fit["params"], fit["settings"], fit["conditions"]

    membraneparams.k_sorp_vw   = p["k_vw_sorp"]
    membraneparams.k_desorp_vw = p["S_desorp"] * p["k_vw_sorp"]
    membraneparams.k_sorp_lw   = p["k_lw_sorp"]
    membraneparams.k_desorp_lw = p["S_desorp"] * p["k_lw_sorp"]
    membraneparams.k_ca_lw     = p["k_ca"]
    membraneparams.sf_D        = s["sf_D"]
    cellparams.vanadium_uptake = s["vanadium_uptake"]
    r_extra = p["R_extra (Ω·cm²)"]

    soc, T, Q_v = c["soc"], c["T_C"], c["Q_v_ml_min"]

    data = pd.read_csv(_HFR_CSV)
    data = data[data["test_no"] == test_no]
    j_charge    = data[data["mode"] == "charge"]["j_appl (A/m2)"].values.astype(float)
    j_discharge = data[data["mode"] == "discharge"]["j_appl (A/m2)"].values.astype(float)
    HFR_exp_c   = data[data["mode"] == "charge"]["value (ohm cm2)"].values.astype(float)
    HFR_exp_d   = data[data["mode"] == "discharge"]["value (ohm cm2)"].values.astype(float)

    print(f"Test {test_no}: SOC={soc}  T={T}°C  Q={Q_v} ml/min  "
          f"({len(j_charge)} charge + {len(j_discharge)} discharge pts)", flush=True)
    t0 = time.time()
    HFR_sim_c = [_compute_HFR(j, soc, T, Q_v, r_extra) for j in j_charge]
    HFR_sim_d = [_compute_HFR(-j, soc, T, Q_v, r_extra) for j in j_discharge]
    print(f"  test {test_no} done in {time.time()-t0:.0f}s", flush=True)

    return dict(
        test_no=test_no, soc=soc, T=T, Q_v=Q_v,
        source_fit_json=fit_path.name,
        j_charge=j_charge.tolist(),    HFR_sim_charge=HFR_sim_c,    HFR_exp_charge=HFR_exp_c.tolist(),
        j_discharge=j_discharge.tolist(), HFR_sim_discharge=HFR_sim_d, HFR_exp_discharge=HFR_exp_d.tolist(),
    )


def _resim_or_cached(test_no: int, force: bool) -> dict | None:
    fit_path = _discover_fit_json(test_no)
    if fit_path is None:
        print(f"[skip] test {test_no}: no fit JSON found in results/test fitting/test {test_no}/",
              flush=True)
        return None

    cache = _cache_path(test_no)
    if cache.exists() and not force:
        with open(cache) as f:
            cached = json.load(f)
        if cached.get("source_fit_json") == fit_path.name:
            print(f"[cache] test {test_no}  (from {fit_path.name})", flush=True)
            return cached
        print(f"[stale cache] test {test_no}: newer fit found ({fit_path.name}), re-simulating",
              flush=True)

    result = _resim_test(test_no, fit_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(result, f, indent=2)
    return result


def _resim_or_cached_star(args_tuple):
    return _resim_or_cached(*args_tuple)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    args = parse_args()
    n_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    hfr_data  = pd.read_csv(_HFR_CSV)
    all_tests = sorted(hfr_data["test_no"].unique().astype(int).tolist())
    tests     = args.tests if args.tests else all_tests

    print(f"Re-simulating HFR for {len(tests)} test(s)  |  workers={n_workers}  "
          f"cache={'off (--force)' if args.force else 'on'}", flush=True)

    t_start = time.time()
    work = [(t, args.force) for t in tests]
    if n_workers <= 1:
        results = [_resim_or_cached_star(w) for w in work]
    else:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_resim_or_cached_star, work)

    skipped = [t for t, r in zip(tests, results) if r is None]
    results = [r for r in results if r is not None]
    print(f"\nDone in {(time.time()-t_start)/60:.1f} min  |  "
          f"{len(results)}/{len(tests)} tests simulated"
          + (f"  |  skipped (no fit): {skipped}" if skipped else ""), flush=True)

    if not results:
        print("Nothing to save.")
        return

    # -- Flatten to long format (same schema as data/experimental/HFR.csv) ------
    rows = []
    for r in results:
        for j, sim, exp in zip(r["j_charge"], r["HFR_sim_charge"], r["HFR_exp_charge"]):
            rows.append({
                "test_no": r["test_no"], "SOC": r["soc"], "T(°C)": r["T"],
                "Q_v (ml/min)": r["Q_v"], "mode": "charge",
                "j_appl (A/m2)": j,
                "HFR_sim (ohm cm2)": sim, "HFR_exp (ohm cm2)": exp,
            })
        for j, sim, exp in zip(r["j_discharge"], r["HFR_sim_discharge"], r["HFR_exp_discharge"]):
            rows.append({
                "test_no": r["test_no"], "SOC": r["soc"], "T(°C)": r["T"],
                "Q_v (ml/min)": r["Q_v"], "mode": "discharge",
                "j_appl (A/m2)": j,
                "HFR_sim (ohm cm2)": sim, "HFR_exp (ohm cm2)": exp,
            })

    df = pd.DataFrame(rows).sort_values(["test_no", "mode", "j_appl (A/m2)"])
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows ({len(results)} tests) -> {_OUT_CSV}")
    print("Now run: python scripts/Experimental/analysis/plot_hfr_by_condition.py")


if __name__ == "__main__":
    main()
