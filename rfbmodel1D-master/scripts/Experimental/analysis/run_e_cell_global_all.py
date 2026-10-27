"""
Re-simulate E_cell(j) for every test covered by a global k_ca fit
(fit_e_cell_global_v2.py Stage 1 output, results/test fitting/global/
global_k_ca_*.json) and write the results to a single tidy CSV for later
plotting.

This does NOT re-fit anything — it only re-runs the forward DAE simulation
at each test's already-fitted parameters (shared k_ca + that test's own
j0_an from the global JSON, plus that test's own HFR-fit membrane
parameters from HFR_fit_test{N}_*.json) to reconstruct the E_cell(j) curve.

A global fit file covers one temperature: several tests (e.g. 9, spanning
the SOC / Q_v grid) fit jointly with one shared cathode rate constant k_ca
and a separate anode exchange current j0_an per test. Every
global_k_ca_*.json with "algorithm": "grid_refine_v2" found in
results/test fitting/global/ is processed; each is expected to cover a
single temperature (asserted from data/experimental/E_cell.csv).

Output
------
  results/test fitting/e_cell_global_by_condition/e_cell_global_simulated_all.csv
      Long format, same schema as data/experimental/E_cell.csv plus
      E_sim (V), k_ca, j0_an, source_global_json columns:
      test_no, SOC, T(°C), Q_v (ml/min), mode, j_appl (A/m2),
      E_sim (V), E_exp (V), k_ca, j0_an, source_global_json

  results/test fitting/test N/e_cell_global_resim_testN.json
      Per-test cache, re-used automatically on the next run unless a newer
      global fit exists for that test, or --force is passed.

Once the CSV exists, use plot_e_cell_global_by_condition.py to make figures
from it — that script has no simulation dependencies (pandas + matplotlib
only), so it's safe to edit and re-run instantly without touching the DAE
solver.

Usage
-----
  python scripts/Experimental/analysis/run_e_cell_global_all.py --workers 8
  python scripts/Experimental/analysis/run_e_cell_global_all.py --tests 1 4 7 --force
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
from simulate_HFR import build_cell, build_atol, soc_cutoff_event                    # noqa: E402
from rfbmodel1D.parameters.params import (                                           # noqa: E402
    cathodeparams, anodeparams, membraneparams, cellparams)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests",   type=int, nargs="+", default=None,
                   help="Restrict to these test numbers (default: every test covered "
                        "by a global fit)")
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

_ROOT     = Path(__file__).parent.parent.parent.parent
_ECELL_CSV = _ROOT / "data" / "experimental" / "E_cell.csv"
_HFR_CSV   = _ROOT / "data" / "experimental" / "HFR.csv"
_RES       = _ROOT / "results" / "test fitting"
_GLOBAL_DIR = _RES / "global"
_OUT_DIR   = _RES / "e_cell_global_by_condition"
_OUT_CSV   = _OUT_DIR / "e_cell_global_simulated_all.csv"


def _discover_global_fits() -> list[Path]:
    """All Stage-1 global k_ca fits (v2, multi-test) in results/test fitting/global/."""
    out = []
    for path in sorted(_GLOBAL_DIR.glob("global_k_ca_fit*.json")):
        with open(path) as f:
            fit = json.load(f)
        if fit.get("algorithm") == "grid_refine_v2" and len(fit.get("tests", [])) > 1:
            out.append(path)
    return out


def _discover_hfr_fit(test_no: int) -> Path | None:
    """Latest HFR fit JSON for a test number, or None if it hasn't been fitted."""
    candidates = sorted((_RES / f"test {test_no}").glob(f"HFR_fit_test{test_no}_*.json"))
    return candidates[-1] if candidates else None


def _cache_path(test_no: int) -> Path:
    return _RES / f"test {test_no}" / f"e_cell_global_resim_test{test_no}.json"


# ------------------------------------------------------------------------------
# Forward E_cell simulation at fixed (already-fitted) parameters
# ------------------------------------------------------------------------------
def _compute_E_cell(j_appl: float, soc: float, T: float, Q_v: float) -> float | None:
    """Steady-state E_cell for one operating point. Returns None on solver failure."""
    try:
        cell, y0, yp0 = build_cell(soc, j_appl, T, Q_v)
    except Exception as e:
        print(f"    -> init failed j={j_appl:+.0f}: {e}", flush=True)
        return None
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

    y_f       = sol.y[:, -1]
    Nc        = cell.mesh.cathode_mesh.n_x
    n_vars_ca = cell.layout.n_vars_cathode
    off_ca    = cell.layout.n_vars_tank
    Phi_el    = y_f[off_ca:off_ca + Nc * n_vars_ca].reshape(Nc, n_vars_ca)[:, 4]
    dx        = cell.mesh.cathode_mesh.dx[0] * cell.cathode.thickness
    return float(Phi_el[0] + j_appl * (dx / 2) / cell.cathode.eff_conductivity)


def _resim_test(test_no: int, k_ca: float, j0_an: float,
                 hfr_fit_path: Path, global_fit_name: str) -> dict:
    """Re-simulate E_cell(j) for one test at its global-fit parameters."""
    with open(hfr_fit_path) as f:
        hfr_fit = json.load(f)
    p, s = hfr_fit["params"], hfr_fit["settings"]

    membraneparams.k_sorp_vw   = p["k_vw_sorp"]
    membraneparams.k_desorp_vw = p["S_desorp"] * p["k_vw_sorp"]
    membraneparams.k_sorp_lw   = p["k_lw_sorp"]
    membraneparams.k_desorp_lw = p["S_desorp"] * p["k_lw_sorp"]
    membraneparams.k_ca_lw     = p["k_ca"]
    membraneparams.sf_D        = s["sf_D"]
    cellparams.vanadium_uptake = s["vanadium_uptake"]
    cathodeparams.reaction_rate_constant = k_ca
    anodeparams.j_0                      = j0_an

    data = pd.read_csv(_ECELL_CSV)
    data = data[data["test_no"] == test_no]
    soc = float(data["SOC"].iloc[0])
    T   = float(data["T(°C)"].iloc[0])
    Q_v = float(data["Q_v (ml/min)"].iloc[0])

    j_charge    = data[data["mode"] == "charge"]["j_appl (A/m2)"].values.astype(float)
    j_discharge = data[data["mode"] == "discharge"]["j_appl (A/m2)"].values.astype(float)
    E_exp_c     = data[data["mode"] == "charge"]["value (V)"].values.astype(float)
    E_exp_d     = data[data["mode"] == "discharge"]["value (V)"].values.astype(float)

    print(f"Test {test_no}: SOC={soc}  T={T}°C  Q={Q_v} ml/min  "
          f"k_ca={k_ca:.4e}  j0_an={j0_an:.4e}  "
          f"({len(j_charge)} charge + {len(j_discharge)} discharge pts)", flush=True)
    t0 = time.time()
    E_sim_c = [_compute_E_cell(j, soc, T, Q_v) for j in j_charge]
    E_sim_d = [_compute_E_cell(-j, soc, T, Q_v) for j in j_discharge]
    print(f"  test {test_no} done in {time.time()-t0:.0f}s", flush=True)

    return dict(
        test_no=test_no, soc=soc, T=T, Q_v=Q_v,
        k_ca=k_ca, j0_an=j0_an,
        source_global_json=global_fit_name,
        j_charge=j_charge.tolist(),    E_sim_charge=E_sim_c,    E_exp_charge=E_exp_c.tolist(),
        j_discharge=j_discharge.tolist(), E_sim_discharge=E_sim_d, E_exp_discharge=E_exp_d.tolist(),
    )


def _resim_or_cached(test_no: int, k_ca: float, j0_an: float,
                      hfr_fit_path: Path, global_fit_path: Path, force: bool) -> dict | None:
    cache = _cache_path(test_no)
    if cache.exists() and not force:
        with open(cache) as f:
            cached = json.load(f)
        if (cached.get("source_global_json") == global_fit_path.name
                and cached.get("k_ca") == k_ca and cached.get("j0_an") == j0_an):
            print(f"[cache] test {test_no}  (from {global_fit_path.name})", flush=True)
            return cached
        print(f"[stale cache] test {test_no}: newer/different global fit "
              f"({global_fit_path.name}), re-simulating", flush=True)

    result = _resim_test(test_no, k_ca, j0_an, hfr_fit_path, global_fit_path.name)
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

    global_fits = _discover_global_fits()
    if not global_fits:
        raise SystemExit(
            f"No multi-test grid_refine_v2 global fits found in {_GLOBAL_DIR}. "
            f"Run fit_e_cell_global_v2.py --stage 1 first."
        )

    ecell_data = pd.read_csv(_ECELL_CSV)
    work = []
    skipped_no_hfr = []
    for global_fit_path in global_fits:
        with open(global_fit_path) as f:
            fit = json.load(f)
        tests = [int(t) for t in fit["tests"]]
        k_ca  = float(fit["params"]["k_ca"])
        j0_an = {int(t): float(v) for t, v in fit["params"]["j0_an"].items()}

        # sanity check: a global fit is meant to cover ONE temperature
        temps = ecell_data[ecell_data["test_no"].isin(tests)].groupby("test_no")["T(°C)"].first()
        if temps.nunique() > 1:
            raise SystemExit(
                f"{global_fit_path.name} covers tests at more than one temperature "
                f"({dict(temps)}) — expected one temperature per global fit."
            )
        print(f"{global_fit_path.name}: T={temps.iloc[0]:.0f}°C  tests={tests}  "
              f"k_ca={k_ca:.4e}", flush=True)

        for t in (args.tests if args.tests else tests):
            if t not in tests:
                continue
            hfr_fit_path = _discover_hfr_fit(t)
            if hfr_fit_path is None:
                print(f"[skip] test {t}: no HFR fit JSON found in "
                      f"results/test fitting/test {t}/", flush=True)
                skipped_no_hfr.append(t)
                continue
            work.append((t, k_ca, j0_an[t], hfr_fit_path, global_fit_path, args.force))

    print(f"\nRe-simulating E_cell for {len(work)} test(s)  |  workers={n_workers}  "
          f"cache={'off (--force)' if args.force else 'on'}", flush=True)

    t_start = time.time()
    if n_workers <= 1:
        results = [_resim_or_cached_star(w) for w in work]
    else:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_resim_or_cached_star, work)

    results = [r for r in results if r is not None]
    print(f"\nDone in {(time.time()-t_start)/60:.1f} min  |  "
          f"{len(results)}/{len(work)} tests simulated"
          + (f"  |  skipped (no HFR fit): {skipped_no_hfr}" if skipped_no_hfr else ""),
          flush=True)

    if not results:
        print("Nothing to save.")
        return

    # -- Flatten to long format (same schema as data/experimental/E_cell.csv) ---
    rows = []
    for r in results:
        for j, sim, exp in zip(r["j_charge"], r["E_sim_charge"], r["E_exp_charge"]):
            rows.append({
                "test_no": r["test_no"], "SOC": r["soc"], "T(°C)": r["T"],
                "Q_v (ml/min)": r["Q_v"], "mode": "charge",
                "j_appl (A/m2)": j,
                "E_sim (V)": sim, "E_exp (V)": exp,
                "k_ca": r["k_ca"], "j0_an": r["j0_an"],
                "source_global_json": r["source_global_json"],
            })
        for j, sim, exp in zip(r["j_discharge"], r["E_sim_discharge"], r["E_exp_discharge"]):
            rows.append({
                "test_no": r["test_no"], "SOC": r["soc"], "T(°C)": r["T"],
                "Q_v (ml/min)": r["Q_v"], "mode": "discharge",
                "j_appl (A/m2)": j,
                "E_sim (V)": sim, "E_exp (V)": exp,
                "k_ca": r["k_ca"], "j0_an": r["j0_an"],
                "source_global_json": r["source_global_json"],
            })

    df = pd.DataFrame(rows).sort_values(["test_no", "mode", "j_appl (A/m2)"])
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows ({len(results)} tests) -> {_OUT_CSV}")
    print("Now run: python scripts/Experimental/analysis/plot_e_cell_global_by_condition.py")


if __name__ == "__main__":
    main()
