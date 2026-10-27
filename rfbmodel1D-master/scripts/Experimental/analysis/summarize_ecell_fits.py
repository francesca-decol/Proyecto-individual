"""
Summarize the E-cell fit quality across all tests re-simulated by
run_ecell_all_tests.py.

Reads each results/test fitting/test N/ecell_resim_testN.json (the
per-test simulation cache) and, via its provenance fields, the k_ca / j0_an
that were actually used — either the shared per-temperature global fit
(results/test fitting/global/global_k_ca_{24,32,40}_low_SSA.json) or, for
the two SOC=50%/Q_v=45/T=32°C replicate tests (14, 15) that aren't part of
that joint fit, their own per-test Stage-2 fit
(results/test fitting/test N/global_j0an_testN_*.json) — and produces:
  - a printed table
  - results/test fitting/ecell_fits_summary.csv

Metrics reported
----------------
  RMSE  [V]  — sqrt(mean((E_sim - E_exp)^2)) over every charge + discharge
               point for that test.
  APE   [%]  — mean absolute percentage error, |E_sim - E_exp| / E_exp * 100,
               averaged over the same points.

Usage
-----
  python scripts/Experimental/analysis/summarize_ecell_fits.py
  python scripts/Experimental/analysis/summarize_ecell_fits.py --csv path/to/output.csv
"""
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT    = Path(__file__).parent.parent.parent.parent
_RES     = _ROOT / "results" / "test fitting"
_RES_GLB = _RES / "global"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=str, default=None,
                   help="Output CSV path (default: results/test fitting/ecell_fits_summary.csv)")
    return p.parse_args()


def _resim_json(test_no: int) -> Path | None:
    path = _RES / f"test {test_no}" / f"ecell_resim_test{test_no}.json"
    return path if path.exists() else None


def _resolve_k_ca_j0an(test_no: int, source_global_fit: str | None) -> tuple[float, float]:
    """k_ca, j0_an actually used for this test, resolved from its provenance
    filename — either the shared per-temperature global fit or, for tests
    not covered by it, that test's own Stage-2 fit."""
    if source_global_fit is None:
        return float("nan"), float("nan")

    if source_global_fit.startswith("global_j0an_test"):
        # Per-test Stage-2 fit, lives in that test's own folder.
        path = _RES / f"test {test_no}" / source_global_fit
        with open(path) as f:
            fit = json.load(f)
        return float(fit["params"]["k_ca"]), float(fit["params"]["j0_an"])

    # Shared multi-test global fit (global_k_ca_{T}_low_SSA.json).
    path = _RES_GLB / source_global_fit
    with open(path) as f:
        fit = json.load(f)
    k_ca  = float(fit["params"]["k_ca"])
    j0_an = float(fit["params"]["j0_an"][str(test_no)])
    return k_ca, j0_an


def _rmse_ape(r: dict) -> tuple[float, float, int]:
    """RMSE [V], APE [%], n_points over every charge+discharge point for one
    test's resim cache. (nan, nan, 0) if no simulated points are available
    (e.g. no j0_an fit at all for that test)."""
    sim = np.array(
        [v for v in r["E_sim_charge"] if v is not None]
        + [v for v in r["E_sim_discharge"] if v is not None], dtype=float)
    exp_c = [e for e, s in zip(r["E_exp_charge"], r["E_sim_charge"]) if s is not None]
    exp_d = [e for e, s in zip(r["E_exp_discharge"], r["E_sim_discharge"]) if s is not None]
    exp = np.array(exp_c + exp_d, dtype=float)

    ok = np.isfinite(sim) & np.isfinite(exp) & (exp != 0)
    if not ok.any():
        return float("nan"), float("nan"), 0

    rmse = float(np.sqrt(np.mean((sim[ok] - exp[ok]) ** 2)))
    ape  = float(np.mean(np.abs(sim[ok] - exp[ok]) / exp[ok]) * 100.0)
    return rmse, ape, int(ok.sum())


def main():
    args = parse_args()

    rows = []
    test_folders = sorted(_RES.glob("test *"), key=lambda p: int(p.name.split()[-1]))

    for folder in test_folders:
        test_no = int(folder.name.split()[-1])
        path = _resim_json(test_no)
        if path is None:
            continue

        with open(path) as f:
            r = json.load(f)

        k_ca, j0_an = _resolve_k_ca_j0an(test_no, r.get("source_global_fit"))
        rmse, ape, n_pts = _rmse_ape(r)

        rows.append({
            "test":              test_no,
            "SOC":               r["soc"],
            "T (°C)":            r["T"],
            "Q_v (ml/min)":      r["Q_v"],
            "k_ca":              k_ca,
            "j0_an":             j0_an,
            "RMSE (V)":          rmse,
            "APE (%)":           ape,
            "n_pts":             n_pts,
            "hfr_fit":           r.get("source_hfr_fit"),
            "global_fit":        r.get("source_global_fit"),
        })

    if not rows:
        print(f"No ecell_resim_testN.json files found under {_RES}. "
              f"Run scripts/Experimental/analysis/run_ecell_all_tests.py first.")
        return

    df = pd.DataFrame(rows).sort_values("test").reset_index(drop=True)

    # -- Print table ------------------------------------------------------------
    header = (f"{'test':>5}  {'SOC':>5}  {'T(°C)':>6}  {'Q_v':>5}  "
              f"{'k_ca':>10}  {'j0_an':>10}  "
              f"{'RMSE(V)':>8}  {'APE%':>7}  {'n_pts':>6}  {'global_fit':<32}")
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        rmse_str = f"{row['RMSE (V)']:>8.4f}" if pd.notna(row["RMSE (V)"]) else f"{'n/a':>8}"
        ape_str  = f"{row['APE (%)']:>7.2f}"  if pd.notna(row["APE (%)"])  else f"{'n/a':>7}"
        k_ca_str  = f"{row['k_ca']:>10.3e}"  if pd.notna(row["k_ca"])  else f"{'n/a':>10}"
        j0an_str  = f"{row['j0_an']:>10.3e}" if pd.notna(row["j0_an"]) else f"{'n/a':>10}"
        print(f"{int(row['test']):>5}  {row['SOC']:>5.2f}  {row['T (°C)']:>6.0f}  "
              f"{row['Q_v (ml/min)']:>5.0f}  {k_ca_str}  {j0an_str}  "
              f"{rmse_str}  {ape_str}  {int(row['n_pts']):>6}  "
              f"{str(row['global_fit']):<32}")

    print(f"\n{len(df)} test(s) summarised.")
    no_fit = df.loc[df["n_pts"] == 0, "test"].tolist()
    if no_fit:
        print(f"[note] RMSE/APE are n/a for tests with no simulated points: {no_fit}")

    # -- Save CSV -----------------------------------------------------------------
    csv_path = Path(args.csv) if args.csv else _RES / "ecell_fits_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path}")


if __name__ == "__main__":
    main()
