"""
Summarize the latest HFR fit results across all tests.

Reads the most recent HFR_fit_testN_*.json from each
results/test fitting/test N/ folder and produces:
  - a printed table
  - results/test fitting/hfr_fits_summary.csv

Metrics reported
----------------
  MSE    [Ω²·cm⁴]  — directly from the HFR_fit_testN_*.json (DE objective at
                     the fitted parameters)
  RMSE   [Ω·cm²]    — sqrt(MSE)
  NRMSE  [%]        — RMSE / mean(HFR_exp) * 100 (normalised RMSE against the
                      mean experimental HFR for that test)
  APE    [%]        — mean absolute percentage error, |HFR_sim - HFR_exp| /
                      HFR_exp * 100 averaged over every individual charge +
                      discharge point in results/test fitting/test N/
                      hfr_resim_testN.json (produced by run_hfr_all_tests.py).
                      NaN if that file doesn't exist yet for a test.

Usage
-----
  python scripts/Experimental/analysis/summarize_hfr_fits.py
  python scripts/Experimental/analysis/summarize_hfr_fits.py --csv path/to/output.csv
"""
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT    = Path(__file__).parent.parent.parent.parent
_RES     = _ROOT / "results" / "test fitting"
_HFR_CSV = _ROOT / "data" / "experimental" / "HFR.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=str, default=None,
                   help="Output CSV path (default: results/test fitting/hfr_fits_summary.csv)")
    return p.parse_args()


def latest_fit_json(test_no: int) -> Path | None:
    folder = _RES / f"test {test_no}"
    candidates = sorted(folder.glob(f"HFR_fit_test{test_no}_*.json"))
    return candidates[-1] if candidates else None


def resim_json(test_no: int) -> Path | None:
    path = _RES / f"test {test_no}" / f"hfr_resim_test{test_no}.json"
    return path if path.exists() else None


def pointwise_ape(test_no: int) -> tuple[float, int]:
    """Mean absolute percentage error over every charge+discharge point in
    hfr_resim_testN.json. Returns (ape_pct, n_points); (nan, 0) if unavailable."""
    path = resim_json(test_no)
    if path is None:
        return float("nan"), 0

    with open(path) as f:
        r = json.load(f)

    sim = np.array(r["HFR_sim_charge"] + r["HFR_sim_discharge"], dtype=float)
    exp = np.array(r["HFR_exp_charge"] + r["HFR_exp_discharge"], dtype=float)
    ok  = np.isfinite(sim) & np.isfinite(exp) & (exp != 0)
    if not ok.any():
        return float("nan"), 0

    ape = float(np.mean(np.abs(sim[ok] - exp[ok]) / exp[ok]) * 100.0)
    return ape, int(ok.sum())


def main():
    args = parse_args()

    hfr_exp = pd.read_csv(_HFR_CSV)

    rows = []
    test_folders = sorted(_RES.glob("test *"), key=lambda p: int(p.name.split()[-1]))

    for folder in test_folders:
        test_no = int(folder.name.split()[-1])
        json_path = latest_fit_json(test_no)
        if json_path is None:
            continue

        with open(json_path) as f:
            fit = json.load(f)

        mse  = fit["mse"]
        rmse = mse ** 0.5

        # Mean experimental HFR for this test (both charge and discharge)
        exp_vals = hfr_exp[hfr_exp["test_no"] == test_no]["value (ohm cm2)"]
        if exp_vals.empty:
            nrmse = float("nan")
            hfr_exp_mean = float("nan")
        else:
            hfr_exp_mean = exp_vals.mean()
            nrmse = rmse / hfr_exp_mean * 100.0

        ape, n_pts = pointwise_ape(test_no)

        p = fit["params"]
        c = fit["conditions"]

        rows.append({
            "test":           test_no,
            "SOC":            c["soc"],
            "T (°C)":         c["T_C"],
            "Q_v (ml/min)":   c["Q_v_ml_min"],
            "k_vw_sorp":      p["k_vw_sorp"],
            "k_lw_sorp":      p["k_lw_sorp"],
            "S_desorp":       p["S_desorp"],
            "k_ca":           p["k_ca"],
            "R_extra (Ω·cm²)": p.get("R_extra (Ω·cm²)", float("nan")),
            "MSE (Ω²cm⁴)":   mse,
            "RMSE (Ω·cm²)":  rmse,
            "HFR_exp_mean":   hfr_exp_mean,
            "NRMSE (%)":      nrmse,
            "APE (%)":        ape,
            "APE_n_pts":      n_pts,
            "success":        fit["success"],
            "nit":            fit["nit"],
            "nfev":           fit["nfev"],
            "json":           json_path.name,
        })

    if not rows:
        print("No fit results found in", _RES)
        return

    df = pd.DataFrame(rows).sort_values("test").reset_index(drop=True)

    # -- Print table -----------------------------------------------------------
    header = (f"{'test':>5}  {'SOC':>5}  {'T(°C)':>6}  "
              f"{'k_vw_sorp':>10}  {'k_lw_sorp':>10}  {'S_desorp':>10}  "
              f"{'k_ca':>10}  {'R_extra':>8}  "
              f"{'RMSE':>8}  {'MSE':>10}  {'NRMSE%':>7}  {'APE%':>7}  {'n_pts':>6}")
    print(header)
    print("-" * len(header))

    for _, r in df.iterrows():
        ape_str = f"{r['APE (%)']:>7.2f}" if pd.notna(r["APE (%)"]) else f"{'n/a':>7}"
        print(f"{int(r['test']):>5}  {r['SOC']:>5.2f}  {r['T (°C)']:>6.0f}  "
              f"{r['k_vw_sorp']:>10.3e}  {r['k_lw_sorp']:>10.3e}  {r['S_desorp']:>10.3e}  "
              f"{r['k_ca']:>10.3e}  {r['R_extra (Ω·cm²)']:>8.4f}  "
              f"{r['RMSE (Ω·cm²)']:>8.4f}  {r['MSE (Ω²cm⁴)']:>10.6f}  "
              f"{r['NRMSE (%)']:>7.2f}  {ape_str}  {int(r['APE_n_pts']):>6}")

    print(f"\n{len(df)} test(s) summarised.")
    missing_resim = df.loc[df["APE_n_pts"] == 0, "test"].tolist()
    if missing_resim:
        print(f"[note] APE is n/a for tests without a hfr_resim_testN.json yet: {missing_resim}\n"
              f"       Run scripts/Experimental/analysis/run_hfr_all_tests.py to generate them.")

    # -- Save CSV --------------------------------------------------------------
    csv_path = Path(args.csv) if args.csv else _RES / "hfr_fits_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path}")


if __name__ == "__main__":
    main()
