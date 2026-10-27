"""
Orchestrate the full fitting pipeline for all experimental tests.

Phase 1 — HFR fit (fit_HFR_v2.py):
    Runs for every test present in data/experimental/HFR.csv (tests 1-29
    once all are added). Each test is fitted independently and in sequence.

Phase 2 — E_cell global fit (fit_e_cell_global_v2.py, Stage 1):
    After Phase 1, tests that appear in BOTH HFR.csv and E_cell.csv are
    grouped by temperature. One Stage-1 global fit (coarse-to-fine k_ca
    grid + per-test minimize_scalar for j0_an) is run per temperature group.

Usage
-----
  # Full pipeline (HFR + E_cell) for all available tests:
  python scripts/Experimental/fitting/run_all_fits.py

  # Skip Phase 1 (HFR already fitted), run only Phase 2:
  python scripts/Experimental/fitting/run_all_fits.py --skip-hfr

  # Restrict to specific test numbers:
  python scripts/Experimental/fitting/run_all_fits.py --tests 5 16 25

  # Pass extra args to the HFR fit (differential_evolution):
  python scripts/Experimental/fitting/run_all_fits.py --workers 8 --maxiter 20 --popsize 20

  # Pass extra args to the E_cell fit (coarse-to-fine k_ca grid):
  python scripts/Experimental/fitting/run_all_fits.py --workers 8 --grid-points 24 --refine-points 12

  # Phase 1 only:
  python scripts/Experimental/fitting/run_all_fits.py --skip-ecell
  python scripts/Experimental/fitting/run_all_fits.py --skip-ecell --workers 10 --tests 1 2 3
  python scripts/Experimental/fitting/run_all_fits.py --skip-ecell --workers 10 --tests 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
"""
import sys
import os
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from collections import defaultdict

_ROOT       = Path(__file__).parent.parent.parent.parent
_HFR_CSV    = _ROOT / "data" / "experimental" / "HFR.csv"
_ECELL_CSV  = _ROOT / "data" / "experimental" / "E_cell.csv"
_FIT_HFR    = Path(__file__).parent / "fit_HFR_v2.py"
_FIT_ECELL  = Path(__file__).parent / "fit_e_cell_global_v2.py"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests",      type=int, nargs="+", default=None,
                   help="Restrict to these test numbers (default: all in HFR.csv)")
    p.add_argument("--skip-hfr",   action="store_true",
                   help="Skip Phase 1 (HFR fits already done)")
    p.add_argument("--skip-ecell", action="store_true",
                   help="Skip Phase 2 (E_cell global fit)")
    p.add_argument("--workers",    type=int, default=-1,
                   help="Parallel workers (both phases)")
    # -- Phase 1 (fit_HFR_v2.py, differential_evolution) ------------------------
    p.add_argument("--maxiter",    type=int, default=10)
    p.add_argument("--popsize",    type=int, default=15)
    p.add_argument("--seed",       type=int, default=42)
    # -- Phase 2 (fit_e_cell_global_v2.py, coarse-to-fine k_ca grid) -------------
    p.add_argument("--grid-points",   type=int,   default=16)
    p.add_argument("--refine-points", type=int,   default=8)
    p.add_argument("--refine-span",   type=float, default=1.5)
    p.add_argument("--inner-xatol",   type=float, default=0.03)
    p.add_argument("--inner-maxiter", type=int,   default=20)
    p.add_argument("--polish-xatol",  type=float, default=1e-3)
    p.add_argument("--polish-maxiter", type=int,  default=40)
    return p.parse_args()


def _run(cmd: list[str], label: str) -> int:
    print(f"\n{'='*70}", flush=True)
    print(f"  {label}", flush=True)
    print(f"  CMD: {' '.join(str(c) for c in cmd)}", flush=True)
    print(f"{'='*70}", flush=True)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_ROOT) + (os.pathsep + existing if existing else "")
    result = subprocess.run([sys.executable] + [str(c) for c in cmd], env=env)
    if result.returncode != 0:
        print(f"\n[WARNING] {label} exited with code {result.returncode}", flush=True)
    return result.returncode


def main():
    args = parse_args()

    # -- Discover available tests ----------------------------------------------
    hfr_data   = pd.read_csv(_HFR_CSV)
    ecell_data = pd.read_csv(_ECELL_CSV) if _ECELL_CSV.exists() else pd.DataFrame()

    hfr_tests_all = sorted(hfr_data["test_no"].unique().astype(int).tolist())
    if args.tests:
        hfr_tests = [t for t in args.tests if t in hfr_tests_all]
        missing   = [t for t in args.tests if t not in hfr_tests_all]
        if missing:
            print(f"[WARNING] Tests not found in HFR.csv and will be skipped: {missing}")
    else:
        hfr_tests = hfr_tests_all

    print(f"\nHFR tests to process: {hfr_tests}")

    de_extra = [
        "--workers", str(args.workers),
        "--maxiter", str(args.maxiter),
        "--popsize", str(args.popsize),
        "--seed",    str(args.seed),
    ]
    ecell_extra = [
        "--workers",        str(args.workers),
        "--grid-points",    str(args.grid_points),
        "--refine-points",  str(args.refine_points),
        "--refine-span",    str(args.refine_span),
        "--inner-xatol",    str(args.inner_xatol),
        "--inner-maxiter",  str(args.inner_maxiter),
        "--polish-xatol",   str(args.polish_xatol),
        "--polish-maxiter", str(args.polish_maxiter),
    ]

    # -- Phase 1: HFR fits -----------------------------------------------------
    hfr_failed = []
    if not args.skip_hfr:
        print(f"\n{'#'*70}")
        print(f"  PHASE 1 — HFR fits ({len(hfr_tests)} tests)")
        print(f"{'#'*70}")
        for test_no in hfr_tests:
            rc = _run(
                [_FIT_HFR, "--test", str(test_no)] + de_extra,
                f"fit_HFR_v2  test {test_no}",
            )
            if rc != 0:
                hfr_failed.append(test_no)
        if hfr_failed:
            print(f"\n[WARNING] HFR fits failed for tests: {hfr_failed}")
    else:
        print("\n[skip] Phase 1 (--skip-hfr)")

    # -- Phase 2: E_cell global fits -------------------------------------------
    if args.skip_ecell:
        print("\n[skip] Phase 2 (--skip-ecell)")
        return

    if ecell_data.empty:
        print(f"\n[WARNING] {_ECELL_CSV} not found — skipping Phase 2.")
        return

    print(f"\n{'#'*70}")
    print(f"  PHASE 2 — E_cell global fits (Stage 1, grouped by temperature)")
    print(f"{'#'*70}")

    # Tests that have both HFR and E_cell data, excluding HFR failures
    ecell_tests_all = set(ecell_data["test_no"].unique().astype(int).tolist())
    hfr_succeeded   = [t for t in hfr_tests if t not in hfr_failed]
    ecell_candidates = [t for t in hfr_succeeded if t in ecell_tests_all]

    if not ecell_candidates:
        print("[WARNING] No tests have both HFR and E_cell data — skipping Phase 2.")
        return

    # Group by temperature (use E_cell data's T column for grouping)
    temp_groups: dict[float, list[int]] = defaultdict(list)
    for test_no in ecell_candidates:
        T = float(ecell_data[ecell_data["test_no"] == test_no]["T(°C)"].iloc[0])
        temp_groups[T].append(test_no)

    print(f"\nTemperature groups for Stage-1 fit:")
    for T, tests in sorted(temp_groups.items()):
        print(f"  T={T}°C  ->  tests {tests}")

    ecell_failed = []
    for T, tests in sorted(temp_groups.items()):
        tests_str = ",".join(str(t) for t in sorted(tests))
        rc = _run(
            [_FIT_ECELL, "--stage", "1", "--tests", tests_str] + ecell_extra,
            f"fit_e_cell_global_v2  Stage 1  T={T}°C  tests=[{tests_str}]",
        )
        if rc != 0:
            ecell_failed.append((T, tests))

    # -- Summary ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*70}")
    if not args.skip_hfr:
        ok  = [t for t in hfr_tests if t not in hfr_failed]
        print(f"  Phase 1 HFR:    {len(ok)}/{len(hfr_tests)} succeeded"
              + (f"  failed={hfr_failed}" if hfr_failed else ""))
    if not args.skip_ecell:
        n_grp = len(temp_groups)
        n_fail = len(ecell_failed)
        print(f"  Phase 2 E_cell: {n_grp - n_fail}/{n_grp} temperature groups succeeded"
              + (f"  failed={ecell_failed}" if ecell_failed else ""))


if __name__ == "__main__":
    main()
