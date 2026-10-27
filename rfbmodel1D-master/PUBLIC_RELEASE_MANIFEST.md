# Public release manifest

Working list of what carries over to the clean public repo. Nothing has been
copied yet — review/edit this, then we build the new repo from the KEEP list.

## KEEP — core package

- `rfbmodel1D/` (entire package, 33 files)
- `setup.py`
- `.gitignore` (carry over, maybe tighten further — see NEVER INCLUDE)

## KEEP — scripts

- `scripts/__init__.py`
- `scripts/functions/` (`__init__.py`, `calculate_current_density.py`, `membrane_model.py`)
- `scripts/simulation/` (all files + `README.md`)
- `scripts/Validation/` (`check_conservation.py`, `grid_convergence_cathode.py`,
  `grid_convergence_membrane.py`, `plot_diffusivity_model.py`,
  `validate_sorption.py`, `diffusivity_model_comparison.pdf`)
- `scripts/Experimental/` -> rename to `scripts/fitting/`, keep only:
  - `fit_HFR_v2.py`
  - `fit_e_cell_global_v2.py`
  - `run_all_fits.py` (**update to call `fit_e_cell_global_v2.py`, not v1, before publishing**)
  - `run_hfr_all_tests.py`
  - `plot_hfr_by_condition.py`
  - `summarize_hfr_fits.py`
  - `simulate_HFR.py` (imported as a helper module by the fit scripts — required)
  - `simulate_e_cell_experiments.py`
  - `simulate_sensitivity.py`
  - `water_fit_diffusion.py`
  - `water_fit_sorption.py`
  - `fit_cathode_a_w.py`

## DROP — superseded / one-off scripts

- `scripts/Experimental/fit_e_cell.py` — superseded by `fit_e_cell_global_v2.py`
- `scripts/Experimental/fit_e_cell_global.py` — superseded by v2 (drop v1 unless
  you want to keep it for provenance of the DE->grid-search rationale)
- `scripts/Experimental/_diag_test25_high_charge.py` — one-off debug script

## KEEP — notebooks

- `notebooks/plotting/` (all 8 — these are the publication figures)
- `notebooks/unit_test/` (all subfolders incl. `.gif` validation animations)
- `notebooks/parameter_sensitivity/` (all 3)

## REVIEW — likely drop, confirm with you first

- `notebooks/complete_cell/debug_initialization.ipynb`
- `notebooks/complete_cell/debug_membrane.ipynb`
- `notebooks/complete_cell/debug_time_integration.ipynb`
- `notebooks/fitting/e_cell_fit_manual.ipynb`
- `notebooks/fitting/hfr_fit_multiplier.ipynb`
- `notebooks/fitting/test_time_evolution.ipynb`
- `notebooks/fitting/diffusivity_k_fitting.ipynb` — possibly duplicates
  `scripts/Experimental/water_fit_diffusion.py`; keep whichever is the real
  source of truth, drop the other

## KEEP — data & figures

- `data/experimental/` (all CSVs + `README.md`)
- `figures/` (all 34 PDFs — curated thesis figures)

## REVIEW — results

- `results/` — currently has 35+ tracked fit-result JSONs plus `.npz` files.
  All regenerable from `data/` + the fit scripts. Recommend: don't carry over
  wholesale. Either (a) drop entirely, or (b) hand-pick the final/best fit
  JSON per test (one file each, not the full timestamped history) if you want
  to ship reproducible "these are the parameters used in the thesis" values.

## NEVER INCLUDE

- `.conda/Library/bin/libcrypto-3-x64.dll.c~.conda_trash.conda_trash` — accidental
  conda trash file, currently tracked in git
- `.claude/settings.local.json`, `.vscode/settings.json` — personal tool config
- `prompts/` — internal debugging notes (`dae_debugging.txt`,
  `dae_improvements.txt`, `fitting.txt`, `membrane_debugging.txt`).
  `prompts/Model Equations.pdf` — keep only if it's meant as real
  documentation; otherwise drop with the rest.
- `build/`, `*.egg-info/`, `__pycache__/` — build artifacts (already gitignored,
  just confirm none are tracked)
