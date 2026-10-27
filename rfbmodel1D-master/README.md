# rfbmodel1D

A 1-D, transient DAE model of a full **H₂–V regenerative redox flow battery (RFB)** cell,
coupling hydrogen anode kinetics, vanadium cathode electrochemistry, and membrane water/proton
transport. The model is fit against experimental polarization (E-cell) and high-frequency
resistance (HFR) data and used to decompose overpotential contributions across operating
conditions (state of charge, catholyte flow rate, temperature).

This repository accompanies the paper:

> *TODO: Authors. "Title of the paper." Journal, Year. https://doi.org/...*

If you use this code or the digitized datasets in `data/experimental/`, please cite the paper
above (and the original experimental sources listed in [`data/experimental/README.md`](data/experimental/README.md)).

## Repository structure

```
rfbmodel1D/            Core package: cell/subdomain models, DAE residuals, electrochemistry, meshing
scripts/
  simulation/           Run a single simulation from the CLI (see scripts/simulation/README.md)
  fitting/               Parameter-fitting routines (HFR, E-cell) against experimental data
  Validation/            Conservation checks and mesh-convergence studies
notebooks/
  plotting/               Notebooks that generate the paper's figures
  parameter_sensitivity/  Sensitivity analyses (configuration and operating parameters)
  unit_test/              Isolated validation of individual transport terms (diffusion, migration, ...)
data/experimental/     Experimental datasets used for fitting/validation, with sources documented
figures/                Curated output figures
setup.py
requirements.txt
```

## Installation

Requires Python 3.10+.

```bash
git clone <this-repo-url>
cd rfbmodel1D
pip install -r requirements.txt
pip install -e .
```

## Quick start

Run a simulation from the command line:

```bash
python scripts/simulation/run_simulation.py --tf 10 --soc 0.8 --rtol 1e-4 --out my_run
```

This writes `my_run.npz`, loadable with:

```python
import numpy as np
data = np.load("my_run.npz")
tsol = data["tsol"]   # time points [s]
ysol = data["ysol"]   # state vectors (n_vars x n_time)
```

See [`scripts/simulation/README.md`](scripts/simulation/README.md) for the full CLI reference and
the state-vector layout.

## Reproducing the paper's figures

The notebooks under `notebooks/plotting/` regenerate the figures used in the paper from the data
in `data/experimental/` and the fitted parameters in `rfbmodel1D/parameters/params.py`. Each
notebook is self-contained; run cells top to bottom.

## Data

All experimental datasets, their sources, and column definitions are documented in
[`data/experimental/README.md`](data/experimental/README.md). Several datasets were digitized from
figures in the cited publications (e.g., via WebPlotDigitizer) rather than obtained as raw data
from the original authors.

## License

TODO: add a LICENSE file and name it here (e.g., MIT).

## Contact

Joaquín Barros — jbarrosvan@uc.cl
