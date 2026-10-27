"""
Usage:
python scripts/Validation/plot_diffusivity_model.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from rfbmodel1D.parameters.params import SimulationParams
from rfbmodel1D.cells.cell import Cell

T = 303  # K

# Okabe-Ito colorblind-safe palette. "This work" (hwang) is pinned to black
# so it reads as the reference curve rather than one of eight equals.
COLORS = ['#000000', '#E69F00', '#56B4E9', '#009E73',
          '#CC9900', '#0072B2', '#D55E00', '#CC79A7']
LINEWIDTHS = [2.2, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4]
LINESTYLES = ['-', '-', '-', '--', '-', '-', '-', '-']  # dashed = Zawodzinski,
# kept distinct since it currently sits ~1 order of magnitude below the rest
# above lambda~8 and is worth flagging visually pending a source check.

RC = {
    'font.family':      'serif',
    'font.size':         10,
    'axes.labelsize':    11,
    'axes.titlesize':    11,
    'axes.linewidth':    0.9,
    'legend.fontsize':   8,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.top':         True,
    'ytick.right':       True,
    'mathtext.fontset': 'cm',
}

# All models implemented in rfbmodel1D/cells/methods/membrane_water_diffusivity.py.
MODELS = ["hwang", "bimodal", "springer", "zawodzinski",
          "motupally", "gerteisen", "pino-munoz", "goshtabi"]
# Numbered tags cross-reference Table tab:diffusivity_models_summary in the SI
# -- pure numbers alone would force the reader back to the caption for every
# curve; pairing number + short name keeps quick visual ID without leaving
# the figure.
LABELS = ["This work", "[9] Zhang 2011", "[10] Springer 1991", "[11] Zawodz. 1991",
          "[12] Motupally 2024", "[13] Gerteisen 2009", "[14] Pino-Muñoz 2017", "[15] Goshtasbi 2016"]


def plot_diffusivity(out_dir: Path):
    lam_range = np.linspace(1, 22, 300)
    SimulationParams.CellParams.T = T
    cell = Cell(params=SimulationParams)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.4, 3.0))  # Elsevier single-column width
        for i, (label, model, color, lw, ls) in enumerate(
                zip(LABELS, MODELS, COLORS, LINEWIDTHS, LINESTYLES)):
            ax.plot(lam_range, cell.membrane_water_diffusivity(lam_range, model),
                    color=color, lw=lw, ls=ls, label=label,
                    zorder=5 if i == 0 else 2)  # keep "This work" on top
        ax.set_xlabel(r'Water content, $\lambda$')
        ax.set_ylabel(r'$D_\mathrm{w}$ (m$^2$ s$^{-1}$)')
        ax.set_title(f'$T = {T}$ K')
        ax.set_yscale('log')
        ax.set_xlim(0, 22)
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.25)
        ax.grid(False, which='minor')
        ax.set_axisbelow(True)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
                  ncol=2, frameon=False, columnspacing=1.2, handlelength=1.8,
                  handletextpad=0.5, labelspacing=0.4)
        fig.tight_layout(pad=0.5)
        out = out_dir / "diffusivity_model_comparison.pdf"
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
    print(f"Figure saved -> {out}")

def main():
    out_dir = Path(__file__).parent
    plot_diffusivity(out_dir)

if __name__ == "__main__":
    main()