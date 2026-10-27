"""
Plot E_cell(j) grouped by (SOC, Q_v), one figure per condition, with a curve
per test (charge + discharge together) — publication style matching
notebooks/plotting/e_cell_global_publication.ipynb, but sourced from the
*global* k_ca fit (fit_e_cell_global_v2.py Stage 1: one shared cathode rate
constant per temperature, a separate anode exchange current j0_an per test)
instead of a per-test individual fit.

Marker colour denotes cell temperature (magenta: 40 °C; olive: 24 °C;
green: 32 °C), while filled and open markers distinguish charging and
discharging half-cycles, respectively, within each test. In the
SOC 50% / Q_v 45 ml/min condition, two additional replicate measurements
at 32 °C are shown as x (Repeat 1, test 15) and + (Repeat 2, test 14) —
experimental points only, no model curve, to avoid overlapping the primary
32 °C line (test 16).

Pure plotting — reads the CSV produced by run_e_cell_global_all.py and has
no simulation dependencies (pandas + matplotlib only), so it's safe to edit
freely and re-run instantly without touching the DAE solver.

Input
-----
  results/test fitting/e_cell_global_by_condition/e_cell_global_simulated_all.csv
  (run scripts/Experimental/run_e_cell_global_all.py first to generate it)

Usage
-----
  python scripts/Experimental/plot_e_cell_global_by_condition.py
  python scripts/Experimental/plot_e_cell_global_by_condition.py --csv path/to/other.csv --out-dir path/to/figs
"""
import argparse
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

_ROOT        = Path(__file__).parent.parent.parent
_RES         = _ROOT / "results" / "test fitting"
_OUT_DIR     = _RES / "e_cell_global_by_condition"
_DEFAULT_CSV = _OUT_DIR / "e_cell_global_simulated_all.csv"

# Fixed color per cell temperature.
_T_COLORS = {
    24: "#8c8c00",  # olive
    32: "#1a9850",  # green
    40: "#c51b7d",  # magenta
}

# Replicate measurements (SOC 50% / Q_v 45 ml/min only): marker shape and
# model-line style denote the repeat identity (color still follows T).
_REPEAT_MARKERS = {
    15: ("x", "Repeat 1", "--"),
    14: ("+", "Repeat 2", ":"),
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'axes.linewidth': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 1.4,
    'savefig.dpi': 600,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=str, default=None,
                   help=f"Input CSV (default: {_DEFAULT_CSV})")
    p.add_argument("--out-dir", type=str, default=None,
                   help=f"Output directory for plots (default: {_OUT_DIR})")
    return p.parse_args()


def _t_color(T: float) -> str:
    key = int(round(T))
    if key not in _T_COLORS:
        raise KeyError(f"No color defined for T={T}°C. Add it to _T_COLORS.")
    return _T_COLORS[key]


def main():
    args     = parse_args()
    csv_path = Path(args.csv) if args.csv else _DEFAULT_CSV
    out_dir  = Path(args.out_dir) if args.out_dir else _OUT_DIR

    if not csv_path.exists():
        raise SystemExit(
            f"{csv_path} not found. Run "
            f"scripts/Experimental/run_e_cell_global_all.py first to generate it."
        )

    df = pd.read_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = df.groupby(["SOC", "Q_v (ml/min)"])
    print(f"{groups.ngroups} (SOC, Q_v) condition group(s) -> plotting...")

    for (soc, Qv), g in groups:
        test_nos = sorted(g["test_no"].unique().astype(int).tolist())

        fig, ax = plt.subplots(figsize=(4.5, 4))
        temps_plotted = {}   # T (rounded) -> (color, test_no), for the legend

        for t in test_nos:
            gt = g[g["test_no"] == t]
            T  = float(gt["T(°C)"].iloc[0])
            c  = _t_color(T)

            g_ch  = gt[gt["mode"] == "charge"].sort_values("j_appl (A/m2)")
            g_dis = gt[gt["mode"] == "discharge"].sort_values("j_appl (A/m2)")

            if t in _REPEAT_MARKERS:
                # Replicate: same color as its temperature, own linestyle for
                # the model curve and own marker shape for the experimental points.
                marker, _, linestyle = _REPEAT_MARKERS[t]
                for gm in (g_ch, g_dis):
                    sim_ok = gm.dropna(subset=["E_sim (V)"])
                    ax.plot(sim_ok["j_appl (A/m2)"], sim_ok["E_sim (V)"],
                            linestyle, color=c, lw=1.4)
                g_all = pd.concat([g_ch, g_dis])
                ax.plot(g_all["j_appl (A/m2)"], g_all["E_exp (V)"],
                        marker, ms=6, mec=c, mew=1.2, ls='none')
                continue

            temps_plotted[int(round(T))] = (c, t)

            # Model (solid line, no marker) — charge and discharge together
            for gm in (g_ch, g_dis):
                sim_ok = gm.dropna(subset=["E_sim (V)"])
                ax.plot(sim_ok["j_appl (A/m2)"], sim_ok["E_sim (V)"],
                        '-', color=c, lw=1.4)

            # Experimental: filled circle = charge, open circle = discharge
            ax.plot(g_ch["j_appl (A/m2)"], g_ch["E_exp (V)"],
                    'o', ms=5, mfc=c, mec=c, mew=0.9, ls='none')
            ax.plot(g_dis["j_appl (A/m2)"], g_dis["E_exp (V)"],
                    'o', ms=5, mfc='none', mec=c, mew=0.9, ls='none')

        ax.set_xlabel(r'Current density / A m$^{-2}$')
        ax.set_ylabel(r'$E_{cell}$ / V')
        ax.set_title(f'SOC={soc:.0%}  Q$_v$={Qv:.0f} ml/min')
        ax.set_ylim(0.3, 1.6)
        ax.set_xlim(0, 6000)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Legend: one colored line per temperature, marker-meaning keys,
        # and (where present) the two replicate markers.
        temp_handles = [Line2D([0], [0], color=c, lw=1.4, label=f"T={T}°C (test {test_no})")
                        for T, (c, test_no) in sorted(temps_plotted.items())]
        marker_handles = [
            Line2D([0], [0], marker='o', ms=5, mfc='0.3', mec='0.3', ls='none', label='Exp., charge'),
            Line2D([0], [0], marker='o', ms=5, mfc='none', mec='0.3', ls='none', label='Exp., discharge'),
        ]
        repeat_handles = [
            Line2D([0], [0], marker=_REPEAT_MARKERS[t][0], ms=6,
                    color=_t_color(g[g["test_no"] == t]["T(°C)"].iloc[0]),
                    mec=_t_color(g[g["test_no"] == t]["T(°C)"].iloc[0]),
                    mew=1.2, lw=1.4, ls=_REPEAT_MARKERS[t][2],
                    label=f"{_REPEAT_MARKERS[t][1]} (test {t})")
            for t in test_nos if t in _REPEAT_MARKERS
        ]
        ax.legend(handles=temp_handles + marker_handles + repeat_handles,
                  frameon=False, loc='best')

        fig.tight_layout(pad=0.4)
        out_path = out_dir / f"e_cell_global_SOC{soc:.1f}_Qv{int(Qv)}.pdf"
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
