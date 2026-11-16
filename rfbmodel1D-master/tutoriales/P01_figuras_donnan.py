"""
P01 — Fig. 1 y Fig. 2 (solo donnan_membrana.py, sin simular la celda)
=====================================================================
Composición de la solución = tanque (guía, Paso 5):
    c_V4 = (1 − SoC)·1000, c_V5 = SoC·1000, c_H = 4963, c_HSO4 = 4963·3/5 mol/m3, T = 305,15 K

    python tutoriales/P01_figuras_donnan.py     (desde rfbmodel1D-master/)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from donnan_membrana import C_F, donnan, kappa_donnan, sin_vanadio, solucion_tanque

SALIDA = Path("resultados"); SALIDA.mkdir(exist_ok=True)
T = 305.15
soc = np.linspace(0.1, 0.9, 81)

# ---- Cálculo en la malla de SoC -------------------------------------------------------------
especies = ["H", "V4", "V5", "HSO4", "SO4"]
c_m = {e: np.empty_like(soc) for e in especies}
k_con, k_sin, dphi = (np.empty_like(soc) for _ in range(3))
for j, s in enumerate(soc):
    c_s = solucion_tanque(s)
    dphi[j], cm = donnan(c_s, C_F, T)
    _, cm0 = donnan(sin_vanadio(c_s), C_F, T)
    for e in especies:
        c_m[e][j] = cm[e]
    k_con[j], k_sin[j] = kappa_donnan(cm, T), kappa_donnan(cm0, T)
f = k_con / k_sin

# ---- Estilo: líneas de 2 px, ejes discretos, colores categóricos en orden fijo --------------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3dd", "grid.linewidth": 0.8,
                     "axes.edgecolor": "#8a897f", "font.size": 10, "lines.linewidth": 2})
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
nombre = {"H": "H$^+$", "V4": "VO$^{2+}$ (V(IV), z=2)", "V5": "VO$_2^+$ (V(V), z=1)",
          "HSO4": "HSO$_4^-$", "SO4": "SO$_4^{2-}$"}

# ---- Fig. 1: concentraciones dentro de la membrana -----------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for e, c in zip(especies, COL):
    ax.plot(soc, c_m[e], color=c, label=nombre[e])
    ax.annotate(nombre[e], (soc[-1], c_m[e][-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color="#3d3d3a")
ax.axhline(C_F, color="#8a897f", lw=1, ls="--")
ax.text(0.1, C_F + 80, "$c_f$ = 1390", fontsize=8, color="#5e5d56")
ax.set(xlabel="SoC [-]", ylabel="$c_{i,m}$ [mol/m$^3$]", xlim=(0.1, 1.08), ylim=(0, None),
       title="Fig. 1 — Concentraciones en la membrana (equilibrio Donnan)")
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.62), frameon=False)
fig.tight_layout(); fig.savefig(SALIDA / "P01_fig1_cm_vs_soc.png", dpi=150)

# ---- Fig. 2: κ_D con y sin vanadio (izq.) y f (der.) — dos paneles, no doble eje -----------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(soc, k_sin, color=COL[0], label="sin vanadio")
ax[0].plot(soc, k_con, color=COL[1], label="con vanadio")
ax[0].set(xlabel="SoC [-]", ylabel="$\\kappa_D$ [S/m]", title="$\\kappa_D$ por Nernst–Einstein")
ax[0].legend(frameon=False)
ax[1].plot(soc, f, color=COL[0])
for s in (0.2, 0.5, 0.9):
    j = np.argmin(abs(soc - s))
    ax[1].plot(soc[j], f[j], "o", ms=6, color=COL[0], mec="white", mew=1.5)
    ax[1].annotate(f"{f[j]:.4f}", (soc[j], f[j]), xytext=(0, 8), textcoords="offset points",
                   ha="center", fontsize=8)
ax[1].set(xlabel="SoC [-]", ylabel="f = $\\kappa_D$/$\\kappa_D$(sin V) [-]",
          title="Factor f(SoC)")
fig.suptitle("Fig. 2 — Conductividad Donnan con/sin vanadio (composición de tanque)", fontsize=11)
fig.tight_layout(); fig.savefig(SALIDA / "P01_fig2_kappa_f.png", dpi=150)

# ---- Datos de respaldo ----------------------------------------------------------------------
np.savetxt(SALIDA / "P01_fig12_datos.csv",
           np.column_stack([soc, dphi, *[c_m[e] for e in especies], k_con, k_sin, f]),
           delimiter=",", comments="", fmt="%.8g",
           header="soc,dphi_D_V,c_H_m,c_V4_m,c_V5_m,c_HSO4_m,c_SO4_m,kappa_D_conV,kappa_D_sinV,f")
for s in (0.2, 0.5, 0.9):
    j = np.argmin(abs(soc - s))
    print(f"SoC={s:.1f}: Δφ_D={dphi[j]*1e3:.2f} mV  κ_D con V={k_con[j]:.2f}  sin V={k_sin[j]:.2f} S/m  f={f[j]:.4f}")
print("Listo: resultados/P01_fig1_cm_vs_soc.png, P01_fig2_kappa_f.png, P01_fig12_datos.csv")
