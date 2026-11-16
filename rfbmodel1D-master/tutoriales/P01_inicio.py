"""
P01 — Uptake de vanadio y conductividad de la membrana (punto de partida)
=========================================================================
Guía: docs/proyectos/P01_uptake_vanadio.md

Qué hace este script (ya funciona tal cual):
  Paso 1. Corre la celda base a tres SOC y mide la caída óhmica en la membrana.
  Paso 2. Muestra cómo leer la composición del electrolito en la cara cátodo/membrana,
          que es la entrada de su equilibrio Donnan.
  Paso 3. Engancha un factor sobre κ de la membrana con una subclase de Cell.
          Hoy el factor vale 1 (no cambia nada); usted lo reemplaza por κ_Donnan(SoC)/κ_Donnan(sin V).

    python tutoriales/P01_inicio.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from iiq3843_utils import configurar, construir_celda, integrar, perfiles, voltaje
from rfbmodel1D.cells.cell import Cell
from rfbmodel1D.parameters.params import Vars

SALIDA = Path("resultados"); SALIDA.mkdir(exist_ok=True)


# ---- Paso 2: composición en la cara cátodo/membrana --------------------------------
def composicion_interfaz(cell):
    """Concentraciones del electrolito [mol/m3] en el lado cátodo de la cara con la membrana.

    El modelo guarda en cell.bc los valores de interfaz que recalcula en cada llamada al residuo:
      - update_vanadium() escribe c_V4*0,1229 y c_V5*0,0952 como condición de la membrana;
        dividiendo por esos coeficientes se recupera la concentración del último nodo del cátodo.
      - el H+ de la cara es la incógnita de acoplamiento c_H* (Vars.c_h, cátodo, derecha).
    """
    c_V4 = cell.bc[Vars.c_v4]["membrane"]["left"].value / 0.1229
    c_V5 = cell.bc[Vars.c_v5]["membrane"]["left"].value / 0.0952
    c_H  = cell.bc[Vars.c_h]["cathode"]["right"].value
    return {"c_V4": c_V4, "c_V5": c_V5, "c_H": c_H}


# ---- Paso 3: enganche de κ -----------------------------------------------------------
def factor_kappa(cell, comp):
    """TODO (P01): devuelva κ_Donnan(comp) / κ_Donnan(sin vanadio).

    Escriba κ_Donnan en src/donnan_membrana.py (ver la guía, sección 'Paso a paso') e
    impórtelo aquí. Con factor = 1 el modelo es idéntico a la base (prueba de consistencia).
    """
    return 1.0


class CeldaDonnan(Cell):
    """Cell con κ de membrana corregido por vanadio absorbido.

    El residuo llama a self.update_kappa_io_m(...) en cada evaluación; al redefinir el método
    aquí, el modelo usa su versión sin tocar rfbmodel1D/.
    """
    def update_kappa_io_m(self, lambda_w, Phi_io_membrane):
        super().update_kappa_io_m(lambda_w, Phi_io_membrane)        # κ(λ) original [S/m]
        comp = composicion_interfaz(self)
        self.membrane.kappa_io = self.membrane.kappa_io * factor_kappa(self, comp)


# ---- Paso 1: corridas a tres SOC -----------------------------------------------------
J = -500.0
filas = []
for soc in (0.2, 0.5, 0.9):
    configurar(T_C=32.0, NC=100, NM=60)
    cell, y0, yp0 = construir_celda(soc, "galvanostatico", J, clase=CeldaDonnan)
    sol = integrar(cell, y0, yp0, 300.0)
    y = sol.y[:, -1]
    p = perfiles(cell, y)
    dphi_m = p["phi_io_m"][-1] - p["phi_io_m"][0]          # caída óhmica en la membrana [V]
    comp = composicion_interfaz(cell)
    filas.append((soc, voltaje(cell, y), dphi_m, p["kappa_m"].mean(), comp["c_V4"], comp["c_V5"], comp["c_H"]))
    print(f"SOC={soc:.1f}: V={filas[-1][1]:.4f} V  ΔΦ_m={dphi_m*1e3:.2f} mV  κ_m medio={p['kappa_m'].mean():.3f} S/m  "
          f"interfaz c_V4={comp['c_V4']:.0f} c_V5={comp['c_V5']:.0f} c_H={comp['c_H']:.0f} mol/m3")

f = np.array(filas)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(f[:, 0], f[:, 3], "o-"); ax[0].set(xlabel="SOC", ylabel="κ membrana medio [S/m]")
ax[1].plot(f[:, 0], np.abs(f[:, 2]) * 1e3, "o-"); ax[1].set(xlabel="SOC", ylabel="|ΔΦ_m| [mV] a 500 A/m2")
fig.tight_layout(); fig.savefig(SALIDA / "P01_base.png", dpi=110)
np.savetxt(SALIDA / "P01_base.csv", f, delimiter=",",
           header="soc,V,dPhi_m_V,kappa_m_Sm,c_V4,c_V5,c_H", comments="")
print("Listo: resultados/P01_base.png y P01_base.csv. Ahora implemente factor_kappa().")
