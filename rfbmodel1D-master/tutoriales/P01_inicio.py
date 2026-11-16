"""
P01 — Uptake de vanadio y conductividad de la membrana
======================================================
Guía: docs/proyectos/P01_uptake_vanadio.md

Qué hace este script:
  Paso 1. Corre la celda a tres SOC y mide la caída óhmica en la membrana.
  Paso 2. Lee la composición del electrolito en la cara cátodo/membrana (entrada del Donnan).
  Paso 3. Engancha un factor sobre κ de la membrana con una subclase de Cell:
          f = κ_Donnan(composición de la cara) / κ_Donnan(misma composición sin vanadio).

Uso (desde rfbmodel1D-master/):
    python tutoriales/P01_inicio.py base      # factor = 1: reproduce la tabla del Paso 1
    python tutoriales/P01_inicio.py const     # factor 0,8 constante a SOC 0,75 (referencia de la guía)
    python tutoriales/P01_inicio.py donnan    # Tabla 1: base vs. Donnan a SOC 0,2 / 0,5 / 0,9  (por defecto)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from iiq3843_utils import configurar, construir_celda, integrar, perfiles, voltaje
from rfbmodel1D.cells.cell import Cell
from rfbmodel1D.parameters.params import Vars

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from donnan_membrana import donnan, kappa_donnan, sin_vanadio, solucion

SALIDA = Path("resultados"); SALIDA.mkdir(exist_ok=True)
J = -500.0                  # densidad de corriente [A/m2], negativa = descarga

# Qué factor aplica CeldaDonnan:  1.0 (base), un número (constante) o "donnan".
MODO_FACTOR = 1.0


# ---- Paso 2: composición en la cara cátodo/membrana --------------------------------
def composicion_interfaz(cell):
    """Concentraciones del electrolito [mol/m3] en el lado cátodo de la cara con la membrana.

    El modelo guarda en cell.bc los valores de interfaz que recalcula en cada llamada al residuo:
      - update_vanadium() escribe c_V4*0,1229 y c_V5*0,0952 como condición de la membrana;
        dividiendo por esos coeficientes se recupera la concentración del último nodo del cátodo.
      - el H+ de la cara es la incógnita de acoplamiento c_H* (Vars.c_h, cátodo, derecha).
      - el HSO4- NO está en cell.bc (su borde es Neumann 0). CeldaDonnan guarda el último nodo
        del cátodo en self._nodo_cara al evaluar update_kappa_io_cathode.
      - c_H_nodo: H+ del último nodo del cátodo (estado actual). Es el que usa factor_kappa:
        c_H* en cell.bc lo escribe update_BC_cathode_membrane DESPUÉS de update_kappa_io_m,
        así que dentro del residuo trae el valor de la evaluación anterior (vale 0 en la primera).
    """
    c_V4 = cell.bc[Vars.c_v4]["membrane"]["left"].value / 0.1229
    c_V5 = cell.bc[Vars.c_v5]["membrane"]["left"].value / 0.0952
    c_H  = cell.bc[Vars.c_h]["cathode"]["right"].value
    nodo = getattr(cell, "_nodo_cara", {"c_H": np.nan, "c_HSO4": np.nan})
    return {"c_V4": c_V4, "c_V5": c_V5, "c_H": c_H,
            "c_H_nodo": nodo["c_H"], "c_HSO4": nodo["c_HSO4"]}


# ---- Paso 3: enganche de κ -----------------------------------------------------------
def factor_kappa(cell, comp):
    """f = κ_Donnan(comp) / κ_Donnan(comp sin vanadio)   (o 1 / constante, según MODO_FACTOR)."""
    if MODO_FACTOR != "donnan":
        return float(MODO_FACTOR)
    # Las cuatro del último nodo del cátodo (estado actual); SO4 por electroneutralidad
    c_s = solucion(comp["c_V4"], comp["c_V5"], comp["c_H_nodo"], comp["c_HSO4"])
    c_f, T = cell.membrane.c_f, cell.T
    _, c_m  = donnan(c_s, c_f, T)                  # con vanadio
    _, c_m0 = donnan(sin_vanadio(c_s), c_f, T)     # sin vanadio (SO4 recalculado)
    return kappa_donnan(c_m, T) / kappa_donnan(c_m0, T)


class CeldaDonnan(Cell):
    """Cell con κ de membrana corregido por vanadio absorbido.

    El residuo llama, en este orden, a update_vanadium, self.update_kappa_io_cathode(y_cathode)
    y self.update_kappa_io_m(...). Redefinir ambos métodos aquí basta para que el modelo use
    esta versión sin tocar rfbmodel1D/.
    """
    def update_kappa_io_cathode(self, y_cathode):
        super().update_kappa_io_cathode(y_cathode)
        n = self.layout.n_vars_cathode
        self._nodo_cara = {"c_H":    float(y_cathode[-n + 2]),    # último nodo del cátodo [mol/m3]
                           "c_HSO4": float(y_cathode[-n + 3])}

    def update_kappa_io_m(self, lambda_w, Phi_io_membrane):
        super().update_kappa_io_m(lambda_w, Phi_io_membrane)        # κ(λ) original [S/m]
        comp = composicion_interfaz(self)
        self.f_ultimo = factor_kappa(self, comp)
        self.membrane.kappa_io = self.membrane.kappa_io * self.f_ultimo


# ---- Una corrida ------------------------------------------------------------------------
def correr(soc, modo=1.0, t_final=300.0):
    """Corre la celda a un SOC con MODO_FACTOR = modo y devuelve un diccionario de resultados."""
    global MODO_FACTOR
    MODO_FACTOR = modo
    configurar(T_C=32.0, NC=100, NM=60)
    cell, y0, yp0 = construir_celda(soc, "galvanostatico", J, clase=CeldaDonnan)
    sol = integrar(cell, y0, yp0, t_final)
    y = sol.y[:, -1]
    p = perfiles(cell, y)                      # re-evalúa el residuo en y (actualiza κ, bc, i_io)
    comp = composicion_interfaz(cell)
    return {"soc": soc, "V": voltaje(cell, y),
            "dphi_m": p["phi_io_m"][-1] - p["phi_io_m"][0],   # caída óhmica en la membrana [V]
            "kappa_m": p["kappa_m"].mean(), "i_io": p["i_io"],
            "f": factor_kappa(cell, comp), **comp}


def imprimir(r, tag=""):
    print(f"{tag}SOC={r['soc']:.2f}: V={r['V']:.4f} V  ΔΦ_m={r['dphi_m']*1e3:.2f} mV  "
          f"κ_m medio={r['kappa_m']:.3f} S/m  f={r['f']:.4f}  i_io={r['i_io']:.2f} A/m2  "
          f"cara: c_V4={r['c_V4']:.0f} c_V5={r['c_V5']:.0f} c_H*={r['c_H']:.0f} "
          f"c_H,nodo={r['c_H_nodo']:.0f} c_HSO4={r['c_HSO4']:.0f}")


# ---- Paso 1 (factor = 1): reproduce la tabla base ------------------------------------
def paso1_base():
    filas = []
    for soc in (0.2, 0.5, 0.9):
        r = correr(soc, 1.0); imprimir(r)
        filas.append((soc, r["V"], r["dphi_m"], r["kappa_m"], r["c_V4"], r["c_V5"], r["c_H"]))
    f = np.array(filas)
    np.savetxt(SALIDA / "P01_base_factor1.csv", f, delimiter=",",
               header="soc,V,dPhi_m_V,kappa_m_Sm,c_V4,c_V5,c_H", comments="")
    ref = SALIDA / "P01_base.csv"
    if ref.exists():
        g = np.loadtxt(ref, delimiter=",", skiprows=1)
        print(f"Comparación con {ref}: idéntico bit a bit = {np.array_equal(f, g)}, "
              f"máx. diferencia relativa = {np.max(np.abs(f - g) / np.abs(g)):.1e}")


# ---- Referencia de la guía: factor 0,8 constante a SOC 0,75 -------------------------
def referencia_const():
    for modo in (1.0, 0.8):
        imprimir(correr(0.75, modo), tag=f"factor={modo}: ")
    print("Guía: factor 0,8 → ΔΦ_m = 17,04 mV, V = 1,1209 V;  sin factor → 13,63 mV, 1,1247 V")


# ---- Paso 7: Tabla 1, base vs. Donnan ------------------------------------------------
def tabla1():
    filas = []
    for soc in (0.2, 0.5, 0.9):
        rb = correr(soc, 1.0);      imprimir(rb, "base:   ")
        rd = correr(soc, "donnan"); imprimir(rd, "Donnan: ")
        for r in (rb, rd):
            assert abs(r["i_io"] - J) < 5e-3, f"i_io = {r['i_io']} ≠ {J} A/m2"
        frac = (rd["dphi_m"] - rb["dphi_m"]) / rd["dphi_m"]
        filas.append((soc, rd["f"], rb["dphi_m"] * 1e3, rd["dphi_m"] * 1e3, rb["V"], rd["V"],
                      (rd["V"] - rb["V"]) * 1e3, frac, rb["i_io"], rd["i_io"]))
    t = np.array(filas)
    cab = ("soc,f_cara,dPhi_m_base_mV,dPhi_m_Donnan_mV,V_base_V,V_Donnan_V,deltaV_mV,"
           "frac_vanadio,i_io_base_Am2,i_io_Donnan_Am2")
    np.savetxt(SALIDA / "P01_tabla1.csv", t, delimiter=",", header=cab, comments="",
               fmt=["%.2f", "%.5f", "%.4f", "%.4f", "%.5f", "%.5f", "%.4f", "%.5f", "%.3f", "%.3f"])
    print("\n" + cab.replace(",", " | "))
    for row in t:
        print(" | ".join(f"{v:.4f}" for v in row))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(t[:, 0], np.abs(t[:, 2]), "o-", label="base (f = 1)")
    ax[0].plot(t[:, 0], np.abs(t[:, 3]), "o-", label="Donnan")
    ax[0].set(xlabel="SOC", ylabel="|ΔΦ_m| [mV] a 500 A/m2"); ax[0].legend(frameon=False)
    ax[1].plot(t[:, 0], t[:, 6], "o-", color="#2a78d6")
    ax[1].set(xlabel="SOC", ylabel="ΔV = V_Donnan − V_base [mV]")
    fig.tight_layout(); fig.savefig(SALIDA / "P01_tabla1.png", dpi=130)
    print("Listo: resultados/P01_tabla1.csv y P01_tabla1.png")


if __name__ == "__main__":
    accion = sys.argv[1] if len(sys.argv) > 1 else "donnan"
    {"base": paso1_base, "const": referencia_const, "donnan": tabla1}[accion]()
