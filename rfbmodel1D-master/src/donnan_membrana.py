"""
donnan_membrana.py — Equilibrio Donnan en la cara membrana/electrolito y κ por Nernst–Einstein
===============================================================================================
Proyecto P01 (IIQ3843). Todas las concentraciones en mol/m3, potenciales en V, T en K.

Especies (claves de los diccionarios):
    "H"    H+      z = +1
    "V4"   VO2+    z = +2   (vanadio IV)
    "V5"   VO2+    z = +1   (vanadio V, VO2+ "dioxo")
    "HSO4" HSO4-   z = -1
    "SO4"  SO4 2-  z = -2

Ecuaciones (guía P01, Paso 3):
    c_i,m = c_i,s · exp(-z_i F Δφ_D / RT)              (equilibrio Donnan, actividades = concentraciones)
    Σ z_i c_i,m − c_f = 0                              (electroneutralidad con carga fija −c_f)
    κ_D  = F²/(RT) · Σ z_i² D_i,m c_i,m                (Nernst–Einstein)

Supuestos:
    - actividades = concentraciones (sin coeficientes de actividad);
    - c_f por volumen de membrana hidratada (1390 mol/m3, MembraneParams);
    - D_i,m de MembraneParams, constantes (no dependen de λ);
    - MembraneParams no trae D para SO4 2-: se usa D_SO4,m = D_HSO4,m (4e-11 m2/s).
      Ojo: con este electrolito (c_H ≈ 5000 mol/m3 > c_f) el Δφ_D es de solo ≈ −2 mV, la
      exclusión de co-iones es débil y el SO4 2- dentro de la membrana es ~1300–1700 mol/m3.
      Aporta ~1–2 % de κ_D; en el cociente f casi se cancela.
"""
import numpy as np
from scipy.optimize import brentq

from rfbmodel1D.parameters.params import membraneparams, cellparams

# ---- Constantes y parámetros ------------------------------------------------------------
F = cellparams.F          # [C/mol]  mismas constantes que usa el modelo
R = cellparams.R          # [J/mol/K]

Z = {"H": +1, "V4": +2, "V5": +1, "HSO4": -1, "SO4": -2}

D_M = {                                   # difusividades dentro de la membrana [m2/s]
    "H":    membraneparams.D_H_plus,      # 3.35e-9
    "V4":   membraneparams.D_V4,          # 5e-12
    "V5":   membraneparams.D_V5,          # 1.17e-12
    "HSO4": membraneparams.D_HSO4_minus,  # 4e-11
    "SO4":  membraneparams.D_HSO4_minus,  # SUPUESTO: no hay dato propio para SO4 2-
}

C_F = membraneparams.c_f                  # 1390 mol/m3


# ---- Composición de la solución ------------------------------------------------------------
def solucion(c_V4, c_V5, c_H, c_HSO4):
    """Arma el diccionario de la solución y calcula SO4 2- por electroneutralidad.

    c_SO4 = (2 c_V4 + c_V5 + c_H − c_HSO4) / 2   (igual que update_kappa_io_cathode).
    Si se quita el vanadio (c_V4 = c_V5 = 0) el SO4 se recalcula aquí automáticamente.
    """
    c_SO4 = (2 * c_V4 + c_V5 + c_H - c_HSO4) / 2
    return {"H": c_H, "V4": c_V4, "V5": c_V5, "HSO4": c_HSO4, "SO4": c_SO4}


def sin_vanadio(c_s):
    """Misma solución, con c_V4 = c_V5 = 0 y SO4 recalculado."""
    return solucion(0.0, 0.0, c_s["H"], c_s["HSO4"])


# ---- Equilibrio Donnan ------------------------------------------------------------------------
def residuo_electroneutralidad(c_m, c_f=C_F):
    """Σ z_i c_i,m − c_f  [mol/m3]. Debe ser ~0 en la solución."""
    return sum(Z[i] * c for i, c in c_m.items()) - c_f


def _c_membrana(c_s, u):
    """c_i,m para un potencial adimensional u = F Δφ_D / (RT)."""
    return {i: c_s[i] * np.exp(-Z[i] * u) for i in c_s}


def donnan(c_s: dict, c_f=C_F, T=305.15, intervalo=0.3):
    """Resuelve el equilibrio Donnan.

    c_s       : concentraciones en la solución {especie: mol/m3} (claves de Z; puede faltar alguna)
    c_f       : carga fija de la membrana [mol/m3]
    T         : temperatura [K]
    intervalo : busca Δφ_D en [−intervalo, +intervalo] V

    Devuelve (dphi_D [V], c_m {especie: mol/m3}).
    Convención: Δφ_D = φ_membrana − φ_solución. Con carga fija negativa sale Δφ_D < 0.
    """
    fRT = F / (R * T)
    c_s = {i: float(c) for i, c in c_s.items() if c != 0.0}

    def g(dphi):
        return residuo_electroneutralidad(_c_membrana(c_s, fRT * dphi), c_f)

    # g(Δφ) es estrictamente decreciente (dg/dΔφ = −fRT Σ z_i² c_i,m < 0): raíz única.
    a, b = -intervalo, +intervalo
    if not (g(a) > 0 > g(b)):
        raise ValueError(f"Sin cambio de signo en ±{intervalo} V: g(a)={g(a):.3g}, g(b)={g(b):.3g}")
    dphi = brentq(g, a, b, xtol=1e-15, rtol=1e-15, maxiter=200)
    return dphi, _c_membrana(c_s, fRT * dphi)


# ---- Conductividad -----------------------------------------------------------------------------
def kappa_donnan(c_m: dict, T=305.15, D=D_M):
    """κ_D = F²/(RT) Σ z_i² D_i,m c_i,m   [S/m]."""
    return F**2 / (R * T) * sum(Z[i] ** 2 * D[i] * c for i, c in c_m.items())


def factor_f(c_s: dict, c_f=C_F, T=305.15):
    """f = κ_D(con vanadio) / κ_D(misma solución sin vanadio). Adimensional, 0 < f ≤ 1."""
    _, c_m = donnan(c_s, c_f, T)
    _, c_m0 = donnan(sin_vanadio(c_s), c_f, T)
    return kappa_donnan(c_m, T) / kappa_donnan(c_m0, T)


def solucion_tanque(soc, c_V=1000.0, c_H=4963.0):
    """Composición del tanque usada en la guía (Paso 5)."""
    return solucion((1 - soc) * c_V, soc * c_V, c_H, c_H * 3 / 5)
