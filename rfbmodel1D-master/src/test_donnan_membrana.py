"""
Pruebas de src/donnan_membrana.py (guía P01, Paso 5 y tabla 'Verificación').

    .venv/bin/python src/test_donnan_membrana.py      (o: pytest src/)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from donnan_membrana import (C_F, D_M, F, R, donnan, factor_f, kappa_donnan,
                             residuo_electroneutralidad, solucion, solucion_tanque)

T = 305.15


def test_electroneutralidad():
    """Residuo relativo < 1e-8 en todo el rango de SoC de la guía."""
    peor = 0.0
    for soc in np.linspace(0.1, 0.9, 81):
        _, c_m = donnan(solucion_tanque(soc), C_F, T)
        peor = max(peor, abs(residuo_electroneutralidad(c_m, C_F)) / C_F)
    assert peor < 1e-8, peor
    return peor


def test_analitico_1a1():
    """Solución 1:1 de solo H+ y HSO4-: c_H,m = c_f/2 + sqrt(c_f²/4 + c_s²)."""
    peor = 0.0
    for c_s in (10.0, 500.0, 1390.0, 4963.0, 10000.0):
        _, c_m = donnan({"H": c_s, "HSO4": c_s}, C_F, T)
        c_H_an = C_F / 2 + np.sqrt(C_F**2 / 4 + c_s**2)
        c_A_an = c_s**2 / c_H_an
        err = max(abs(c_m["H"] - c_H_an) / c_H_an, abs(c_m["HSO4"] - c_A_an) / c_A_an)
        peor = max(peor, err)
    assert peor < 1e-10, peor
    return peor


def test_kappa_solo_H():
    """Solo H+ (c_H,m = c_f): κ_D = F²/(RT)·D_H·c_f ≈ 17 S/m a 305 K."""
    k = kappa_donnan({"H": C_F}, T)
    assert 16.5 < k < 17.5, k
    assert np.isclose(k, F**2 / (R * T) * D_M["H"] * C_F, rtol=1e-14)
    return k


def test_f_tiende_a_1():
    """c_V → 0 ⇒ f → 1 (y 0 < f ≤ 1 siempre)."""
    fs = []
    for cV in (1000.0, 100.0, 10.0, 1.0, 1e-3, 1e-6):
        fs.append(factor_f(solucion(0.5 * cV, 0.5 * cV, 4963.0, 4963.0 * 3 / 5), C_F, T))
    assert all(0 < f <= 1 for f in fs), fs
    assert np.all(np.diff(fs) > 0), fs          # crece al bajar el vanadio
    assert abs(fs[-1] - 1) < 1e-8, fs[-1]
    return fs


if __name__ == "__main__":
    print(f"1) Electroneutralidad: peor residuo relativo = {test_electroneutralidad():.2e}  (< 1e-8)  OK")
    print(f"2) Caso 1:1 vs. analítico: peor error relativo = {test_analitico_1a1():.2e}  OK")
    print(f"3) κ_D solo H+ a {T} K = {test_kappa_solo_H():.3f} S/m  (≈ 17)  OK")
    fs = test_f_tiende_a_1()
    print("4) f con c_V = 1000 → 1e-6 mol/m3: " + ", ".join(f"{f:.6f}" for f in fs) + "  OK")
    print("Todas las pruebas pasaron.")
