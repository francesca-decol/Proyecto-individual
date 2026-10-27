"""Shared membrane diffusion model for Experimental and Validation scripts."""

import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from pathlib import Path
from rfbmodel1D.parameters.params import D_params, k_params, membraneparams
from rfbmodel1D.cells.methods.membrane_water_diffusivity import D_hwang_raw

V_W     = 1.809e-5
V_M     = 5.56e-4
S_IM    = 0.12
LAM_LIQ = 16.0
N_FVM   = 100
c_f     = membraneparams.c_f
PARAMS_FILE = Path(__file__).resolve().parent.parent.parent / "rfbmodel1D" / "parameters" / "params.py"

def D_model(lam, params=D_params, sf=1):
    """
    Same formula as D_hwang_raw in rfbmodel1D/cells/methods/membrane_water_diffusivity.py,
    just without the temperature correction (fit data was measured at 300 K).
    """
    return D_hwang_raw(lam, params) * sf

def update_params_file(var_name, values, params_file=PARAMS_FILE):
    """Rewrite the `<var_name> = [...]` (or np.array([...])) literal in params.py in place."""
    text = params_file.read_text(encoding="utf-8")
    values_str = ", ".join(f"{v:.8e}" for v in values)
    pattern = re.compile(rf"({var_name}\s*=\s*(?:np\.array\()?\[)[^\]]*(\][^\n]*)")
    new_text, n = pattern.subn(lambda m: m.group(1) + values_str + m.group(2), text, count=1)
    if n == 0:
        raise ValueError(f"Could not find `{var_name} = [...]` in {params_file}")
    params_file.write_text(new_text, encoding="utf-8")
    print(f"Updated {var_name} in {params_file}")

def _k_eff(lam, condition, k_params):
    """ Modify absolute mass transfer coefficient, related to water volume fraction f_v"""
    state = condition["state"]
    mode  = condition["mode"]
    k_vw_sorp, k_vw_desorp, k_lw_sorp, k_lw_desorp = k_params
    if state == "liquid":  # Liquid conditions
        s   = 1
        s_r = 1
    elif state == "vapor": # Only vapor conditions
        s   = 0
        s_r = 0
    f_v  = lam * V_W / (lam * V_W + V_M)
    if mode == "sorption":
        k_vw = k_vw_sorp
        k_lw = k_lw_sorp
    elif mode == "desorption":
        k_vw = k_vw_desorp
        k_lw = k_lw_desorp
    return (k_vw * (1.0 - s) + k_lw * s_r) * f_v # type: ignore


def membrane_ode(_t, lam, a_left, a_right, left_condition, right_condition, DX, D_params, k_params, lam_liquid=21):
    """FVM membrane ODE with k_eff interfacial mass-transfer BCs.
    Liquid side is assumed in equilibrium
    """
    lam_eq_left  = return_lambda_eq(a_left)
    lam_eq_right = return_lambda_eq(a_right) 
    lam_face     = (lam[:-1] + lam[1:]) / 2 
    D_face     = D_model(lam_face, params=D_params)
    dlam       = np.empty(len(lam))
    N_e        = -D_face[1:]  * c_f * (lam[2:]   - lam[1:-1]) / DX
    N_w        = -D_face[:-1] * c_f * (lam[1:-1] - lam[:-2])  / DX
    dlam[1:-1] = (N_w - N_e) / DX
    k_left     = _k_eff(lam[0],  left_condition,  k_params)
    k_right    = _k_eff(lam[-1], right_condition, k_params)
    if left_condition["state"]=="vapor":
        N_left_BC  = -k_left * c_f * (lam[0]  - lam_eq_left)
    else:
        D_face_0  = D_model(lam[0], params=D_params)
        N_left_BC  = -D_face_0 * c_f * (lam[0]  - lam_liquid) / (DX/2)
    dlam[0]    = (N_left_BC  - N_w[0])   / DX
    if right_condition["state"]=="vapor":
        N_right_BC = -k_right * c_f * (lam_eq_right - lam[-1])
    else:
        D_face_last = D_model(lam[-1], params=D_params)
        N_right_BC = -D_face_last * c_f * (lam_liquid - lam[-1]) / (DX/2)
    dlam[-1]   = (N_e[-1] - N_right_BC) / DX
    return dlam/c_f

def cost_D(params, k_params, t_fit, x_lists, lam_lists, lam0, t0, a_left, a_right, left_condition, right_condition, DX):
    """ Cost function for diffusivity parameter fitting"""
    X_NODES = (np.arange(N_FVM) + 0.5) * DX
    n_res = sum(len(x) for x in x_lists)
    try:
        sol = solve_ivp(membrane_ode, (t0, t_fit[-1]), lam0,
                        args=(a_left, a_right, left_condition, right_condition, DX, params, k_params),
                        method="BDF", t_eval=t_fit)
    except (ValueError, FloatingPointError):
        return np.full(n_res, 1e6)
    if not sol.success:
        return np.full(n_res, 1e6)
    res = []
    for i, (x_exp, lam_exp) in enumerate(zip(x_lists, lam_lists)):
        lam_sim = np.interp(x_exp, X_NODES, sol.y[:, i])
        res.extend(lam_sim - lam_exp)
    return np.array(res)

def cost_k(params, D_params, t_fit, x_lists, lam_lists, lam0, t0, a_left, a_right, left_condition, right_condition, DX):
    """ Cost function for mass transfer coefficient parameter fitting"""
    X_NODES = (np.arange(N_FVM) + 0.5) * DX
    sol = solve_ivp(membrane_ode, (t0, t_fit[-1]), lam0,
                    args=(a_left, a_right, left_condition, right_condition, DX, D_params, params),
                    method="BDF", t_eval=t_fit)
    if not sol.success:
        return np.full(sum(len(x) for x in x_lists), 1e6)
    res = []
    for i, (x_exp, lam_exp) in enumerate(zip(x_lists, lam_lists)):
        lam_sim = np.interp(x_exp, X_NODES, sol.y[:, i])
        res.extend(lam_sim - lam_exp)
    return np.array(res)

def cost_all(params, t_fit, x_lists, lam_lists, lam0, t0, a_left, a_right, left_condition, right_condition, DX):
    """ Cost function for mass transfer coefficient parameter fitting"""
    D_params, k_params = params[:8], params[8:]
    X_NODES = (np.arange(N_FVM) + 0.5) * DX
    sol = solve_ivp(membrane_ode, (t0, t_fit[-1]), lam0,
                    args=(a_left, a_right, left_condition, right_condition, DX, D_params, k_params),
                    method="BDF", t_eval=t_fit)
    if not sol.success:
        return np.full(sum(len(x) for x in x_lists), 1e6)
    res = []
    for i, (x_exp, lam_exp) in enumerate(zip(x_lists, lam_lists)):
        lam_sim = np.interp(x_exp, X_NODES, sol.y[:, i])
        res.extend(lam_sim - lam_exp)
    return np.array(res)

def return_lambda_eq(a, T = None):
    """
    Calculate equilibrium water content from water activity.
    Water activity includes saturation: a = RH + 2s, where RH is relative
    humidity and s is liquid saturation. The function provides continuous
    transition from vapor (a<1) through mixed phase (1≤a≤3) to fully liquid (a>3).
    Args:
        a: Water activity (dimensionless, typically 0 to 3+)
           - a < 1: vapor phase only (sorption isotherm)
           - 1 ≤ a ≤ 3: vapor + liquid saturation (linear transition)
           - a > 3: fully liquid
    Returns:
        lambda_eq: Equilibrium water content (dimensionless)
    Reference:
        Equation obtained from (1) Ye, M.; Pino-Muñoz, C. A.; Cannon, C. G.; Sun, Y.; Kucernak, A.; Klusener, P. A. A.; Brandon, N. 
        Understanding H2-X Redox Flow Battery Performance Through Modelling and Design of Experiments.
        SSRN 2025. https://doi.org/10.2139/ssrn.5126397.
    """
    if T == None:
        T = 298
    if a < 1:
        # Vapor phase: polynomial sorption isotherm
        lambda_298 = lambda_298K(a)
        lambda_363 = lambda_363K(a)
        # Linear interpolation between 298K and 363K
        lambda_eq = lambda_298 + (lambda_363 - lambda_298) * (T - 298) / (363 - 298)
    elif a <= 3:
        # Mixed phase: linear interpolation from saturated vapor to liquid
        lambda_298 = (86.7  - 181.3  + 145.7  - 49.1  + 10.4 + 1.4)
        lambda_363 = (215.0 - 480.9  + 393.5  - 135.9 + 24.0 + 0.8)
        lambda_1 = lambda_298 + (lambda_363 - lambda_298) * (T - 298) / (363 - 298)
        # Starting from a=1
        lambda_eq = lambda_1 + 4*(a - 1)
    else:
        # Fully liquid phase: maximum water content
        lambda_eq = 22 
    return lambda_eq
def lambda_298K(a):
    """Calculate lambda at 298K for 0 <= a <= 1"""
    return (86.7 * a**5 - 181.3 * a**4 + 145.7 * a**3 - 49.1 * a**2 + 10.4 * a + 1.4)
def lambda_363K(a):
    """Calculate lambda at 363K for 0 <= a <= 1"""
    return (215.0 * a**5 - 480.9 * a**4 + 393.5 * a**3 - 135.9 * a**2 + 24.0 * a + 0.8)
