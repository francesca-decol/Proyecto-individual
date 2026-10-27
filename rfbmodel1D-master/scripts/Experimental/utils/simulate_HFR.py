"""
Shared setup helpers for the HVRFB DAE model, used by the fitting/plotting
scripts and notebooks in this repo: build_cell, build_atol, soc_cutoff_event.
"""
import copy
import numpy as np
from rfbmodel1D.parameters.params import SimulationParams, Vars
from rfbmodel1D.cells.cell import Cell

# ------------------------------------------------------------------------------
# Build absolute tolerances
# ------------------------------------------------------------------------------
def build_atol(cell):
    Nc = cell.mesh.cathode_mesh.n_x
    Nm = cell.mesh.membrane_mesh.n_x
    conc           = 1e-1
    potential      = 1e-3
    lambda_        = 1e-2
    overpotential  = 1e-4
    saturation     = 1e-3
    volume         = 1e-6
    gas_conc       = 1e-3
    atol_tank      = np.array([conc, conc, conc, conc, volume],                  dtype=float)
    atol_cath_node = np.array([conc, conc, conc, conc, potential, potential],     dtype=float)
    atol_mem_node  = np.array([lambda_, potential],                               dtype=float)
    atol_anode     = np.array([gas_conc, gas_conc, saturation,
                               overpotential, gas_conc, gas_conc, saturation],    dtype=float)
    atol_coupling = np.array([conc], dtype=float)  # H+ concentration at cathode-membrane interface

    return np.concatenate([
        atol_tank,
        np.tile(atol_cath_node, Nc),
        np.tile(atol_mem_node,  Nm),
        atol_anode,
        atol_coupling
    ])
# ------------------------------------------------------------------------------
# Terminal event: stop once cathode tank SOC drops below SOC_MIN
# ------------------------------------------------------------------------------
def soc_cutoff_event(t, y, yp):
    """Terminal event: fires (zero crossing, decreasing) when tank SOC < SOC_MIN."""
    SOC_MIN = 0.02  # stop at 2 % SOC — below this concentration starvation dominates
    c_V4    = y[0]   # tank V4  [mol/m³]
    c_V5    = y[1]   # tank V5  [mol/m³]
    soc     = c_V5 / (c_V4 + c_V5 + 1e-10)
    return soc - SOC_MIN
soc_cutoff_event.terminal  = True # type: ignore
soc_cutoff_event.direction = -1   # only trigger while SOC is decreasing # type: ignore
# ------------------------------------------------------------------------------
# Setup helpers
# ------------------------------------------------------------------------------
def build_cell(soc, j_appl, T, Q_v):
    sim_params = copy.deepcopy(SimulationParams)          # local copy — thread-safe
    sim_params.CathodeParams.Q_total = Q_v * 1e-6 / 60   # Convert ml/min to m3/s
    sim_params.CellParams.T          = T + 273.15         # Convert °C to K
    cell         = Cell(params=sim_params)
    cell.bc_mode = "galvanostatic"
    cell.modify_BC(Vars.phi_el, "cathode", "left", "galvanostatic", j_appl)
    V_tank   = 50000                      # [m3] Big volume to keep tank concentrations constant
    c_T_V4   = (1 - soc) * 1000           # [mol/m3]
    c_T_V5   = soc * 1000
    c_T_H    = 4963
    c_T_HSO4 = c_T_H * 3 / 5
    c_0 = np.array([c_T_V4, c_T_V5, c_T_H, c_T_HSO4, V_tank])
    y0, yp0 = cell.initialize_dae_state(c_0, xtol=1e-6)
    return cell, y0, yp0