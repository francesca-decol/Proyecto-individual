import copy
import numpy as np
from rfbmodel1D.parameters.params import SimulationParams, Vars, cathodeparams, agdlparams, membraneparams
from rfbmodel1D.cells.cell import Cell

"""
Build a cell for operation-sensitivity studies (RH at the anode channel,
temperature, flow rate), reusing the same tank/BC setup as simulate_HFR.build_cell.
"""


def build_cell_op(soc, j_appl, T, Q_v, RH=1.0):
    """
    soc    : tank state of charge [-]            (sets c_T_V4, c_T_V5)
    j_appl : applied current density [A/m2]      (galvanostatic, + = charge)
    T      : temperature [°C]
    Q_v    : cathode volumetric flow rate [ml/min]
    RH     : relative humidity at the anode channel (water activity BC) [-]
    """
    sim_params = copy.deepcopy(SimulationParams)          # local copy — thread-safe
    sim_params.CathodeParams.Q_total = Q_v * 1e-6 / 60   # Convert ml/min to m3/s
    sim_params.CellParams.T          = T + 273.15         # Convert °C to K
    cell         = Cell(params=sim_params)
    cell.bc_mode = "galvanostatic"
    cell.modify_BC(Vars.phi_el, "cathode", "left", "galvanostatic", j_appl)
    cell.modify_BC(Vars.a, "anode", "right", "dirichlet", RH)
    V_tank   = 50000                      # [m3] Big volume to keep tank concentrations constant
    c_T_V4   = (1 - soc) * 1000           # [mol/m3]
    c_T_V5   = soc * 1000
    c_T_H    = 4963
    c_T_HSO4 = c_T_H * 3 / 5
    c_0 = np.array([c_T_V4, c_T_V5, c_T_H, c_T_HSO4, V_tank])
    y0, yp0 = cell.initialize_dae_state(c_0, xtol=1e-6)
    return cell, y0, yp0


def build_cell_conf(soc, j_appl, T, Q_v, RH=1.0,
                     cathode_porosity=None, agdl_porosity=None,
                     membrane_thickness=None, cathode_ssa=None,
                     cathode_thickness=None, agdl_thickness=None,
                     cathode_conductivity=None):
    """
    Same as build_cell_op, but additionally allows sweeping geometry /
    structural ("configuration") parameters. Any parameter left as None
    keeps its default value from rfbmodel1D.parameters.params.

    cathode_porosity      : cathode porosity [-]
    agdl_porosity         : AGDL porosity [-]
    membrane_thickness    : membrane thickness [m]
    cathode_ssa           : cathode specific surface area [m2/m3]
    cathode_thickness     : cathode thickness [m]
    agdl_thickness        : AGDL thickness [m]
    cathode_conductivity  : cathode electronic conductivity [S/m]
    """
    if cathode_porosity is not None:
        cathodeparams.porosity = cathode_porosity
    if agdl_porosity is not None:
        agdlparams.porosity = agdl_porosity
    if membrane_thickness is not None:
        membraneparams.thickness = membrane_thickness
    if cathode_ssa is not None:
        cathodeparams.specific_surface_area = cathode_ssa
    if cathode_thickness is not None:
        cathodeparams.thickness = cathode_thickness
    if agdl_thickness is not None:
        agdlparams.thickness = agdl_thickness
    if cathode_conductivity is not None:
        cathodeparams.conductivity = cathode_conductivity

    return build_cell_op(soc, j_appl, T, Q_v, RH=RH)
