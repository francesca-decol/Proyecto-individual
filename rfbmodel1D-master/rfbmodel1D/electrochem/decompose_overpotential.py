from .butler_volmer import cathode_surface_concentrations
import numpy as np

def decompose_overpotential(self, y):
    """
    Decompose the cathode overpotential at a converged DAE state ``y[time]`` into
    its physical contributions:
    - eta_total  = Phi_el - Phi_io - OCP(c_bulk)     (as solved)
    - eta_diff   = OCP(c_surface) - OCP(c_bulk)      (surface-vs-bulk film shift, Psi3/Psi4)
    - eta_act    = eta_total - eta_diff              (remaining kinetic overpotential)
    Returns a dict of per-node arrays (length Nc), except OCP_tank which is scalar.
    """
    state  = self.split_state(y)
    cath   = state["cathode"]
    y_tank = state["tank"]

    c_V4, c_V5, c_H = cath[:, 0], cath[:, 1], cath[:, 2]
    Phi_el, Phi_io  = cath[:, 4], cath[:, 5]

    OCP_bulk = self.compute_OCP([c_V4, c_V5, c_H], "cathode")
    OCP_tank = self.compute_OCP([y_tank[0], y_tank[1], y_tank[2]], "cathode")

    eta_total = Phi_el - Phi_io - OCP_bulk

    c_s_V4, c_s_V5, c_V4_bulk, c_V5_bulk, Psi3, Psi4, beta_0 = \
        cathode_surface_concentrations(self, eta_total, c_V4, c_V5, c_H)

    OCP_surf = self.compute_OCP([c_s_V4, c_s_V5, c_H], "cathode")

    eta_diff   = OCP_surf - OCP_bulk

    R, T, F = self.R, self.T, self.F

    
    eta_act = eta_total - eta_diff

    return {
        "eta_total":  eta_total,
        "eta_act":    eta_act,
        "eta_diff":   eta_diff,
        "OCP_bulk":   OCP_bulk,
        "OCP_surf":   OCP_surf,
        "OCP_tank":   OCP_tank,
        "c_s_V4":     c_s_V4,
        "c_s_V5":     c_s_V5,
        "Psi3":       Psi3,
        "Psi4":       Psi4,
        "beta_0":     beta_0,
    }
