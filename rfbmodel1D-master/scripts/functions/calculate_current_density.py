import numpy as np
from rfbmodel1D.parameters.params import Vars


def calculate_current_density(cell, y):
    """
    Return face-based current densities (Nc+1 and Nm+1 values) so that
    boundary faces use the BC-defined flux instead of an extrapolated node gradient.
    """
    Nc = cell.mesh.cathode_mesh.n_x
    Nm = cell.mesh.membrane_mesh.n_x
    segs        = cell.split_state(y)
    y_ca_final  = segs["cathode"]
    y_mem_final = segs["membrane"]

    phi_el_ca  = y_ca_final[:, 4]
    phi_io_ca  = y_ca_final[:, 5]
    phi_io_mem = y_mem_final[:, 1]

    # -- geometry -------------------------------------------------------------
    dx_ca = cell.mesh.cathode_mesh.dx * cell.cathode.thickness   # Nc
    dx_m  = cell.mesh.membrane_mesh.dx * cell.membrane.thickness # Nm

    edges_ca = np.concatenate([[0.0], np.cumsum(dx_ca)])  # Nc+1 face positions
    edges_m  = np.concatenate([[0.0], np.cumsum(dx_m)])   # Nm+1

    x_center_ca = 0.5 * (edges_ca[:-1] + edges_ca[1:])   # Nc cell centres
    x_center_m  = 0.5 * (edges_m[:-1]  + edges_m[1:])    # Nm cell centres

    # Distance between adjacent cell centres (gradient denominator for interior faces)
    dx_face_ca = x_center_ca[1:] - x_center_ca[:-1]      # Nc-1
    dx_face_m  = x_center_m[1:]  - x_center_m[:-1]       # Nm-1

    # -- transport coefficients ------------------------------------------------
    sigma_ca = cell.cathode.eff_conductivity               # scalar
    kappa_ca = cell.cathode.kappa_io                       # Nc
    kappa_m  = cell.membrane.kappa_io                      # Nm

    # Harmonic-mean face conductivities for interior faces
    kappa_faces_ca = ((dx_ca[1:] + dx_ca[:-1]) /
                      (dx_ca[1:] / kappa_ca[:-1] + dx_ca[:-1] / kappa_ca[1:]))   # Nc-1

    kappa_faces_m  = (2.0 * kappa_m[:-1] * kappa_m[1:] /
                      (kappa_m[:-1] + kappa_m[1:] + 1e-30))                       # Nm-1

    # -- boundary currents from BCs --------------------------------------------
    bc_phi_el_ca_L = cell.bc[Vars.phi_el]["cathode"]["left"]
    bc_phi_el_ca_R = cell.bc[Vars.phi_el]["cathode"]["right"]
    bc_phi_io_ca_L = cell.bc[Vars.phi_io]["cathode"]["left"]
    bc_phi_io_ca_R = cell.bc[Vars.phi_io]["cathode"]["right"]
    bc_phi_io_m_L  = cell.bc[Vars.phi_io]["membrane"]["left"]
    bc_phi_io_m_R  = cell.bc[Vars.phi_io]["membrane"]["right"]

    i_el_bc_L    = bc_to_current(phi_el_ca,  bc_phi_el_ca_L, sigma_ca,     dx_ca, "left")
    i_el_bc_R    = bc_to_current(phi_el_ca,  bc_phi_el_ca_R, sigma_ca,     dx_ca, "right")
    i_io_ca_bc_L = bc_to_current(phi_io_ca,  bc_phi_io_ca_L, kappa_ca[0], dx_ca, "left")
    i_io_ca_bc_R = bc_to_current(phi_io_ca,  bc_phi_io_ca_R, kappa_ca[-1],dx_ca, "right")

    if cell.bc_mode == "galvanostatic":
        i_io_m_bc_L = -kappa_m[0]  * bc_phi_io_m_L.value
        i_io_m_bc_R = -kappa_m[-1] * (bc_phi_io_m_R.value - phi_io_mem[-1]) / (dx_m[-1] / 2)
    elif cell.bc_mode == "potentiostatic":
        i_io_m_bc_L = -kappa_m[0]  * (phi_io_mem[0]  - bc_phi_io_m_L.value) / (dx_m[0]  / 2)
        i_io_m_bc_R = -kappa_m[-1] * (bc_phi_io_m_R.value - phi_io_mem[-1]) / (dx_m[-1] / 2)
    else:
        i_io_m_bc_L = i_io_m_bc_R = np.nan

    # -- face-based current arrays (Nc+1, Nc+1, Nm+1) -------------------------
    i_el_ca = np.empty(Nc + 1)
    i_el_ca[0]    = i_el_bc_L
    i_el_ca[1:Nc] = -sigma_ca * (phi_el_ca[1:] - phi_el_ca[:-1]) / dx_face_ca
    i_el_ca[Nc]   = i_el_bc_R

    i_io_ca = np.empty(Nc + 1)
    i_io_ca[0]    = i_io_ca_bc_L
    i_io_ca[1:Nc] = -kappa_faces_ca * (phi_io_ca[1:] - phi_io_ca[:-1]) / dx_face_ca
    i_io_ca[Nc]   = i_io_ca_bc_R

    i_io_mem = np.empty(Nm + 1)
    i_io_mem[0]    = i_io_m_bc_L
    i_io_mem[1:Nm] = -kappa_faces_m * (phi_io_mem[1:] - phi_io_mem[:-1]) / dx_face_m
    i_io_mem[Nm]   = i_io_m_bc_R

    # -- face coordinates for plotting -----------------------------------------
    L_ca  = cell.cathode.thickness
    x_ca  = edges_ca              # 0 … L_ca       (Nc+1 points)
    x_mem = L_ca + edges_m        # L_ca … L_ca+Lm (Nm+1 points)

    return {
        "x_ca":          x_ca,
        "x_mem":         x_mem,
        "phi_el_ca":     phi_el_ca,
        "phi_io_ca":     phi_io_ca,
        "phi_io_mem":    phi_io_mem,
        "i_el_ca":       i_el_ca,
        "i_io_ca":       i_io_ca,
        "i_io_mem":      i_io_mem,
        "i_el_bc_L":     i_el_bc_L,
        "i_el_bc_R":     i_el_bc_R,
        "i_io_ca_bc_L":  i_io_ca_bc_L,
        "i_io_ca_bc_R":  i_io_ca_bc_R,
    }


def bc_to_current(phi, bc, coeff, dx, side):
    """Convert a potential BC to a face current density: i = -coeff * dΦ/dx."""
    if bc.type == "neumann":
        dphi_dx = bc.value
    elif bc.type == "galvanostatic":
        dphi_dx = -bc.value / coeff
    elif bc.type == "dirichlet":
        if side == "left":
            dphi_dx = (phi[0] - bc.value) / (dx[0] / 2)
        else:
            dphi_dx = (bc.value - phi[-1]) / (dx[-1] / 2)
    else:
        return np.nan
    return -coeff * dphi_dx