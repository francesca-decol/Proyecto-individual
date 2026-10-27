import numpy as np
from rfbmodel1D.parameters.params import Vars

def water_transport_residual(self, t, u, du, i_io, lam_eq_left, lam_eq_right): 
    """
    Standalone membrane water transport residual (lambda only).

    Matches membrane_residual (lambda block) in dae_residual_full_cell.py
    exactly, with i_io as an explicit parameter so the function can be used
    for water-only fitting/debugging without a running full-cell DAE.

    State vector layout: u = lambda at cell centres, shape (n_x,)
        
    Returns:
        res  : residual vector, shape (n_x,)
    """
    N             = self.mesh.membrane_mesh.n_x
    dx_m          = self._dx_m
    dx_face_m     = self._dx_face_m
    lam           = u
    du_lam        = du
    F             = self.F
    c_f           = self.membrane.c_f
    res           = np.zeros(N)
    diffusive_model = self.membrane.diffusivity_model
    # -- Anode conditions --------------------------------------------------
    s = 0.1201

    # -- Water transport properties ----------------------------------------
    k_ca_lw     = self.membrane.k_ca_lw 
    k_sorp_vw   = self.membrane.k_sorp_vw
    k_desorp_vw = self.membrane.k_desorp_vw
    k_sorp_lw   = self.membrane.k_sorp_lw
    k_desorp_lw = self.membrane.k_desorp_lw

    # -- Interior nodes ----------------------------------------------------
    if i_io > 0:
        lam_e   = lam[1:-1] 
        lam_w   = lam[:-2] 
    else: 
        lam_e = lam[2:]
        lam_w = lam[1:-1]
    D_e     = self.membrane_water_diffusivity((lam[1:-1] + lam[2:])/2, diffusive_model)
    D_w     = self.membrane_water_diffusivity((lam[:-2] + lam[1:-1])/2, diffusive_model)
    zeta_e  = self.EOD_coefficient(lam_e)
    zeta_w  = self.EOD_coefficient(lam_w)
    N_e     = -D_e * (lam[2:]   - lam[1:-1]) / dx_face_m[1:]  + zeta_e * i_io / (F * c_f)
    N_w     = -D_w * (lam[1:-1] - lam[:-2])  / dx_face_m[:-1] + zeta_w * i_io / (F * c_f)
    res[1:-1] = du_lam[1:-1] * dx_m[1:-1] + N_e - N_w

    # -- Boundary conditions -----------------------------------------------
    V_w         = self.membrane.V_w                          # Molar volume of water (m³/mol)
    V_m         = self.membrane.V_m                          # Molar volume of dry membrane (m³/mol)
    s_im        = self.anode.s_im                            # Immobile saturation
    beta_sr     = 100000                                     # Smoothing parameter for saturation correction
    s_r         = np.logaddexp(0.0, beta_sr * (s - s_im)) / (beta_sr * (1 - s_im))
    f_v_an      = lam[-1] * V_w / (lam[-1]*V_w + V_m)        # Volume fraction of water in membrane at right interface
    f_v_ca      = lam[0] * V_w / (lam[0]*V_w + V_m)          # Volume fraction of water in membrane at left interface
    k_ca_lw     = self.membrane.k_ca_lw * f_v_ca             # Corrected by volume fraction (m/s)
    k_sorp_vw   = self.membrane.k_sorp_vw * f_v_an * (1-s)   # Sorption rate for vapor water (m/s)
    k_desorp_vw = self.membrane.k_desorp_vw * f_v_an * (1-s) # Desorption rate for vapor water (m/s)
    k_sorp_lw   = self.membrane.k_sorp_lw * f_v_an * s_r     # Sorption rate for liquid water (m/s)
    k_desorp_lw = self.membrane.k_desorp_lw * f_v_an * s_r   # Desorption rate for liquid water (m/s)

    # LEFT boundary, cathode side (node 0)
    D_e_BC    = self.membrane_water_diffusivity((lam[0] + lam[1]) / 2, diffusive_model)
    if i_io > 0:
        zeta_e_BC = self.EOD_coefficient(lam[0])
    else: 
        zeta_e_BC = self.EOD_coefficient(lam[1])
    N_e       = -D_e_BC * (lam[1] - lam[0]) / dx_face_m[0] + zeta_e_BC * i_io / (F * c_f)
    N_w       = k_ca_lw * (lam_eq_left - lam[0])
    res[0]    = du_lam[0] * dx_m[0] + N_e - N_w

    # RIGHT boundary (node N-1)
    D_w_BC    = self.membrane_water_diffusivity((lam[-1] + lam[-2]) / 2, diffusive_model)
    if i_io > 0:
        zeta_w_BC = self.EOD_coefficient(lam[-1])
    else:
        zeta_w_BC = self.EOD_coefficient(lam[-2])
    N_w       = -D_w_BC * (lam[-1] - lam[-2]) / dx_face_m[-1] + zeta_w_BC * i_io / (F * c_f)
    if lam_eq_right < lam[-1]:
        # Desorption
        N_e = (k_desorp_vw + k_desorp_lw) * (lam[-1] - lam_eq_right)
    else:
        # Sorption
        N_e = (k_sorp_vw + k_sorp_lw) * (lam[-1] - lam_eq_right)
    res[-1]   = du_lam[-1] * dx_m[-1] + N_e - N_w
    return res
