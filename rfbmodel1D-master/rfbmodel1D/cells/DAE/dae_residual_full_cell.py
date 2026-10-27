import numpy as np
from scipy.optimize import fsolve
from rfbmodel1D.parameters.params import SimulationParams, Vars

def dae_residual_full_cell(self, t, y, yp):
    """
    Full-cell DAE residual.

    State vector layout:
      [tank(5) | cathode(Nc×6) | membrane(Nm×2) | anode(7) | coupling(n_cp)]
    Coupling block: [c_H_ca_BC]
    """
    # -- Block sizes -------------------------------------------------------
    Nc    = self.mesh.cathode_mesh.n_x
    Nm    = self.mesh.membrane_mesh.n_x
    nv_t  = self.layout.n_vars_tank        # 5  : V4, V5, H+, HSO4, V
    nv_c  = self.layout.n_vars_cathode     # 6  : V4, V5, H+, HSO4, Phi_el, Phi_io
    nv_m  = self.layout.n_vars_membrane    # 2  : lam, Phi_io
    nv_a  = self.layout.n_vars_anode       # 7  : ACL(4) + AGDL(3)
    nv_cp = self.layout.n_vars_coupling    # 1  : c_H_ca_BC, 

    nt, nc, nm, na, ncp = nv_t, Nc*nv_c, Nm*nv_m, nv_a, nv_cp

    # -- Offsets (start index of each block) -------------------------------
    o_t  = 0
    o_c  = o_t  + nt
    o_m  = o_c  + nc
    o_a  = o_m  + nm
    o_cp = o_a  + na

    # -- Unpack state and derivative vectors -------------------------------
    y_tank,     yp_tank     = y[o_t :o_c ], yp[o_t :o_c ]
    y_cathode,  yp_cathode  = y[o_c :o_m ], yp[o_c :o_m ]
    y_membrane, yp_membrane = y[o_m :o_a ], yp[o_m :o_a ]
    y_acl,      yp_acl      = y[o_a :o_cp], yp[o_a :o_cp]
    y_coupling              = y[o_cp:o_cp + ncp]

    # -- Convenience interface views ---------------------------------------
    lam            = y_membrane.reshape(Nm, nv_m)[:, 0]
    Phi_io_membrane = y_membrane.reshape(Nm, nv_m)[:, 1]
    y_cathode_N0   = y_cathode[:nv_c]    # left  face -> tank interface
    y_cathode_Nend = y_cathode[-nv_c:]   # right face -> membrane interface
    y_membrane_N0  = y_membrane[:nv_m]   # left  face -> cathode interface
    y_membrane_Nend= y_membrane[-nv_m:]  # right face -> anode interface
    y_c_H_BC       = y_coupling[0]       # H+ concentration BC for cathode interface
    y_cathode_V4   = y_cathode.reshape(Nc, nv_c)[:, 0] # V4+ concentration profile in cathode for coupling source term
    y_cathode_V5   = y_cathode.reshape(Nc, nv_c)[:, 1] # V5+ concentration profile in cathode for coupling source term

    # -- Update conductivities (must precede residuals) --------------------
    update_vanadium(self, y_cathode_Nend)
    self.update_kappa_io_cathode(y_cathode)
    self.update_kappa_io_m(lam, Phi_io_membrane)

    # -- Update interface BCs (order matters for correct coupling) ---------
    N_tank = update_BC_tank_cathode    (self, y_tank,        y_cathode_N0  )
    update_BC_cathode_membrane         (self, y_cathode_Nend, y_membrane_N0, y_c_H_BC, y_cathode_V4, y_cathode_V5)
    update_BC_membrane_anode           (self, y_membrane_Nend, y_acl[:nv_a] )


    # -- Domain residuals --------------------------------------------------
    res_tank     = tank_residual    (self, t, y_tank,     yp_tank,    N_tank, y_cathode)
    res_cathode  = cathode_residual (self, t, y_cathode,  yp_cathode        )
    res_membrane = membrane_residual(self, t, y_membrane, yp_membrane       )
    res_anode    = anode_residual   (self, t, y_acl,      yp_acl            )
    res_coupling = coupling_residual(self,    y_cathode_Nend, y_membrane_N0, y_coupling)

    # -- Scale residuals ---------------------------------------------------
    # Divide each equation block by a characteristic magnitude so that
    # scaled residuals are O(1) near the solution.
    #
    # Derivation (100 A/m² operating point, see dae_improvements.txt):
    #   Tank species   [mol/s]:      A_ip * u_ip * c_ref
    #   Cathode species [mol/m²/s]:  u_ip * c_ref/H * dx_mean  (dx in res)
    #   Cathode Phi    [A/m²]:       (i_ref/L_ca) * dx_mean
    #   Membrane lam   [m/s]:        k_ca * delta_lam
    #   Membrane Phi   [A/m²]:       i_ref (ionic current)
    #   ACL/AGDL vap/sat [mol/m²/s]: R_w ~ 0.1
    #   ACL/AGDL H2    [mol/m²/s]:   i_ref / (2F)
    #   ACL eta        [A/m²]:       i_ref
    #   Coupling       [mol/m²/s]:   i_ref / F
    i_ref   = 100.0
    c_V_ref = 500.0
    c_H_ref = 3000.0
    A_ip    = self.cathode.width * self.cathode.thickness
    u_sc    = max(self.cathode.u_ip, 1e-5)   # floor avoids zero at startup
    H_ca    = self.cathode.height
    L_ca    = self.cathode.thickness
    dx_mean = np.mean(self._dx_ca)
    # Tank [mol/s]
    s_T = np.array([A_ip*u_sc*c_V_ref, A_ip*u_sc*c_V_ref,
                    A_ip*u_sc*c_H_ref, A_ip*u_sc*c_H_ref, 1e-6])
    
    # Cathode [mol/m²/s for species, A/m² for potentials]
    s_ca_V  = u_sc * c_V_ref / H_ca * dx_mean
    s_ca_H  = u_sc * c_H_ref / H_ca * dx_mean
    s_ca_ph = i_ref / L_ca * dx_mean
    scale_cathode  = np.tile([s_ca_V, s_ca_V, s_ca_H, s_ca_H, s_ca_ph, s_ca_ph], Nc)
    # Membrane [m/s for lambda, A/m² for Phi_io]
    s_m_lam = self.membrane.k_ca_lw * 5.0          # k_ca * typical delta_lam
    s_m_phi = i_ref
    scale_membrane = np.tile([s_m_lam, s_m_phi], Nm)
    # Anode [mol/m²/s for mass, A/m² for eta]
    s_H2 = i_ref / (2.0 * self.F)
    scale_anode = np.array([0.1, s_H2, 0.1, i_ref, 0.1, s_H2, 0.1])
    # Coupling [mol/m²/s]
    scale_coupling = np.array([i_ref / self.F])

    res_tank     = res_tank     / s_T
    res_cathode  = res_cathode  / scale_cathode
    res_membrane = res_membrane / scale_membrane
    res_anode    = res_anode    / scale_anode
    res_coupling = res_coupling / scale_coupling

    # -- Assemble ----------------------------------------------------------
    residual = np.concatenate([
        res_tank, res_cathode, res_membrane, res_anode, res_coupling])

    # -- Debug counter -----------------------------------------------------
    if not hasattr(self, '_res_counter'):
        self._res_counter = 0
    self._res_counter += 1
    if self._res_counter % 20000 == 0:
        print(f"Call {self._res_counter}: t={t:.6f}, max|res|={np.max(np.abs(residual)):.6e}")

    return residual

def cathode_residual(self, t, y_cathode, yp_cathode):
    N       = self.mesh.cathode_mesh.n_x       # Number of nodes
    dx      = self._dx_ca     # Grid spacing in computational domain. length of each cell
    dx_face = self._dx_face_ca 
    n_vars  = self.layout.n_vars_cathode  # Number of variables per node in cathode
    res_view = np.empty((N, n_vars))
    view = y_cathode.reshape((N, n_vars))
    du_view = yp_cathode.reshape((N, n_vars))
    # Unpack variables [V4, V5, H+, HSO4, Phi_el, Phi_io]
    c_V4 = view[:, 0]
    c_V5 = view[:, 1]
    c_H  = view[:, 2]
    c_HSO4 = view[:, 3]
    Phi_el = view[:, 4]
    Phi_io = view[:, 5]
    du_c_V4 = du_view[:, 0]
    du_c_V5 = du_view[:, 1]
    du_c_H = du_view[:, 2]
    du_c_HSO4 = du_view[:, 3]
    # Cathode properties
    bc_mode  = self.bc_mode # "galvanostatic" or "potentiostatic"
    F        = self.F
    porosity = self.cathode.porosity
    v        = self.cathode.u_tp # Through-plane velocity (into electrode thickness)
    bc_phi_left = self.bc[Vars.phi_io]["cathode"]["left"]
    bc_phi_right = self.bc[Vars.phi_io]["cathode"]["right"]
    # Overpotential and kinetics
    c_eta = np.concatenate([c_V4, c_V5, c_H])
    self.compute_overpotential(Phi_el, Phi_io, c_eta, electrode="cathode")
    eta = self.cathode.eta
    BV = self.butler_volmer(eta, c_eta, "cathode")  # Negative for discharge
    j_n = BV  # A/m3
    def species_transport(c, du_c, D_eff, z, mu, bc_left, bc_right, source_term=0.0):
        b = np.zeros(N)
        Pe = (v * dx) / D_eff
        schemes = np.where(Pe < 2, 0, np.where(v > 0, 1, 2)) # 0: central, 1: upwind_pos (v>0), 2: upwind_neg (v<0)
        # Neumann BC flux 
        b = self.get_boundary_flux_left(bc_left, b)
        b = self.get_boundary_flux_right(bc_right, b)
        # Diffusion  ---------
        N_e_diff = -D_eff * (c[2:] - c[1:-1]) / dx_face[1:]
        N_w_diff = -D_eff * (c[1:-1] - c[:-2]) / dx_face[:-1]
        N_left_BC = self.apply_diffusion_left_BC_vec(c, bc_left, D_eff, dx)
        N_right_BC = self.apply_diffusion_right_BC_vec(c, bc_right, D_eff, dx)
        N_flux = np.zeros(N)
        N_flux[0]    = N_w_diff[0] - N_left_BC
        N_flux[1:-1] = N_e_diff - N_w_diff
        N_flux[-1]   = N_right_BC - N_e_diff[-1]
        # Convection ---------
        # East face-centered values (distance-weighted for non-uniform grid)
        dx_left_e = dx[1:-1]
        dx_right_e = dx[2:]
        c_e_central = (dx_right_e * c[1:-1] + dx_left_e * c[2:]) / (dx_left_e + dx_right_e)
        c_e_upwind_pos = c[1:-1]                   # upwind (v>0) at east face
        c_e_upwind_neg = c[2:]                     # upwind (v<0) at east face
        c_e_all = np.stack([c_e_central, c_e_upwind_pos, c_e_upwind_neg], axis=0)
        N_e_conv = v * c_e_all[schemes[1:-1], np.arange(len(schemes[1:-1]))]
        # West face-centered values (distance-weighted for non-uniform grid)
        dx_left_w = dx[:-2]
        dx_right_w = dx[1:-1]
        c_w_central = (dx_right_w * c[:-2] + dx_left_w * c[1:-1]) / (dx_left_w + dx_right_w)
        c_w_upwind_pos = c[:-2]
        c_w_upwind_neg = c[1:-1]
        c_w_all = np.stack([c_w_central, c_w_upwind_pos, c_w_upwind_neg], axis=0)
        N_w_conv = v * c_w_all[schemes[:-2], np.arange(len(schemes[:-2]))]
        N_flux[0]     += N_w_conv[0] - v * (bc_left.value)    #  West convection at left boundary
        N_flux[1:-1]  += N_e_conv - N_w_conv
        N_flux[-1]    += 0 - N_e_conv[-1]  # No east convection at right boundary
        # Migration ---------
        lower, diag, upper = self.return_migration_elements(c, dx_face)
        mig = self.apply_migration_operator(lower, diag, upper, Phi_io)
        mig, b = self.apply_migration_left_electrode_BC_vec(
            mig, b, bc_left, bc_phi_left, c, Phi_io, dx, dx_face)
        mig, b = self.apply_migration_right_electrode_BC_vec(
            mig, b, bc_right, bc_phi_right, c, Phi_io, dx, dx_face)
        # In plane convection source term
        H             = self.cathode.height # Cathode height
        u_ip          = self.cathode.u_ip   # Cathode in plane velocity (Superficial velocity)
        c_T           = bc_left.value       # Cathode tank concentration
        source_term_convection_ip = (u_ip) * (c_T - c)/H  # Source term convection in plane [mol/m3/s]. Uses superficial velocity.
        return (
            porosity * du_c * dx
            + N_flux
            - z * mu * F  * mig
            - b
            - source_term * dx
            - source_term_convection_ip * dx
        )
    # Acid dissociation source coupling for H+/HSO4-
    k_d = self.k_d
    Beta = self.Beta
    b_dis = -k_d * ((c_H - c_HSO4) / (c_H + c_HSO4 + 1e-12) - Beta)
    # Species transport equations
    res_view[:, 0] = species_transport(
        c_V4,
        du_c_V4,
        self.cathode.D_V4_eff,
        +2,
        self.cathode.mu_V4,
        self.bc[Vars.c_v4]["cathode"]["left"],
        self.bc[Vars.c_v4]["cathode"]["right"],
        source_term=-BV / F,
    )
    res_view[:, 1] = species_transport(
        c_V5,
        du_c_V5,
        self.cathode.D_V5_eff,
        +1,
        self.cathode.mu_V5,
        self.bc[Vars.c_v5]["cathode"]["left"],
        self.bc[Vars.c_v5]["cathode"]["right"],
        source_term=BV / F,)
    res_view[:, 2] = species_transport(
        c_H,
        du_c_H,
        self.cathode.D_H_plus_eff,
        +1,
        self.cathode.mu_H_plus,
        self.bc[Vars.c_h]["cathode"]["left"],
        self.bc[Vars.c_h]["cathode"]["right"],
        source_term=2 * BV / F + b_dis,)
    res_view[:, 3] = species_transport(
        c_HSO4,
        du_c_HSO4,
        self.cathode.D_HSO4_minus_eff,
        -1,
        self.cathode.mu_HSO4_minus,
        self.bc[Vars.c_hso4]["cathode"]["left"],
        self.bc[Vars.c_hso4]["cathode"]["right"],
        source_term=-b_dis,)
    # 
    # Electronic potential
    #
    Div = self._Div_ca           # Divergence of matrix
    sigma_el = self.cathode.eff_conductivity # Effective anode electronic conductivity
    b_BC     = np.zeros(N)
    bc_left_phi_el  = self.bc[Vars.phi_el]["cathode"]["left"]
    bc_right_phi_el = self.bc[Vars.phi_el]["cathode"]["right"]
    Div, b_BC = self.apply_electronic_left_BC(Div, b_BC, bc_left_phi_el, sigma_el, dx, dx_face)
    Div, b_BC = self.apply_electronic_right_BC(Div, b_BC, bc_right_phi_el, sigma_el, dx, dx_face)
    # Equation in the form
    # - sigma / dx * Div.Phi_el - b_BC + j_n dx = 0
    res_view[:,4] = -sigma_el * Div @ Phi_el  - b_BC  + j_n * dx
    # 
    # Ionic potential
    #
    kappa = self.cathode.kappa_io # Ionic conductivity
    bc_left_phi_io  = self.bc[Vars.phi_io]["cathode"]["left"]
    bc_right_phi_io = self.bc[Vars.phi_io]["cathode"]["right"]
    # Compute face-averaged conductivity 
    dx_left     = dx[:-1]      # cell to the left of face
    dx_right    = dx[1:]       # cell to the right of face
    kappa_left  = kappa[:-1]   # kappa at left cell
    kappa_right = kappa[1:]    # kappa at right cell
    kappa_faces = (dx_right*kappa_left + dx_left*kappa_right) / (dx_right + dx_left)  # distance-weighted average to face centers for non-uniform grid
    # Compute gradient at faces
    grad_faces = (Phi_io[1:] - Phi_io[:-1]) / dx_face 
    flux_faces = -kappa_faces * grad_faces
    # Boundary fluxes
    flux_left_BC  = self.apply_ionic_left_BC_vec(Phi_io, bc_left_phi_io, kappa, dx)
    flux_right_BC = self.apply_ionic_right_BC_vec(Phi_io, bc_right_phi_io, kappa, dx)
    # Equation in the form
    # - k / dx * Div.Phi_io - b_BC - j_n dx = 0
    res_view[0, 5]    = (flux_faces[0] - flux_left_BC) - j_n[0] * dx[0] 
    res_view[1:-1, 5] = (flux_faces[1:] - flux_faces[:-1]) - j_n[1:-1] * dx[1:-1] 
    res_view[-1, 5]   = (flux_right_BC - flux_faces[-1]) - j_n[-1] * dx[-1] 
    return res_view.ravel()

def tank_residual(self, t, y_tank, yp_tank, N_tank, y_cathode):
    N       = 1 # Number of nodes
    n_vars  = self.layout.n_vars_tank   # Number of variables per node in tank
    N_ca    = self.mesh.cathode_mesh.n_x
    n_vars_ca = self.layout.n_vars_cathode
    res_view = np.zeros((N, n_vars))  
    view    = y_tank.reshape((N, n_vars)) # [V4, V5, H+, HSO4, V]
    view_ca = y_cathode.reshape((N_ca, n_vars_ca))
    # Unpack variables for tank
    c_V4   = view[:, 0] # V4+ concentration in tank   [mol/m3]
    c_V5   = view[:, 1] # V5+ concentration in tank   [mol/m3]
    c_H    = view[:, 2] # H+ concentration in tank    [mol/m3]
    c_HSO4 = view[:, 3] # HSO4- concentration in tank [mol/m3]
    V_T    = view[:, 4] # Volume of electrolyte       [m3]
    # Unpack variables for cathode concentrations
    dx_ca = self._dx_ca
    c_V4_ca   = np.dot(view_ca[:, 0], dx_ca)/np.sum(dx_ca) # Average V4+ concentration in cathode   [mol/m3]
    c_V5_ca   = np.dot(view_ca[:, 1], dx_ca)/np.sum(dx_ca) # Average V5+ concentration in cathode   [mol/m3]
    c_H_ca    = np.dot(view_ca[:, 2], dx_ca)/np.sum(dx_ca) # Average H+ concentration in cathode    [mol/m3]
    c_HSO4_ca = np.dot(view_ca[:, 3], dx_ca)/np.sum(dx_ca) # Average HSO4- concentration in cathode [mol/m3]
    # Cross sectional area left side (height x width)
    A_ca_tp = self.cathode.height * self.cathode.width # [m2]
    # Cross sectional area for in plane velocity (width x thickness)
    A_ca_ip = self.cathode.width * self.cathode.thickness # [m2]
    u_ip    = self.cathode.u_ip # In-plane velocity for convection source term [m/s]
    # Residual
    res_view[:, 0] = V_T * yp_tank[0] + A_ca_tp * N_tank[0] + A_ca_ip * u_ip * (c_V4 - c_V4_ca)
    res_view[:, 1] = V_T * yp_tank[1] + A_ca_tp * N_tank[1] + A_ca_ip * u_ip * (c_V5 - c_V5_ca)
    res_view[:, 2] = V_T * yp_tank[2] + A_ca_tp * N_tank[2] + A_ca_ip * u_ip * (c_H - c_H_ca)
    res_view[:, 3] = V_T * yp_tank[3] + A_ca_tp * N_tank[3] + A_ca_ip * u_ip * (c_HSO4 - c_HSO4_ca)
    res_view[:, 4] = yp_tank[4] # Constant volume 
    return res_view.flatten()

def membrane_residual(self, t, y_membrane, yp_membrane):
    N        = self.mesh.membrane_mesh.n_x # Number of nodes
    n_vars   = self.layout.n_vars_membrane # Number of variables per node in membrane
    res_view = np.zeros((N, n_vars))   
    view     = y_membrane.reshape((N, n_vars))
    du_view  = yp_membrane.reshape((N, n_vars))
    # Unpack variables [lam, Phi_io]
    lam      = view[:, 0]
    Phi_io   = view[:, 1]
    du_lam   = du_view[:, 0]
    # Membrane properties
    F           = self.F
    thickness_m = self.membrane.thickness
    dx_m        = self._dx_m
    dx_face_m   = self._dx_face_m
    V_m         = self.membrane.V_m
    V_w         = self.membrane.V_w
    c_f         = self.membrane.c_f
    #
    # Water transport
    diffusive_model = self.membrane.diffusivity_model
    # Calculate EOD 
    i_io    = self.i_io
    if i_io>0:
        lam_e = lam[1:-1]
        lam_w = lam[:-2]
    else:
        lam_e = lam[2:] 
        lam_w = lam[1:-1] 
    # Calculate diffusivity at faces 
    D_e = self.membrane_water_diffusivity((lam[1:-1] + lam[2:])/2, diffusive_model)
    D_w = self.membrane_water_diffusivity((lam[:-2] + lam[1:-1])/2, diffusive_model)
    # Calculate EOD coefficients at faces
    zeta_e = self.EOD_coefficient(lam_e)
    zeta_w = self.EOD_coefficient(lam_w)
    # Water fluxes: N = -D * dlambda/dx + zeta * i / F
    N_e = -D_e  * (lam[2:] - lam[1:-1]) / dx_face_m[1:] + zeta_e * i_io / (F*c_f)
    N_w = -D_w  * (lam[1:-1] - lam[:-2]) / dx_face_m[:-1] + zeta_w * i_io / (F*c_f)
    # Residual: dlam/dt * Δx + (N_e - N_w) = 0
    res_view[1:-1, 0] = du_lam[1:-1] * dx_m[1:-1] + N_e - N_w
    ## Boundary conditions
    # Get water activity boundary conditions from adjacent phases
    a_left  = self.bc[Vars.a]["membrane"]["left"].value 
    # Right side 
    a_w_an = self.bc[Vars.a]["membrane"]["right"].value
    s       = self.bc[Vars.s]["membrane"]["right"].value # Saturation at right interface (anode side)
    s_im    = self.anode.s_im # Immobile saturation
    # Convert activity to equilibrium water content at interfaces
    # Cathode
    lam_eq_left  = self.return_lambda_cath_eq(a_left)   # Equilibrium water content
    lam_eq_left  = self.return_lambda_corr(lam_eq_left) # Corrected by vanadium uptake
    # Anode
    lam_eq_right = self.return_lambda_eq(a_w_an) 
    if lam_eq_right < 2.01:
        lam_eq_right = 2.01  # Minimum water content
    # Asymmetric interfacial kinetics (sorption faster than desorption)
    s_im        = self.anode.s_im  # Immobile saturation 
    beta_sr     = 100000            # Smoothing parameter for saturation correction
    s_r         = np.logaddexp(0.0, beta_sr * (s - s_im)) / (beta_sr * (1 - s_im))
    f_v_an      = lam[-1] * V_w / (lam[-1]*V_w + V_m)        # Volume fraction of water in membrane at right interface
    k_sorp_vw   = self.membrane.k_sorp_vw * f_v_an * (1-s)   # Sorption rate for vapor water (m/s)
    k_desorp_vw = self.membrane.k_desorp_vw * f_v_an * (1-s)                              # Desorption rate for vapor water (m/s)
    k_sorp_lw   = self.membrane.k_sorp_lw * f_v_an * s_r     # Sorption rate for liquid water (m/s)
    k_desorp_lw = self.membrane.k_desorp_lw * f_v_an * s_r                              # Desorption rate for liquid water (m/s)
    k_ca        = self.membrane.k_ca_lw                      # Water transport coefficient for cathode (m/s)
    f_v_ca      = lam[0] * V_w / (lam[0]*V_w + V_m)          # Volume fraction of water in membrane at interface
    k_ca_corr   = k_ca * f_v_ca                              # Corrected by volume fraction (m/s)
    # LEFT BOUNDARY (node 0)
    D_e_BC    = self.membrane_water_diffusivity((lam[1]+lam[0])/2, diffusive_model)
    if i_io>0:
        zeta_e_BC = self.EOD_coefficient(lam[0]) 
    else:
        zeta_e_BC = self.EOD_coefficient(lam[1])
    N_e = -D_e_BC * (lam[1] - lam[0]) / dx_face_m[0] + zeta_e_BC * i_io / (F*c_f)
    delta_lam = lam[0] - lam_eq_left
    blend     = 0.5 * (1.0 + np.tanh(delta_lam / 0.1))  
    k_ca_eff  = blend * k_ca_corr * 10 + (1 - blend) * k_ca_corr 
    N_w       = k_ca_eff * (lam_eq_left - lam[0])
    #if lam[0] < lam_eq_left:
        # Sorption: water enters from exterior, N_w > 0
    #    N_w = k_ca_corr * (lam_eq_left - lam[0]) 
    #else:
    #    # Desorption: water leaves to exterior, N_w < 0
    #    N_w = 10 * k_ca_corr * (lam_eq_left - lam[0]) 
    res_view[0,0] = du_lam[0] * dx_m[0] + N_e - N_w
    # RIGHT BOUNDARY (node n_x-1)
    D_w_BC = self.membrane_water_diffusivity((lam[-1]+lam[-2])/2, diffusive_model)
    if i_io>0:
        zeta_w_BC = self.EOD_coefficient(lam[-2])
    else:
        zeta_w_BC = self.EOD_coefficient(lam[-1])
    N_w = -D_w_BC * (lam[-1] - lam[-2]) / dx_face_m[-1] + zeta_w_BC * i_io / (F*c_f)
    delta_lam = lam[-1] - lam_eq_right
    blend     = 0.5 * (1.0 + np.tanh(delta_lam / 0.1))  # 0 -> sorption, 1 -> desorption
    k_vw_eff  = blend * k_desorp_vw + (1 - blend) * k_sorp_vw
    k_lw_eff  = blend * k_desorp_lw + (1 - blend) * k_sorp_lw
    abv_22    = lam[-1] - 22
    blend     = 0.5 * (1.0 + np.tanh(abv_22 / 0.25))   # w = smoothing width
    k_extra   = k_lw_eff * 10 # Extra kinetic factor for liquid water when membrane is superssaturated.
    N_e       = (k_vw_eff + (k_lw_eff + k_extra * blend)) * delta_lam
    res_view[-1,0] = du_lam[-1] * dx_m[-1] + N_e - N_w      
    # 
    # Ionic potential
    #          
    kappa_io = self.membrane.kappa_io # Membrane ionic conductivity [S/m]
    bc_left_phi_io  = self.bc[Vars.phi_io]["membrane"]["left"]
    bc_right_phi_io = self.bc[Vars.phi_io]["membrane"]["right"]
    # Face conductivity (harmonic mean)
    dx_left  = dx_m[:-1]    # cell to the left of face
    dx_right = dx_m[1:]     # cell to the right of face
    kappa_left = kappa_io[:-1]
    kappa_right = kappa_io[1:]
    kappa_faces = (dx_right + dx_left) / (dx_right / kappa_left + dx_left / kappa_right)
    # Interior fluxes
    grad_faces = (Phi_io[1:] - Phi_io[:-1]) / dx_face_m
    flux_faces = -kappa_faces * grad_faces
    # BC fluxes with fixed BC convention
    # Conductivity is calculated considering the first and last node.
    if bc_left_phi_io.type == "neumann":
        flux_left_BC = -kappa_io[0] * bc_left_phi_io.value
    elif bc_left_phi_io.type == "dirichlet":
        flux_left_BC = -kappa_io[0] * (bc_left_phi_io.value - Phi_io[0]) / (dx_m[0] / 2)
    elif bc_left_phi_io.type == "galvanostatic":
        flux_left_BC = bc_left_phi_io.value
    else:
        raise ValueError(f"Invalid BC type for ionic potential at membrane left boundary: {bc_left_phi_io.type}")
    if bc_right_phi_io.type == "dirichlet":
        flux_right_BC = -kappa_io[-1] * (bc_right_phi_io.value - Phi_io[-1]) / (dx_m[-1] / 2)
    elif bc_right_phi_io.type == "neumann":
        flux_right_BC = -kappa_io[-1] * bc_right_phi_io.value
    elif bc_right_phi_io.type == "galvanostatic":
        flux_right_BC = bc_right_phi_io.value
    else:
        raise ValueError(f"Invalid BC type for ionic potential at membrane right boundary: {bc_right_phi_io.type}")
    # Residuals
    res_view[0, 1] = (flux_faces[0] - flux_left_BC)
    res_view[1:-1, 1] = (flux_faces[1:] - flux_faces[:-1]) 
    res_view[-1, 1] = (flux_right_BC - flux_faces[-1]) 
    return res_view.flatten()

def anode_residual(self, t, y_anode, yp_anode):
    # Sizes
    N      = self.mesh.anode_mesh.n_x
    n_vars = self.layout.n_vars_anode  # [c_vw, c_H2, s, eta, c_vw_agdl, c_H2_agdl, s_agdl]
    view     = y_anode.reshape((N, n_vars))
    du_view  = yp_anode.reshape((N, n_vars))
    res_view = np.zeros((N, n_vars))
    # State variables
    c_vw_acl  = view[:, 0]  # Vapor water in ACL   [mol/m3]
    c_H2_acl  = view[:, 1]  # H2 in ACL            [mol/m3]
    s_acl     = view[:, 2]  # Surface coverage ACL [0-1]
    eta       = view[:, 3]  # Overpotential ACL    [V]
    c_vw_agdl = view[:, 4]  # Vapor water in AGDL  [mol/m3]
    c_H2_agdl = view[:, 5]  # H2 in AGDL           [mol/m3]
    s_agdl    = view[:, 6]  # Surface coverage AGDL [0-1]
    # Properties
    a_anode        = self.anode.specific_surface_area
    F              = self.F
    porosity_acl   = self.anode.porosity
    porosity_agdl  = self.agdl.porosity
    thickness_a    = self.anode.thickness
    thickness_agdl = self.agdl.thickness
    # Butler Volmer (ACL)
    BV = self.BV_lumped(eta, c_H2_acl, "anode")  # [A/m2]
    # Total pressure
    p_total_acl = (c_vw_acl + c_H2_acl) * self.R * self.T
    p_total_agdl = (c_vw_agdl + c_H2_agdl) * self.R * self.T
    # Condensation / evaporation
    s_im     = self.anode.s_im  # Immobile saturation 
    beta_sr  = 100000            # Smoothing parameter for saturation correction
    s_r_acl  = np.logaddexp(0.0, beta_sr * (s_acl - s_im)) / (beta_sr * (1 - s_im))
    s_r_agdl = np.logaddexp(0.0, beta_sr * (s_agdl - s_im)) / (beta_sr * (1 - s_im))
    p_sat    = self.water_saturation_pressure() # [Pa]
    c_sat    = p_sat / (self.R * self.T)        # [mol/m3]
    p_vw_acl  = c_vw_acl * self.R * self.T      # [Pa]
    p_vw_agdl = c_vw_agdl * self.R * self.T     # [Pa]
    rho_lw  = self.anode.rho_lw                 # [kg/m3]
    MW      = self.anode.MW_w                   # [kg/mol]
    k_cond  = self.anode.k_cond                 # Condensation coefficient [1/s]
    k_evap  = self.anode.k_evap                 # Evaporation coefficient  [1/s/atm]
    x_acl = c_vw_acl - c_sat   # > 0 -> condensation, < 0 -> evaporation
    x_agdl = c_vw_agdl - c_sat # > 0 -> condensation, < 0 -> evaporation
    delta_sat = 0.05 * c_sat                    # smoothing width ~5% of c_sat
    w_acl = 0.5 * (1.0 + np.tanh(x_acl / delta_sat))
    w_agdl = 0.5 * (1.0 + np.tanh(x_agdl / delta_sat))
    R_w_cond_acl = -k_cond * thickness_a * porosity_acl * (1 - s_acl) * (c_vw_acl/(c_vw_acl+c_H2_acl)) * x_acl
    R_w_evap_acl = -k_evap * thickness_a * porosity_acl * s_r_acl * (rho_lw/MW) * (p_vw_acl - p_sat) / 101325
    R_w_acl = w_acl * R_w_cond_acl + (1 - w_acl) * R_w_evap_acl
    R_w_cond_agdl = -k_cond * thickness_agdl * porosity_agdl * (1 - s_agdl) * (c_vw_agdl/(c_vw_agdl+c_H2_agdl)) * x_agdl
    R_w_evap_agdl = -k_evap * thickness_agdl * porosity_agdl * s_r_agdl * (rho_lw/MW) * (p_vw_agdl - p_sat) / 101325
    R_w_agdl = w_agdl * R_w_cond_agdl + (1 - w_agdl) * R_w_evap_agdl
    #if c_vw_acl > c_sat: # condensation
    #    R_w_acl = -k_cond * thickness_a * porosity_acl * (1 - s_acl) * (c_vw_acl / (c_vw_acl + c_H2_acl)) * (c_vw_acl - c_sat)
    #else: # c_vw_acl <= c_sat # evaporation
    #    R_w_acl = -k_evap * thickness_a * porosity_acl * s_r_acl * (rho_lw / MW) * (p_vw_acl - p_sat) / 101325
    #if c_vw_agdl > c_sat: # condensation
    #    R_w_agdl = -k_cond * thickness_agdl * porosity_agdl * (1 - s_agdl) * (c_vw_agdl / (c_vw_agdl + c_H2_agdl)) * (c_vw_agdl - c_sat)
    #else: # c_vw_agdl <= c_sat # evaporation
    #    R_w_agdl = -k_evap * thickness_agdl * porosity_agdl * s_r_agdl * (rho_lw / MW) * (p_vw_agdl - p_sat) / 101325
    # Smooth saturation of R_w: tanh limiter is C-infinity, avoids Jacobian kinks
    max_R_w  = 10000
    R_w_acl  = max_R_w * np.tanh(R_w_acl  / max_R_w)
    R_w_agdl = max_R_w * np.tanh(R_w_agdl / max_R_w)
    # Membrane <-> ACL vapor flux
    RH_acl  = c_vw_acl / c_sat
    a_w_an  = RH_acl + 2*s_r_acl
    lam_eq  = self.return_lambda_eq(a_w_an)
    if lam_eq < 2.01:
        lam_eq = 2.01  # Minimum water content
    lam_acl = self.membrane.lam_acl
    f_v     = (lam_acl * self.membrane.V_w) / (lam_acl * self.membrane.V_w + self.membrane.V_m)
    c_f     = self.membrane.c_f # Membrane fixed charge concentration [mol/m3]
    k_desorp_vw = self.membrane.k_desorp_vw * f_v * (1 - s_acl)
    k_desorp_lw = self.membrane.k_desorp_lw * f_v * s_r_acl
    k_sorp_vw   = self.membrane.k_sorp_vw * f_v * (1 - s_acl)
    k_sorp_lw   = self.membrane.k_sorp_lw * f_v * s_r_acl
    delta_lam = lam_acl - lam_eq
    blend     = 0.5 * (1.0 + np.tanh(delta_lam / 0.1))  # 0 -> sorption, 1 -> desorption
    k_vw_eff  = blend * k_desorp_vw + (1 - blend) * k_sorp_vw
    k_lw_eff  = blend * k_desorp_lw + (1 - blend) * k_sorp_lw
    J_iw_vw   = k_vw_eff * (lam_acl-lam_eq) * c_f
    delta_lam = lam_acl - 22
    blend     = 0.5 * (1.0 + np.tanh(delta_lam / 0.25))   # w = smoothing width
    k_extra   = k_lw_eff * 10 # Extra kinetic factor for liquid water when membrane is superssaturated.
    J_iw_liq  = (k_lw_eff + k_extra * blend) * (lam_acl - lam_eq) * c_f
    # Residuals: liquid saturation
    p_cap_acl  = self.capillary_pressure(s_r_acl, "acl")
    p_liq_acl  = p_total_acl - p_cap_acl
    p_cap_agdl = self.capillary_pressure(s_r_agdl, "agdl")
    p_liq_agdl = p_total_agdl - p_cap_agdl
    mu_lw      = 0.001  # [Pa s]
    K_eff_acl  = self.anode.permeability * (1e-6 + s_r_acl)**3
    K_eff_agdl = self.agdl.permeability * (1e-6 + s_r_agdl)**3
    K_eff      = 2 * K_eff_acl * K_eff_agdl / (K_eff_acl + K_eff_agdl + 1e-40)
    J_lw       = - (rho_lw / MW) * K_eff / mu_lw * (p_liq_agdl - p_liq_acl) / (0.5 * (thickness_a + thickness_agdl))
    # ACL - AGDL vapor transport
    T         = self.T
    D_vw_acl  = self.anode.D_vw_eff * (T/333.15)**1.5 * (101325/(p_total_acl)) * (porosity_acl * (1 - s_acl))**1.5
    D_vw_agdl = self.agdl.D_vw_eff * (T/333.15)**1.5 * (101325/(p_total_agdl)) * (porosity_agdl * (1 - s_agdl))**1.5
    D_vw      = (thickness_a + thickness_agdl) / (thickness_a / D_vw_acl + thickness_agdl / D_vw_agdl) 
    J_vw      = -D_vw * (c_vw_agdl - c_vw_acl) / (0.5 * (thickness_a + thickness_agdl))
    # Residuals: ACL vapor water
    res_view[:, 0]  = -du_view[:, 0] * thickness_a * porosity_acl * (1 - s_acl) + J_iw_vw - J_vw + R_w_acl
    # Residuals: ACL H2 transport
    D_H2 = D_vw # Same for both gasses
    J_H2 = -D_H2 * (c_H2_agdl - c_H2_acl) / (0.5 * (thickness_a + thickness_agdl))
    S_BV = -BV / (2 * F)  # [mol/m2/s]
    res_view[:, 1] = -du_view[:, 1] * thickness_a * porosity_acl * (1 - s_acl) - J_H2 + S_BV
    # Residuals: ACL saturation
    res_view[:, 2] = -du_view[:, 2] * (thickness_a * porosity_acl * rho_lw / MW) + J_iw_liq - J_lw - R_w_acl 
    # Residuals: BV reaction balance
    res_view[:, 3] = BV + self.i_io
    # Residuals: AGDL vapor water
    RH_ach = self.bc[Vars.a]["anode"]["right"].value
    c_vw_ach = RH_ach * p_sat / (self.R * self.T)
    J_ch = -D_vw * (c_vw_ach - c_vw_agdl) / (0.5 * (thickness_agdl))
    res_view[:, 4] = -du_view[:, 4] * thickness_agdl * porosity_agdl * (1 - s_agdl) + J_vw - J_ch + R_w_agdl
    # AGDL H2
    D_H2_agdl = D_vw_agdl
    c_H2_ach  = self.bc[Vars.c_h2]["anode"]["right"].value
    J_H2_ch   = -D_H2_agdl * (c_H2_ach - c_H2_agdl) / (0.5 * (thickness_agdl))
    res_view[:, 5] = -du_view[:, 5] * thickness_agdl * porosity_agdl * (1 - s_agdl) + J_H2 - J_H2_ch
    # AGDL saturation
    p_liq_ach      = 103000                # Inlet pressure at channel/AGDL interface [Pa]
    J_lw_ch        = - (rho_lw / MW) * K_eff_agdl / mu_lw * (p_liq_ach - p_liq_agdl) / (0.5 * (thickness_agdl))
    res_view[:, 6] = -du_view[:, 6] * (thickness_agdl * porosity_agdl * rho_lw / MW) + J_lw - J_lw_ch - R_w_agdl 
    return res_view.flatten()

def coupling_residual(self, y_cathode_Nend, y_membrane_N0, y_coupling):
    # Unpack coupling variables
    N = 1 # Number of nodes at coupling interface
    n_vars = self.layout.n_vars_coupling
    view = y_coupling.reshape(N, n_vars)
    res_view = np.zeros((N, n_vars))
    c_H_ca_BC = view[:, 0] # H+ concentration at cathode boundary node
    # -- Cathode side H+ concentration BC --
    # Unpack variables
    c_V4_ca       = y_cathode_Nend[0]  # V4+ concentration at cathode node n_x-1
    c_V5_ca       = y_cathode_Nend[1]  # V5+ concentration at cathode node n_x-1
    c_H_ca        = y_cathode_Nend[2]  # H+ concentration at cathode node n_x-1
    Phi_io_ca     = y_cathode_Nend[5]  # Ionic potential at cathode node n_x-1
    Phi_io_m      = y_membrane_N0[1]   # Ionic potential at membrane node 0
    c_H_m         = self.membrane.c_f  +970# H+ concentration at membrane node 0
    # Constants
    dx_ca = self.mesh.cathode_mesh.dx[-1] * self.cathode.thickness
    dx_m  = self.mesh.membrane_mesh.dx[0] * self.membrane.thickness
    F     = self.F
    R     = self.R
    T     = self.T
    # Ionic conductances at the interface
    g_ca         = self.cathode.kappa_io[-1] / (dx_ca / 2)
    g_m          = self.membrane.kappa_io[0] / (dx_m / 2)
    # Potential jump
    Phi_jump     = 0 #-R*T/F * np.log(c_H_m / c_H_ca_BC)
    # Ionic current
    i_io         = - g_ca * g_m / (g_ca + g_m) * (Phi_io_m - Phi_io_ca - Phi_jump)
    # Ionic potential at cathode side
    Phi_io_star  = (g_m * Phi_io_m + g_ca * Phi_io_ca - g_m * Phi_jump) / (g_ca + g_m)
    # Proton flux
    D_H_eff_ca = self.cathode.D_H_plus_eff  
    mu_H_ca    = self.cathode.mu_H_plus
    N_H_ca     = - D_H_eff_ca * (c_H_ca_BC - c_H_ca) / (dx_ca/2) - mu_H_ca * (c_H_ca_BC + c_H_ca) / 2 * F * (Phi_io_star - Phi_io_ca) / (dx_ca/2)
    # Constant ionic current
    res_view[:,0] = N_H_ca - i_io / F # Ionic current must be equal to proton flux at the interface
    return res_view.flatten()

def update_BC_tank_cathode(self, y_tank, y_cathode_N1):
    # Unpack tank and cathode variables
    view = y_tank.reshape((1, 5)) # [V4, V5, H+, HSO4, V_T]
    c_T_V4   = y_tank[0] # V4+ concentration in tank   [mol/m3]
    c_T_V5   = y_tank[1] # V5+ concentration in tank   [mol/m3]
    c_T_H    = y_tank[2] # H+ concentration in tank    [mol/m3]
    c_T_HSO4 = y_tank[3] # HSO4- concentration in tank [mol/m3]
    V_T      = y_tank[4] # Volume of electrolyte       [m3]
    c_V4     = y_cathode_N1[0] # V4+ concentration at cathode node 1
    c_V5     = y_cathode_N1[1] # V5+ concentration at cathode node 1
    c_H      = y_cathode_N1[2] # H+ concentration at cathode node 1
    c_HSO4   = y_cathode_N1[3] # HSO4- concentration at cathode node 1
    Phi_io    = y_cathode_N1[5] # Ionic potential at cathode node 1
    N_tank   = np.zeros(4)
    # Calculate species fluxes at the channel/cathode interfaces
    # Constants
    dx_ca = self.mesh.cathode_mesh.dx[0] * self.cathode.thickness
    v     = self.cathode.u_tp
    F     = self.F
    # Migration flux
    bc_left_phi = self.bc[Vars.phi_io]["cathode"]["left"] 
    if bc_left_phi.type == "neumann":
        dPhi_0 = bc_left_phi.value
    elif bc_left_phi.type == "dirichlet":
        dPhi_0 = (Phi_io-bc_left_phi.value) / (dx_ca / 2)
    elif bc_left_phi.type == "galvanostatic":
        dPhi_0 = -bc_left_phi.value / self.cathode.kappa_io[0]  
    else:
        raise ValueError(f"Invalid BC type for ionic potential at cathode left boundary: {bc_left_phi.type}")
    # V4
    D_eff = self.cathode.D_V4_eff
    z     = +2
    mu    = self.cathode.mu_V4
    N_tank[0]     = - D_eff  * (c_V4-c_T_V4)/(dx_ca/2) + v * (c_T_V4) #- z * mu * (c_V4+c_T_V4)/2 * F * dPhi_0 #
    # V5
    D_eff = self.cathode.D_V5_eff
    z     = +1
    mu    = self.cathode.mu_V5
    N_tank[1]     = - D_eff  * (c_V5-c_T_V5)/(dx_ca/2) + v * (c_T_V5) #- z * mu * (c_V5+c_T_V5)/2 * F * dPhi_0 
    # H+
    D_eff = self.cathode.D_H_plus_eff
    z     = +1
    mu    = self.cathode.mu_H_plus
    N_tank[2]     = - D_eff  * (c_H-c_T_H)/(dx_ca/2) + v * (c_T_H) #- z * mu * (c_H+c_T_H)/2 * F * dPhi_0 
    # HSO4-
    D_eff = self.cathode.D_HSO4_minus_eff
    z     = -1
    mu    = self.cathode.mu_HSO4_minus
    N_tank[3]     = - D_eff  * (c_HSO4-c_T_HSO4)/(dx_ca/2) + v * (c_T_HSO4) #- z * mu * (c_HSO4+c_T_HSO4)/2 * F * dPhi_0 
    self.modify_BC(Vars.c_v4, "cathode", "left",  "dirichlet", c_T_V4)
    self.modify_BC(Vars.c_v5, "cathode", "left",  "dirichlet", c_T_V5)
    self.modify_BC(Vars.c_h, "cathode", "left",  "dirichlet", c_T_H)
    self.modify_BC(Vars.c_hso4, "cathode", "left",  "dirichlet", c_T_HSO4)
    return N_tank

def update_BC_cathode_membrane(self, y_cathode_Nend, y_membrane_N0, y_c_H_BC, y_cathode_V4, y_cathode_V5):
    # Unpack variables
    c_V4_ca       = y_cathode_Nend[0]  # V4+ concentration at cathode node n_x-1
    c_V5_ca       = y_cathode_Nend[1]  # V5+ concentration at cathode node n_x-1
    c_H_ca        = y_cathode_Nend[2]  # H+ concentration at cathode node n_x-1
    Phi_io_ca     = y_cathode_Nend[5]  # Ionic potential at cathode node n_x-1
    Phi_io_m      = y_membrane_N0[1]   # Ionic potential at membrane node 0
    c_H_m         = self.membrane.c_f  + 970 # H+ concentration at membrane node 0
    # Constants
    dx_ca = self.mesh.cathode_mesh.dx[-1] * self.cathode.thickness
    dx_m  = self.mesh.membrane_mesh.dx[0] * self.membrane.thickness
    F     = self.F
    R     = self.R
    T     = self.T
    # Ionic conductances at the interface
    g_ca         = self.cathode.kappa_io[-1] / (dx_ca / 2)
    g_m          = self.membrane.kappa_io[0] / (dx_m / 2)
    # H+ concentration in cathode side
    c_H_ca_BC    = y_c_H_BC  
    # Potential jump at the interface
    Phi_jump     = 0 #-R*T/F * np.log(c_H_m / c_H_ca_BC)
    Phi_io_star  = (g_m * Phi_io_m + g_ca * Phi_io_ca - g_m * Phi_jump) / (g_ca + g_m) # Phi in cathode side
    dPhi_io_ca   = (Phi_io_star - Phi_io_ca)/(dx_ca/2)
    Phi_io_prime = Phi_jump + Phi_io_star # Phi in membrane side
    dPhi_io_m    = (Phi_io_m - Phi_io_prime)/(dx_m/2)
    # ionic current in membrane
    self.i_io    = - g_ca * g_m / (g_ca + g_m) * (Phi_io_m - Phi_io_ca - Phi_jump)
    # Mode potentiostatic or galvanostatic
    bc_mode = self.bc_mode
    # Update BCs
    self.modify_BC(Vars.c_h, "cathode", "right",  "dirichlet", c_H_ca_BC)
    if bc_mode == "potentiostatic":
        self.modify_BC(Vars.phi_io, "cathode", "right",  "dirichlet", Phi_io_star)
        self.modify_BC(Vars.phi_io, "membrane", "left",  "neumann", dPhi_io_m)
    elif bc_mode == "galvanostatic":
        self.modify_BC(Vars.phi_io, "cathode", "right",  "neumann", dPhi_io_ca)
        self.modify_BC(Vars.phi_io, "membrane", "left",  "neumann", dPhi_io_m)
    # Modify BC for lambda in membrane based on concentration of vanadium
    dx_ca_all = self.mesh.cathode_mesh.dx
    mean_V4     = np.dot(dx_ca_all, y_cathode_V4) / np.sum(dx_ca_all)
    mean_V5     = np.dot(dx_ca_all, y_cathode_V5) / np.sum(dx_ca_all)
    SOC_mean    = mean_V5 / (mean_V4 + mean_V5)
    a_w_cathode = self.return_water_activity_cathode(SOC_mean)
    self.modify_BC(Vars.a, "membrane", "left", "dirichlet", a_w_cathode)
    
def update_BC_membrane_anode(self, y_membrane_Nend, y_anode):
    # lam, H+, Phi_io
    lam_m      = y_membrane_Nend[0]  # Water content at membrane 
    Phi_io_m   = y_membrane_Nend[1]  # Ionic potential at membrane 
    c_H_m      = self.membrane.c_f  + 970 # H+ concentration at membrane 
    # c_vw, c_H, s, eta, c_vw_agdl, c_H_agdl, s_agdl
    c_vw    = y_anode[0]  # Vapor water concentration at anode node 0 [mol/m3]
    c_H2    = y_anode[1]  # H2 concentration at anode [mol/m3]
    s       = y_anode[2]  # Saturation at anode node 0
    eta     = y_anode[3]  # Overpotential at anode node 0
    p_sat   = self.water_saturation_pressure()
    RH      = c_vw / (p_sat / (self.R * self.T))
    s_im    = self.anode.s_im
    beta_sr = 100000            # Smoothing parameter for saturation correction
    s_r     = np.logaddexp(0.0, beta_sr * (s - s_im)) / (beta_sr * (1 - s_im))
    a       = RH + 2*s_r
    # Update anode hydrogen overpotential
    R    = self.R  
    T    = self.T
    F    = self.F
    j_0  = self.anode.j_0*2
    n    = 2
    f    = F / (R * T)
    thickness_a             = self.anode.thickness
    ionomer_volume_fraction = self.anode.ionomer_volume_fraction
    sigma_an                = self.membrane.kappa_io[-1] 
    sigma_eff               = sigma_an * ionomer_volume_fraction**1.5
    c                       = (j_0 * n * f / (sigma_eff * thickness_a))**0.5
    self.eta_proton         = self.BV_lumped(eta, c_H2, "anode") / (sigma_eff * c) * ((np.exp(c*thickness_a)+np.exp(-c*thickness_a))/(np.exp(c*thickness_a)-np.exp(-c*thickness_a))-1/(c*thickness_a))
    # Calculate ionic potential at the interface
    Phi_el = 0 # Asuming that there is no electronic potential drop
    c_anode_ocp = np.array([c_H2, c_H_m])
    E_a = self.compute_OCP(c_anode_ocp, "anode")
    i_io         = self.i_io  # Ionic current from cathode/membrane interface
    Phi_io_prime = Phi_el - eta - E_a - self.eta_proton  # Ionic potential at membrane side of the interface from anode overpotential and Nernst potential
    dPhi_io_m    = -i_io / (self.membrane.kappa_io[-1]) 
    # Update BCs for membrane/anode interface 
    self.modify_BC(Vars.a, "membrane", "right",  "dirichlet", a)
    # Mode potentiostatic or galvanostatic
    bc_mode = self.bc_mode
    if bc_mode == "potentiostatic":
        self.modify_BC(Vars.phi_io, "membrane", "right",  "dirichlet", Phi_io_prime)
    elif bc_mode == "galvanostatic":
        self.modify_BC(Vars.phi_io, "membrane", "right",  "dirichlet", Phi_io_prime)
    self.modify_BC(Vars.s, "membrane", "right", "dirichlet", s)
    self.membrane.lam_acl = lam_m

def update_vanadium(self, y_cathode_Nend):
    c_V4_ca       = y_cathode_Nend[0]  # V4+ concentration at cathode node n_x-1
    c_V5_ca       = y_cathode_Nend[1]  # V5+ concentration at cathode node n_x-1
    # Modify uptake of vanadium
    c_V4_uptake = c_V4_ca * 0.1229
    c_V5_uptake = c_V5_ca * 0.0952
    self.modify_BC(Vars.c_v4, "membrane", "left", "dirichlet", c_V4_uptake) # mol/m3
    self.modify_BC(Vars.c_v5, "membrane", "left", "dirichlet", c_V5_uptake) # mol/m3
    