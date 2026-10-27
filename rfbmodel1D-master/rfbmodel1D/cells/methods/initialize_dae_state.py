import numpy as np
from rfbmodel1D.cells import cell
from rfbmodel1D.parameters.params import SimulationParams, Vars
from scipy.optimize import fsolve

def initialize_dae_state(self, c_tank, xtol=1e-6):
    # -- Tank --------------------------------------------------------------
    c_V4_T   = c_tank[0]  # Concentration of V4+   in the tank [mol/m3]
    c_V5_T   = c_tank[1]  # Concentration of V5+   in the tank [mol/m3]
    c_H_T    = c_tank[2]  # Concentration of H+    in the tank [mol/m3]
    c_HSO4_T = c_tank[3]  # Concentration of HSO4- in the tank [mol/m3]
    V_T      = c_tank[4]  # Volume of the tank [m3]

    # -- Potentials --------------------------------------------------------
    if self.bc_mode == "potentiostatic":
        Phi_el       = self.bc[Vars.phi_el]["cathode"]["left"].value           # Potentiostatic BC [V]
        OCP          = self.compute_OCP(c_tank[:4], electrode="cathode")       # Open circuit potential [V]
        eta_ca_guess = -0.1                                                    # Initial guess for cathode overpotential (negative for discharge) [V]
        Phi_io       = Phi_el - OCP  - eta_ca_guess                            # Initial ionic potential
        self.modify_BC(Vars.phi_io, "cathode", "left",  "neumann", 0)
    elif self.bc_mode == "galvanostatic":
        Phi_el       = 0.8  # Initial guess for cathode electronic potential [V]
        OCP          = self.compute_OCP(c_tank[:4], electrode="cathode")       # Open circuit potential [V]
        i_applied    = self.bc[Vars.phi_el]["cathode"]["left"].value           # Applied current density [A/m2] (+ charge / - discharge)
        tafel_slope  = self.R * self.T / (self.alpha_c * self.F)               # [V], RT/(alpha_c F)
        i_ref        = 50.0                                                    # [A/m2] reference current scale
        eta_ca_guess = np.sign(i_applied) * tafel_slope * np.log1p(abs(i_applied) / i_ref)
        Phi_io       = Phi_el - OCP  - eta_ca_guess                            # Initial ionic potential
        self.modify_BC(Vars.phi_io, "cathode", "left",  "neumann", 0)
    else:
        raise ValueError(f"Unknown BC mode: {self.bc_mode}")
    # -- Cathode Tank: [c_V4, c_V5, c_H, c_HSO4, V] ------------------------------
    y_tank = np.array([c_V4_T, c_V5_T, c_H_T, c_HSO4_T, V_T])

    # -- Cathode: [c_V4, c_V5, c_H, c_HSO4, Phi_el, Phi_io] per node -----
    y_initial_cathode = np.array([c_V4_T, c_V5_T, c_H_T, c_HSO4_T, Phi_el, Phi_io])
    y_cathode         = np.tile(y_initial_cathode, self.mesh.cathode_mesh.n_x)
    self.update_kappa_io_cathode(y_cathode)  # Set initial kappa_io in cathode

    # -- Anode: [c_vw_acl, c_H2_acl, s_acl, eta_acl, 
    #            c_vw_agdl, c_H2_agdl, s_agdl] ---------------------------
    # Channel activity and H2 concentration 
    activity_ch = self.bc[Vars.a]["anode"]["right"].value    # Activity of water in the channel [-]
    c_H2_ch     = self.bc[Vars.c_h2]["anode"]["right"].value # H2 concentration in the channel [mol/m3]
    p_sat       = self.water_saturation_pressure()           # Saturation pressure of water at T [Pa]
    # Initial guess
    c_vw_acl  = activity_ch * p_sat / (self.R * self.T)    # Vapor water in ACL  [mol/m3]
    c_H2_acl  = c_H2_ch                                    # H2 gas in ACL       [mol/m3]  
    s_acl     = 0.121                                       # Liquid saturation in ACL  [-]
    eta_acl   = 0                                          # Overpotential in ACL      [V]
    c_H2_agdl = c_H2_ch                                    # H2 gas in AGDL      [mol/m3]
    c_vw_agdl = activity_ch * p_sat / (self.R * self.T)    # Vapor water in AGDL [mol/m3]
    s_agdl    = 0.121                                      # Liquid saturation in AGDL [-]
    # Equilibrium membrane water content based on anode interface activity
    s_im = self.anode.s_im
    a_an = activity_ch + 2 * (s_acl - s_im)/(1-s_im)  # Activity of water at anode/membrane interface
    lambda_an_eq = self.return_lambda_eq(a_an)  # Equilibrium water content in the membrane at t=0
    
    # Build anode state vector
    y_anode = np.array([c_vw_acl, c_H2_acl, s_acl, eta_acl, c_vw_agdl, c_H2_agdl, s_agdl])

    # -- Membrane: [lam, Phi_io] per node -----------------------------
    Nm = self.mesh.membrane_mesh.n_x
    SOC = c_V5_T / (c_V4_T + c_V5_T)  # State of charge based on tank concentrations
    a_w_cathode = self.return_water_activity_cathode(SOC)
    lambda_cath_eq = self.return_lambda_cath_eq(a_w_cathode) # Initial guess for membrane water content based on cathode activity
    lam_initial = np.linspace(lambda_cath_eq, lambda_an_eq, Nm)          # shape (Nm,)
    phi_io_m0   = Phi_io - 0.02                    # scalar
    Phi_io_m    = np.full(Nm, phi_io_m0)           # shape (Nm,)

    # Build [lam, Phi_io] per node and flatten to 1D as expected by y vector
    y_membrane = np.column_stack((lam_initial, Phi_io_m)).ravel()
    
    # -- Coupling variables: [c_H_ca_BC] -----------------------------
    c_H_ca_BC = c_H_T  # Initial guess for proton concentration at cathode boundary (from tank)
    y_coupling = np.array([c_H_ca_BC])

    # -- Assemble full state vector ----------------------------------------
    # [tank(5) | cathode(6*n_x_ca) | membrane(2*n_x_m) | anode(7)]
    y0 = np.concatenate([y_tank, y_cathode, y_membrane, y_anode, y_coupling])
    # Solve algebraic variables for consistency with DAE at t=0
    y0 = solve_algebraic(self, y0, xtol=xtol)
    if self.bc_mode == "galvanostatic":
        if i_applied > 400: # type: ignore
            pass
            # Solve for warm state (algebraic + fast differential variables) for consistency with DAE at t=0
            #for i_k in np.linspace(0, i_applied, 3)[1:]:   # type: ignore # skip 0, start already have that guess
            #    self.modify_BC(Vars.phi_el, "cathode", "left", "galvanostatic", i_k)
            #    y0 = solve_warm_state(self, y0, xtol=xtol)
    # Solve differential variables for consistency with DAE at t=0
    dy0 = solve_differential(self, y0, xtol=xtol)
    #dy0 = np.zeros_like(y0)  # Start with zero time derivatives (consistent with steady state)
    return y0, dy0

def solve_algebraic(self, y0, xtol=1e-8):
    """
    Solve only algebraic variables using dae_residual_full_cell at t=0.
    y0 is the full state vector:
    [tank(5), cathode(6*Nc), membrane(2*Nm), anode(7), coupling(1)]
    Algebraic variables are:
    - Cathode: Phi_el, Phi_io at each node (2 per node)
    - Membrane: Phi_io at each node (1 per node)
    - Anode: eta_acl (1 variable)
    - Coupling: c_H_ca_BC (1 variable)
    Returns full y vector with algebraic variables solved for consistency with DAE at t=0
    """
    Nc = self.mesh.cathode_mesh.n_x
    Nm = self.mesh.membrane_mesh.n_x
    # Work on a copy
    y_fixed = y0.copy()
    # Global offsets in full y
    off_tank = 0
    off_ca   = off_tank + 5
    off_mem  = off_ca + 6 * Nc
    off_an   = off_mem + 2 * Nm
    off_coupling = off_an + 7
    # Algebraic global indices in y
    idx_phi_el_ca = off_ca + (np.arange(Nc) * 6 + 4)    # Phi_el in cathode
    idx_phi_io_ca = off_ca + (np.arange(Nc) * 6 + 5)    # Phi_io in cathode
    idx_phi_io_m  = off_mem + (np.arange(Nm) * 2 + 1)   # Phi_io in membrane
    idx_eta_acl   = np.array([off_an + 3], dtype=int)   # eta_acl in anode
    idx_c_H_ca_BC = np.array([off_coupling], dtype=int) # H+ concentration BC for cathode interface
    idx_alg       = np.concatenate([
        idx_phi_el_ca, idx_phi_io_ca, idx_phi_io_m, idx_eta_acl, idx_c_H_ca_BC])
    # Initial algebraic guess from y0
    x0 = y_fixed[idx_alg].copy()
    def alg_residual(x):
        y = y_fixed.copy()
        y[idx_alg] = x
        yp = np.zeros_like(y)  # for algebraic consistency solve at t=0
        res_full = self.dae_residual_full_cell(t=0.0, y=y, yp=yp)
        return res_full[idx_alg]
    x_sol, info, ier, msg = fsolve(alg_residual, x0, xtol=xtol, full_output=True)
    if ier != 1:
        raise RuntimeError(f"Algebraic solve failed: {msg}")
    y_sol = y_fixed.copy()
    y_sol[idx_alg] = x_sol
    return y_sol

def solve_differential(self, y0, xtol=1e-8):
    """
    Solve only differential time derivatives du using dae_residual_full_cell at t=0.
    Returns full du0 vector (same size as y0).
    """
    Nc = self.mesh.cathode_mesh.n_x
    Nm = self.mesh.membrane_mesh.n_x
    # Offsets in y = [tank(5), cathode(6*Nc), membrane(2*Nm), anode(7), coupling(1)]
    off_tank = 0
    off_ca   = off_tank + 5
    off_mem  = off_ca + 6 * Nc
    off_an   = off_mem + 2 * Nm
    # Differential variable indices (same indices used for residual equations)
    idx_tank = np.arange(off_tank, off_tank + 5)  # tank: all differential
    idx_ca = off_ca + (np.arange(Nc)[:, None] * 6 + np.array([0, 1, 2, 3]))
    idx_ca = idx_ca.ravel()  # cathode species only
    idx_mem = off_mem + (np.arange(Nm) * 2 + 0)   # membrane lambda only
    idx_an = off_an + np.array([0, 1, 2, 4, 5, 6], dtype=int)  # anode except eta
    idx_diff = np.concatenate([idx_tank, idx_ca, idx_mem, idx_an])
    # Unknowns are only du at differential positions
    x0 = np.zeros(idx_diff.size)
    # Solve 
    def diff_residual(du_diff):
        du = np.zeros_like(y0)
        du[idx_diff] = du_diff
        res_full = self.dae_residual_full_cell(t=0.0, y=y0, yp=du)
        return res_full[idx_diff]
    du_diff_sol, info, ier, msg = fsolve(diff_residual, x0, xtol=xtol, full_output=True)
    du0 = np.zeros_like(y0)
    if ier != 1:
        raise RuntimeError(f"Differential solve failed: {msg}")
    du0[idx_diff] = du_diff_sol
    return du0

def solve_warm_state(self, y0, xtol=1e-8):
    "Solves for the DAE state, given an initial guess, by solving the algebraic and fast differential variables "
    "(c_vw_acl, c_H2_acl, s_acl, c_vw_agdl, c_H2_agdl, s_agdl). Returns y0, dy0."
    Nc = self.mesh.cathode_mesh.n_x
    Nm = self.mesh.membrane_mesh.n_x
    # Work on a copy
    y_fixed = y0.copy()
    # Global offsets in full y
    off_tank = 0
    off_ca   = off_tank + 5
    off_mem  = off_ca + 6 * Nc
    off_an   = off_mem + 2 * Nm
    off_coupling = off_an + 7
    # Algebraic global indices in y
    idx_phi_el_ca = off_ca + (np.arange(Nc) * 6 + 4)    # Phi_el in cathode
    idx_phi_io_ca = off_ca + (np.arange(Nc) * 6 + 5)    # Phi_io in cathode
    idx_phi_io_m  = off_mem + (np.arange(Nm) * 2 + 1)   # Phi_io in membrane
    idx_an = off_an + np.array([0, 1, 2, 4, 5, 6], dtype=int)  # anode except eta
    idx_eta_acl   = np.array([off_an + 3], dtype=int)   # eta_acl in anode
    idx_c_H_ca_BC = np.array([off_coupling], dtype=int) # H+ concentration BC for cathode interface
    idx_warm       = np.concatenate([
        idx_phi_el_ca, idx_phi_io_ca, idx_phi_io_m, idx_an, idx_c_H_ca_BC, idx_eta_acl])
    # Initial algebraic guess from y0
    # Initial algebraic guess from y0
    x0 = y_fixed[idx_warm].copy()
    def warm_residual(x):
        y = y_fixed.copy()
        y[idx_warm] = x
        yp = np.zeros_like(y)  # for algebraic consistency solve at t=0
        res_full = self.dae_residual_full_cell(t=0.0, y=y, yp=yp)
        return res_full[idx_warm]
    x_sol, info, ier, msg = fsolve(warm_residual, x0, xtol=xtol, full_output=True)
    if ier != 1:
        raise RuntimeError(f"Warm solve failed: {msg}")
    y_sol = y_fixed.copy()
    y_sol[idx_warm] = x_sol
    return y_sol
    