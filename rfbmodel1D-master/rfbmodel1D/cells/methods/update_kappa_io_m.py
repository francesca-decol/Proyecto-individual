import numpy as np
from rfbmodel1D.parameters.params import Vars

def update_kappa_io_m(self, lambda_m, Phi_io):
    """Ionic conductivity of the membrane from water content, with optional
    vanadium-uptake correction to proton mobility.
    (1) Weber, A. Z.; Newman, J. Transport in Polymer-Electrolyte Membranes II. Mathematical Model. Journal of The Electrochemical Society 2004, 151 (2).
    (2) Darling, R. M.; Weber, A. Z.; Tucker, M. C.; Perry, M. L. The Influence of Electric Field on Crossover in Redox-Flow Batteries. J. Electrochem. Soc. 2016, 163 (1), A5014–A5022. https://doi.org/10.1149/2.0031601jes."""
    R, T, F  = self.R, self.T, self.F
    V_m, V_w = self.membrane.V_m, self.membrane.V_w   # molar volumes [m3/mol]
    c_H_plus = self.membrane.c_f                      # [mol/m3]
    c_f      = self.membrane.c_f                      # [mol/m3]
    # EOD coefficient integrated over the membrane 
    xi_EOD = np.sum(self.EOD_coefficient(lambda_m) * self.mesh.membrane_mesh.dx)
    # Soft floor at λ=2 with delta=0.3.
    _delta   = 0.3
    lambda_m = 0.5 * (lambda_m + 2.0 + np.sqrt((lambda_m - 2.0)**2 + _delta**2))
    lambda_m = np.minimum(lambda_m, 22.0)
    # Base proton mobility from water content
    T_ref    = 303.15              # [K]
    pre_exp  = 50 / (F**2 * c_f)   # [S/m] converted to [mol2/J/s] for molar mobility
    f_v      = lambda_m * V_w / (lambda_m * V_w + V_m) # Water volume fraction [-]
    u_H_plus = np.where(
        f_v <= 0.45,
        pre_exp * (f_v - 0.06) ** 1.5 * np.exp(15000 / (R) * (1/T_ref-1/T)),
        pre_exp * (0.39)        ** 1.5 * np.exp(15000 / (R) * (1/T_ref-1/T)),
    )                                                  # [mol^2/(J·s)] [1]
    kappa_io_m = F**2 * u_H_plus * c_H_plus            # [S/m]
    if self.vanadium_uptake:                           # Optional correction to proton mobility due to vanadium uptake in the membrane
        # Boundary concentrations
        c_V4_0 = self.bc[Vars.c_v4]["membrane"]["left"].value   # [mol/m3]
        c_V5_0 = self.bc[Vars.c_v5]["membrane"]["left"].value   # [mol/m3]
        # Membrane-internal diffusivities
        D_V4    = 2.44e-10  # m2/s [2]
        D_V5    = 3.28e-10  # m2/s [2]
        u_V4 = 2  * D_V4 / (R * T)   # electrical mobility, |z|=2  -> [m2/V/s]
        u_V5 = 1 * D_V5 / (R * T)   # |z|=1
        # Ionic current density
        i_io = self.bc[Vars.phi_el]["cathode"]["left"].value
        l    = self.membrane.thickness
        z    = self.mesh.membrane_mesh.x_grid    
        delta_Phi = Phi_io[-1] - Phi_io[0] + 1e-12  # guard against /0 at startup
        kappa_ref = abs(i_io) * l / abs(delta_Phi)  # Ohm's law
        c_w = np.sum(lambda_m / (lambda_m * V_w + V_m) * self.mesh.membrane_mesh.dx) # Water concentration in the membrane [mol/m3]
        # Analytical solution to 1D steady-state diffusion with migration and EOD in the membrane [2]
        def vanadium_profile(c_0, z_i, D):
            beta = (z_i * F / (kappa_ref * R * T) + xi_EOD / (c_w * D * F)) * i_io * l
            if np.all(np.abs(beta) < 1e-12):
                return c_0 * (1.0 - z)
            return c_0 * (np.exp(beta * z) - np.exp(beta)) / (1.0 - np.exp(beta))
        c_V4 = vanadium_profile(c_V4_0, +2, D_V4)  
        c_V5 = vanadium_profile(c_V5_0, +1, D_V5)
        self.V4_profile = c_V4
        self.V5_profile = c_V5
        u_H_plus        = u_H_plus - c_V4 / c_f * 4.01e-8 - c_V5 / c_f * 6.62e-8
        kappa_io_m = F**2 * (
            u_H_plus * c_H_plus
            + 2 * c_V4 * u_V4
            + 1 * c_V5 * u_V5
        )
    self.membrane.kappa_io = kappa_io_m

def update_kappa_io_cathode(self, y_cathode):
    """Calculate ionic conductivity of cathode from local concentrations. 
    Considers spatial variation in concentration.
    (1) Newman, J.; Balsara, N. P. Electrochemical Systems. 2021.
    """
    n_vars = self.layout.n_vars_cathode
    N      = self.mesh.cathode_mesh.n_x       # Number of nodes in cathode
    eps    = self.cathode.porosity
    view   = y_cathode.reshape((N, n_vars))
    c_V4   = view[:, 0]  # V4+ concentration [mol/m3]
    c_V5   = view[:, 1]  # V5+ concentration [mol/m3]
    c_H    = view[:, 2]  # H+ concentration [mol/m3]
    c_HSO4 = view[:, 3]  # HSO4- concentration [mol/m3]
    c_SO4  = (2*c_V4 + 1*c_V5 + c_H - c_HSO4) / 2  # SO4(2-) concentration [mol/m3], from electroneutrality
    F      = self.F
    kappa_io = F**2 * (
        (2**2) * self.cathode.mu_V4         * c_V4 +
        (1**2) * self.cathode.mu_V5         * c_V5 +
        (1**2) * self.cathode.mu_H_plus     * c_H +
        (1**2) * self.cathode.mu_HSO4_minus * c_HSO4 +
        (2**2) * self.cathode.mu_SO4_2minus * c_SO4)
    # Bruggeman correction
    self.cathode.kappa_io = kappa_io * eps **1.5