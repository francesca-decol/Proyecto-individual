import numpy as np
from scipy.optimize import root_scalar
def initialize_effective_diffusivities(self):
        self.cathode.D_V4_eff = self.cathode.return_effective_diffusivity(self.D_V4)
        self.cathode.D_V5_eff = self.cathode.return_effective_diffusivity(self.D_V5)
        self.cathode.D_H_plus_eff = self.cathode.return_effective_diffusivity(self.D_H_plus)
        self.cathode.D_HSO4_minus_eff = self.cathode.return_effective_diffusivity(self.D_HSO4_minus)
        self.cathode.D_SO4_2minus_eff = self.cathode.return_effective_diffusivity(self.D_SO4_2minus)
        self.anode.D_H2_eff = self.anode.return_effective_diffusivity(self.D_H2)
        self.anode.D_vw_eff = self.anode.return_effective_diffusivity(self.D_vw)
        self.agdl.D_H2_eff = self.agdl.return_effective_diffusivity(self.D_H2)
        self.agdl.D_vw_eff = self.agdl.return_effective_diffusivity(self.D_vw) 
def initialize_effective_mobilities(self):
        self.cathode.mu_V4 = self.return_mobility(self.cathode.D_V4_eff)
        self.cathode.mu_V5 = self.return_mobility(self.cathode.D_V5_eff)
        self.cathode.mu_H_plus = self.return_mobility(self.cathode.D_H_plus_eff)
        self.cathode.mu_HSO4_minus = self.return_mobility(self.cathode.D_HSO4_minus_eff)
        self.cathode.mu_SO4_2minus = self.return_mobility(self.cathode.D_SO4_2minus_eff)
        self.membrane.mu_V4 = self.return_mobility(self.membrane.D_V4)
        self.membrane.mu_V5 = self.return_mobility(self.membrane.D_V5)
        self.membrane.mu_H_plus = self.return_mobility(self.membrane.D_H_plus)
        self.membrane.mu_HSO4_minus = self.return_mobility(self.membrane.D_HSO4_minus)
def initialize_cathode_viscosity(self):
        # Viscosity of cathode electrolyte
        T = self.T  # Temperature [K]
        R = self.R  # Gas constant [J/(mol K)]
        self.cathode.mu_ca = 5.4e-6 * np.exp(16131/(R*T))
def solve_velocity(self):
    # Total volumetric flow rate in the cathode
    Q = self.cathode.Q_total  # Total volumetric flow rate [m3/s]
    # -- Volumetric flow rate --
    def find_Q_ca(Q_ca):
        Q_ch = Q - Q_ca  # Volumetric flow rate in the channel [m3/s]
        porosity_ca = self.cathode.porosity # [-]
        d_f         = self.d_f              # Fiber diameter [m]
        mu_ca       = self.cathode.mu_ca    # Dynamic viscosity of cathode electrolyte [Pa s]
        K_KC        = self.K_KC             # Kozeny-Carman constant [-]
        K_ca        = (porosity_ca * d_f**2)/(K_KC * (1-porosity_ca)**2) # Permeability of cathode [m2]
        h_ca        = self.cathode.height    # Height of cathode channel [m]
        w_ca        = self.cathode.width     # Width of cathode channel [m]
        t_ca        = self.cathode.thickness # Thickness of cathode [m]
        # Pressure drop across cathode
        A_ca  = t_ca * w_ca                          # Cross-sectional area of cathode channel [m2] NO ESTOY SEGURO SI AREA EFECTTIVA O AREA TOTAL DO CANAL
        dP_ca =  mu_ca * h_ca * Q_ca / (K_ca * A_ca) # Pressure drop across cathode [Pa]
        # Pressure drop across channel
        L_ca    = self.cathode.channel_length # Length of cathode channel [m]
        W_ch    = self.cathode.channel_width  # Width of cathode channel [m]
        H_ch    = self.cathode.channel_depth  # Height of cathode channel [m]
        rho_ca  = self.cathode.rho_ca   # Density of cathode electrolyte [kg/m3]
        D_ch    = 2 * W_ch * H_ch / (W_ch + H_ch) # Hydraulic diameter of channel [m]
        Re      = rho_ca * Q_ch  * D_ch / (mu_ca * W_ch * H_ch) # Reynolds number in channel [-]
        zeta_ch   = (10.35 * np.exp(-3.4 * H_ch / W_ch) + 13.84) * (1/Re)
        n_bend    = self.cathode.n_bend # Number of bends in the channel [-]
        zeta_bend = self.cathode.zeta_bend # Pressure drop coefficient for each bend [-]
        n_ch      = self.cathode.n_channel # Number of parallel channels in the cathode [-]
        dP_ch = (zeta_ch * L_ca/D_ch + n_bend * zeta_bend) * (rho_ca * (Q_ch/n_ch)**2) / (2 * (W_ch * H_ch)**2) # Pressure drop across channel [Pa]
        return dP_ca - dP_ch
    # Solve for Q_ca using root_scalar
    sol = root_scalar(find_Q_ca, bracket=[1e-10, Q*(1-1e-10)], method='bisect')
    if sol.converged:
        Q_ca = sol.root
        self.cathode.Q_ca = Q_ca
        self.cathode.u_ip = Q_ca / (self.cathode.width * self.cathode.thickness) # Superficial velocity in cathode [m/s]. Divide by porosity to get interstitial velocity.
        self.cathode.u_ch = (Q - Q_ca) / (self.cathode.channel_width * self.cathode.channel_depth) # Average velocity in channel [m/s]
    else:
        raise ValueError("Failed to solve for cathode volumetric flow rate Q_ca.")
    