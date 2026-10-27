class Cathode:
    def __init__(self, CathodeParams):
        # Cathode Geometry
        self.thickness = CathodeParams.thickness # Cathode thickness [m]
        self.width     = CathodeParams.width     # Cathode width [m]
        self.height    = CathodeParams.height    # Cathode height [m]
        # Channel geometry
        self.channel_length = CathodeParams.channel_length # Transverse length of cathode channel [m]
        self.channel_width  = CathodeParams.channel_width  # Channel width [m]
        self.channel_depth = CathodeParams.channel_depth  # Channel height [m]
        # Channel friction
        self.n_bend = CathodeParams.n_bend # Number of bends in the channel [-]
        self.zeta_bend = CathodeParams.zeta_bend # Pressure drop coefficient for each bend [-]
        self.n_channel = CathodeParams.n_channel # Number of parallel channels in the cathode [-]
        # Flow parameters
        self.Q_total = CathodeParams.Q_total # Total volumetric flow rate in the cathode [m3/s]
        # Electrode structural properties
        self.porosity               = CathodeParams.porosity                 # [-]
        self.specific_surface_area  = CathodeParams.specific_surface_area    # [m2/m3]
        self.conductivity           = CathodeParams.conductivity             # Electronic conductivity [S/m]
        self.reaction_rate_constant = CathodeParams.reaction_rate_constant   # [m/s]
        self.eff_conductivity       = self.conductivity * (1 - self.porosity)**1.5 # Effective electronic conductivity [S/m]
        # Electrolyte properties (anolyte)
        self.rho_ca = CathodeParams.rho_ca  # Density [kg/m3]
        # Operating conditions
        self.u_tp = CathodeParams.u_tp      # Catholyte velocity in the transverse direction [m/s] (through-plane, into electrode thickness)
        self.u_ip = 0      # Catholyte velocity in the paralel direction [m/s] (in-plane, parallel to channel)
        self.u_ch = 0      # Channel velocity [m/s]
        # Mass transfer coefficient (Butler–Volmer)
        self.kappa_io = 0
        # Effective diffusion coefficients 
        self.D_V4_eff         = 0
        self.D_V5_eff         = 0
        self.D_H_plus_eff     = 0
        self.D_HSO4_minus_eff = 0
        self.D_SO4_2minus_eff = 0
        # Ion mobilities (to be computed later)
        self.mu_V2          = 0
        self.mu_V3          = 0
        self.mu_V4          = 0
        self.mu_V5          = 0
        self.mu_H_plus      = 0
        self.mu_HSO4_minus  = 0
        self.mu_SO4_2minus  = 0
        # OCV fitting parameter (activity coefficient)
        self.F_y = CathodeParams.F_y
    def return_effective_diffusivity(self, D):
        return D * self.porosity**1.5  # Effective diffusivity in the cathode [m2/s]
    
    
    