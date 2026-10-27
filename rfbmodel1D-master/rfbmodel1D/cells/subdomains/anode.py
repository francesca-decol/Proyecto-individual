class Anode:
    def __init__(self, AnodeParams): 
        # Geometry
        self.thickness = AnodeParams.thickness                     # Anode thickness [m]
        self.width = AnodeParams.width                             # Anode width [m]
        # Electrode structural properties
        self.porosity                = AnodeParams.porosity                 # [-]
        self.ionomer_volume_fraction = AnodeParams.ionomer_volume_fraction  # [-]
        self.specific_surface_area   = AnodeParams.specific_surface_area # [m2/m3]
        self.conductivity            = AnodeParams.conductivity          # Electronic conductivity [S/m]
        # Structural properties
        self.permeability = AnodeParams.permeability                   # [m2]
        self.contact_angle = AnodeParams.contact_angle                 # [degrees]
        # Electrochemical kinetics
        self.reaction_rate_constant = AnodeParams.reaction_rate_constant  # [m/s]
        self.j_0 = AnodeParams.j_0 # Exchange current density [A/m2]
        # Effective diffusion coefficients (to be computed later)
        # Transport properties
        self.D_H2_eff = 0                      # Diffusion coefficient of H2 [m2/s]
        self.D_vw_eff = 0                      # Diffusion coefficient of water vapor in AGDL [m2/s]
        self.rho_lw   = AnodeParams.rho_lw     # Density of liquid water [kg/m3]
        self.MW_w     = AnodeParams.MW_w       # Molecular weight of water [kg/mol]
        self.k_cond   = AnodeParams.k_cond     # Condensation rate constant [1/s]
        self.k_evap   = AnodeParams.k_evap     # Evaporation rate
        self.s_im     = AnodeParams.s_im       # Imbibition saturation (minimum saturation for liquid water to be present) [-]
        # Effective solid-phase conductivity
        self.eff_conductivity = self.conductivity * (1 - self.porosity - self.ionomer_volume_fraction)**1.5 # CHECK MAYBE POROSITY ALREADY CONSIDERS IONOMER VOLUME FRACITON
        # Boundary conditions dictionary
        self.bcs = dict()
        self.eta_proton = 0 # Initial guess for anode proton overpotential

    def return_effective_diffusivity(self, D):
        """ The diffusivity corrected by the Bruggeman correlation for porous media is calculated in DAE system"""
        return D 
