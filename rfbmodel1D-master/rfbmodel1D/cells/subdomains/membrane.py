class Membrane:
    def __init__(self, MembraneParams):
        self.thickness = MembraneParams.thickness
        self.c_f = MembraneParams.c_f  # Fixed charge concentration [mol/m3]
        self.z_f = MembraneParams.z_f  # Fixed charge valence
        self.V_m = MembraneParams.V_m  # Dry membrane molar volume [m3/mol]
        self.V_w = MembraneParams.V_w  # Liquid water molar volume [m3/mol]
        # Diffusion properties
        self.D_V4 = MembraneParams.D_V4                 # Diffusion coefficient of V(4) [m2/s]
        self.D_V5 = MembraneParams.D_V5                 # Diffusion coefficient of V(5) [m2/s]
        self.D_H_plus = MembraneParams.D_H_plus         # Diffusion coefficient of H+ [m2/s]
        self.D_HSO4_minus = MembraneParams.D_HSO4_minus # Diffusion coefficient of HSO4- [m2/s]
        # Mobility properties
        self.mu_V2 = 0          # Mobility of V(2) [m2/(V.s)]
        self.mu_V3 = 0          # Mobility of V(3) [m2/(V.s)]
        self.mu_V4 = 0          # Mobility of V(4) [m2/(V.s)]
        self.mu_V5 = 0          # Mobility of V(5) [m2/(V.s)]
        self.mu_H_plus = 0      # Mobility of H+ [m2/(V.s)]
        self.mu_HSO4_minus = 0  # Mobility of HSO4- [m2/(V.s)]
        self.mu_SO4_2minus = 0  # Mobility of SO4(2-) [m2/(V.s)]
        # Interfacial mass transfer coefficients
        self.k_sorp_vw   = MembraneParams.k_sorp_vw                # Sorption rate for vapor water (m/s)
        self.k_desorp_vw = MembraneParams.k_desorp_vw              # Desorption rate for vapor water (m/s)
        self.k_sorp_lw   = MembraneParams.k_sorp_lw                # Sorption rate for liquid water (m/s)
        self.k_desorp_lw = MembraneParams.k_desorp_lw              # Desorption rate for liquid water (m/s)
        self.k_ca_lw     = MembraneParams.k_ca_lw                  # Mass transfer coefficient for liquid water at cathode interface (m/s)
        self.diffusivity_model = MembraneParams.diffusivity_model  # Water diffusivity model name
        # Properties to be updated during simulation
        self.kappa_io = 0
        


    