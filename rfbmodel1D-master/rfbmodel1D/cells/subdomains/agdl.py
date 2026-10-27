class AGDL:
    def __init__(self, AGDLParams): 
        # Geometry
        self.thickness = AGDLParams.thickness                     # AGDL thickness [m]
        self.porosity = AGDLParams.porosity                           # [-]
        self.permeability = AGDLParams.permeability                   # [m2]
        self.contact_angle = AGDLParams.contact_angle                 # [degrees]
        # Transport properties
        self.D_H2_eff = None
        self.D_vw_eff = None
        self.conductivity = AGDLParams.conductivity               # Electrical conductivity [S/m]
        self.eff_conductivity = self.conductivity *(1-self.porosity)**1.5 
        # Boundary conditions dictionary
        self.bcs = dict()

    def return_effective_diffusivity(self, D):
        return D 
