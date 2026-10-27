import numpy as np
from rfbmodel1D.parameters.params import Vars
def compute_OCP(self, c, electrode):
    """Compute the open-circuit voltage (OCV) for the given electrode based on the concentrations of species.
    ---Inputs
    c: list or array of concentrations of species in the electrolyte [mol/m3]
    electrode: string, either "cathode" or "anode".     
    ---Outputs
    E: float, open-circuit voltage (OCV) [V]
    ---Coments
    - Cathode OCV considers all species in the electrolyte, following Eq. from [1] 
    and using a fitting parameter to represent the non-unitary activity coefficients of the catholyte.
    - Anode OCV considers only the H2/H+ redox couple, following the Nernst equation [2].
    (1) Pino-Muñoz, C. A.; Dewage, H. H.; Yufit, V.; Brandon, N. P. A Unit Cell Model of a Regenerative Hydrogen-Vanadium Fuel Cell. J. Electrochem. Soc. 2017, 164 (14), F1717. https://doi.org/10.1149/2.1431714jes.
    (2) Ye, M.; Pino-Muñoz, C. A.; Cannon, C. G.; Sun, Y.; Kucernak, A.; Klusener, P. A. A.; Brandon, N. Understanding H2-X Redox Flow Battery Performance Through Modelling and Design of Experiments. SSRN 2025. https://doi.org/10.2139/ssrn.5126397.
    """
    # Compute OCP 
    E = 0
    if electrode == "cathode":
        c_arr = np.asarray(c, dtype=float)
        c_V4, c_V5, c_H = c_arr[:3] / 1000.0  # Concentrations [mol/L]
        # Constants
        f = self.F/(self.R * self.T)
        # Smooth floor
        eps  = 1e-3
        c_V4 = np.sqrt(c_V4**2 + eps**2)
        c_V5 = np.sqrt(c_V5**2 + eps**2)
        c_H  = np.sqrt(c_H**2  + eps**2)
        # Nernst potential
        a_w_ca = self.bc[Vars.a]["membrane"]["left"].value
        F_y     = 10.02#self.cathode.F_y
        #F_y    = fitted_activity_coef(self)
        Ec0    = 1  + self.dS_r/self.F * (self.T - 298.15) # Standard potential for V4+/V5+ redox couple [V]
        E      = Ec0 + (1/f)*np.log(c_V5 * c_H**2 / (c_V4 * a_w_ca) * F_y)
    if electrode == "anode":
        p_H2 = c[0]*self.R*self.T / 101325 # Concentration of H2 [mol/m3 to atm]
        c_H  = c[1]/1000 # Concentration of H+ in membrane [mol/m3 to mol/L]
        # Constants
        f = self.F/(self.R * self.T)
        # Nernst potential
        Ea0 = 0  # Standard potential for H2/H+ redox couple [V]
        E   = Ea0 + (1/f)*np.log(c_H/np.sqrt(p_H2))
    return E 
        
def fitted_activity_coef(self):
    """Quadratic fit to F_y(T) from cell-potential measurements at
    24, 32, 32 (repeat) and 40 degC -> F_y = 24.2, 25.3, 25.2, 25.0.
    max |residual| = 0.05 
    (1) Ye, M.; Pino-Muñoz, C. A.; Cannon, C. G.; Sun, Y.; Kucernak, A.; Klusener, P. A. A.; Brandon, N. Understanding H2-X Redox Flow Battery Performance Through Modelling and Design of Experiments. SSRN 2025. https://doi.org/10.2139/ssrn.5126397."""
    T_ref = 298.15  # K (25 degC)
    c2, c1, c0 = -0.010156, 0.192188, 24.402344
    dT = self.T - T_ref
    return c0 + c1*dT + c2*dT**2