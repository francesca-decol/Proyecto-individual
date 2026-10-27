import numpy as np

def cathode_surface_concentrations(self, eta, c_V4, c_V5, c_H_plus):
    """
    Calculate the mass-transfer-limited surface concentrations (c_s_V4, c_s_V5) for the cathode 
    Butler-Volmer reaction, together with the Damkohler numbers Psi3, Psi4 and 
    the mass-transfer coefficient beta_0.
    c_surface = c_bulk as Psi3, Psi4 -> 0, i.e. when mass
    transfer is fast relative to kinetics (high beta_0 / low k_ca).
    Following the approach of You et al. (2009).
    You, D.; Zhang, H.; Chen, J. A Simple Model for the Vanadium Redox Battery.
    Electrochimica Acta 2009, 54 (27), 6827–6836. https://doi.org/10.1016/j.electacta.2009.06.086.
    """
    f      = self.F / (self.R * self.T)
    k_ca   = self.cathode.reaction_rate_constant  # Cathode rate constant dependence on T
    v      = self.cathode.u_ip                    # In plane velocity
    k_m    = 7 * self.cathode.D_V4_eff / self.d_f * (self.d_f / (self.cathode.mu_ca / self.cathode.rho_ca))**(0.4)
    beta_0 = k_m * v**0.4                         # Mass transfer coefficient
    alpha  = self.alpha_c                         # Transfer coefficient of reaction
    # sqrt regularisation to avoid negative concentrations
    eps    = 1e-4
    c_V4_bulk = np.sqrt(np.maximum(c_V4, 0)**2 + eps**2)
    c_V5_bulk = np.sqrt(np.maximum(c_V5, 0)**2 + eps**2)
    c_H_plus = np.sqrt(np.maximum(c_H_plus, 0)**2 + eps**2)
    # sign(c_s_V5 - c_V5_bulk) == sign(eta) regardless of concentration scale.
    Psi3 = (k_ca/beta_0) * (c_V5_bulk/c_V4_bulk)**alpha * c_H_plus**(2-2*alpha) * np.exp((1-alpha)*f*eta)
    Psi4 = (k_ca/beta_0) * (c_V4_bulk/c_V5_bulk)**(1-alpha) * c_H_plus**(2-2*alpha) * np.exp(-alpha*f*eta)
    c_s_V4 = (Psi4*c_V5_bulk + (1+Psi4)*c_V4_bulk) / (1+Psi3+Psi4)
    c_s_V5 = (Psi3*c_V4_bulk + (1+Psi3)*c_V5_bulk) / (1+Psi3+Psi4)
    return c_s_V4, c_s_V5, c_V4_bulk, c_V5_bulk, Psi3, Psi4, beta_0

def butler_volmer(self, eta, c, electrode):
    """
    Calculate the Butler-Volmer transfer current per electrode geometric area (A/m2).
    This method computes the current density based on the Butler-Volmer equation,
    accounting for mass transfer limitations and concentration-dependent kinetics (j-eta relationship).
    -------- Input----
    eta : Overpotential (V) at the electrode surface.
    c : Concentration vector containing both reduced and oxidized species concentrations.
        For the cathode: [c_red (V4), c_ox (V5), c_H (H+)].
    electrode : Electrode type: "cathode".
    ------- Output----
    i_n : Current density (A/m2) per geometric electrode area.
    ------- Equation--
    The current density is computed as:
        i_n = a * F * k * (c_ox)^(1-alpha) * (c_red)^(alpha) * (c_H)^(2-2alpha)
              [(c_s_red/c_red) * exp((1-alpha)*f*η) - (c_s_ox/c_ox)*exp(-alpha*f*η)]
    where the mass transfer limitation of the protons is neglected.
    Pino-Muñoz, C. A.; Dewage, H. H.; Yufit, V.; Brandon, N. P. 
    A Unit Cell Model of a Regenerative Hydrogen-Vanadium Fuel Cell. 
    J. Electrochem. Soc. 2017, 164 (14), F1717. https://doi.org/10.1149/2.1431714jes.
    """
    F = self.F
    f  = self.F/(self.R * self.T)
    if electrode == "cathode":
        a      = self.cathode.specific_surface_area  # Cathode specific surface area [m2/m3]
        k_ca   = self.cathode.reaction_rate_constant # Cathode rate constant [m/s]
        alpha  = self.alpha_c                        # Transfer coefficient of reaction [-]
        NC     = self.mesh.cathode_mesh.n_x          # Number of nodes in cathode [-]
        c_V4  = c[:NC]                               # (V4+) [mol/m3]
        c_V5   = c[NC:2*NC]                          # (V5+) [mol/m3]
        c_H_plus    = c[2*NC:3*NC]                   # (H+)  [mol/m3]
        c_s_V4, c_s_V5, c_V4, c_V5, Psi3, Psi4, beta_0 = cathode_surface_concentrations(self, eta, c_V4, c_V5, c_H_plus) # Surface concentrations and mass transfer parameters
        i_0_ca  = F * k_ca * (c_V5)**(1-alpha) * (c_V4**(alpha)) * (c_H_plus**(2-2*alpha))                     # Exchange current density   [A/m2] (per unit area of electrode active surface)
        i_n     = a * i_0_ca * ((c_s_V4/c_V4)*np.exp((1-alpha)*f*eta)-(c_s_V5/c_V5)*np.exp((-alpha)*f*eta))    # Volumetric current density [A/m3] (per unit volume of electrode)
    else:
        print("Electrode not defined")
        i_n = 0
    return i_n
        
def BV_lumped(self, eta, c, electrode):
    """ Calculates the current density per geometric electrode area (A/m2) for the ACL 
    assuming a linear kinetics and low overpotential.
    Ye, M.; Pino-Muñoz, C. A.; Cannon, C. G.; Sun, Y.; Kucernak, A.; Klusener, P. A. A.; Brandon, N. 
    Understanding H2-X Redox Flow Battery Performance Through Modelling and Design of Experiments. 
    SSRN 2025. https://doi.org/10.2139/ssrn.5126397.
    """
    if electrode == "anode":
        R, F, T  = self.R, self.F, self.T
        n  = 2                            # Number of electrons transferred
        i_0 = self.anode.j_0              # Exchange current density [A/m2] 
        i_n = i_0 * (n * F * eta / (R*T)) # Current density [A/m2] per geometric electrode area
    else:
        i_n = 0 
    return i_n
