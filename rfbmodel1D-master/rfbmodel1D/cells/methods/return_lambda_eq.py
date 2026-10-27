from rfbmodel1D.parameters.params import SimulationParams, Vars
import numpy as np
def return_lambda_eq(self, a):
    """
    Calculate equilibrium water content from water activity.
    Water activity includes saturation: a = RH + 2s, where RH is relative
    humidity and s is liquid saturation. The function provides continuous
    transition from vapor (a<1) through mixed phase (1≤a≤3) to fully liquid (a>3).
    Args:
        a: Water activity (dimensionless, typically 0 to 3+)
           - a < 1: vapor phase only (sorption isotherm)
           - 1 ≤ a ≤ 3: vapor + liquid saturation (linear transition)
           - a > 3: fully liquid
    Smooth blending of 0.05 around transitions
    Returns:
        lambda_eq: Equilibrium water content (dimensionless)
    Reference:
        Equation obtained from (1) Ye, M.; Pino-Muñoz, C. A.; Cannon, C. G.; Sun, Y.; Kucernak, A.; Klusener, P. A. A.; Brandon, N. 
        Understanding H2-X Redox Flow Battery Performance Through Modelling and Design of Experiments.
        SSRN 2025. https://doi.org/10.2139/ssrn.5126397.
    """
    T = self.T
    delta1 = delta2 = 0.05
    w1           = 0.5*(1 + np.tanh((a - 1) / delta1))   # 0 -> vapor branch, 1 -> mixed branch
    w2           = 0.5*(1 + np.tanh((a - 3) / delta2))   # 0 -> mixed branch, 1 -> liquid branch
    lambda_vapor = lambda_298K(a) + (lambda_363K(a)-lambda_298K(a))*(T-298)/(363-298)
    lambda_1     = lambda_298K(1) + (lambda_363K(1)-lambda_298K(1))*(T-298)/(363-298)
    lambda_mixed = lambda_1 + 4*(a - 1)
    lambda_liq   = 22.0
    lambda_eq    = (1-w1)*lambda_vapor + w1*((1-w2)*lambda_mixed + w2*lambda_liq)
    return lambda_eq

def return_lambda_cath_eq(self, a):
    """ Calculate equilibrium water content at the cathode interface.
    Polynomial fit to experimental data from [1].
    (1) Tang, Z.; Svoboda, R.; Lawton, J. S.; Aaron, D. S.; Papandrew, A. B.; Zawodzinski, T. A. Composition and Conductivity of Membranes Equilibrated with Solutions of Sulfuric Acid and Vanadyl Sulfate. J. Electrochem. Soc. 2013, 160 (9), F1040–F1047. https://doi.org/10.1149/2.083309jes.

    """
    lambda_eq = 51.09 * a**3 - 70.18 * a**2 + 40.41 * a + 0.14
    return lambda_eq

def return_lambda_corr(self, lam):
    """ Calculate corrected water content based on species uptake.
    The correction accounts for the reduction in water content due to the first hydration shell of vanadium uptake in the membrae
    (1) Elgammal, R. A.; Tang, Z.; Sun, C.-N.; Lawton, J.; Zawodzinski, T. A. Species Uptake and Mass Transport in Membranes for Vanadium Redox Flow Batteries. Electrochimica Acta 2017, 237, 1–11. https://doi.org/10.1016/j.electacta.2017.03.131.
    """
    c_V4 = self.bc[Vars.c_v4]["membrane"]["left"].value/1000 # mol/m3 to M
    c_V5 = self.bc[Vars.c_v5]["membrane"]["left"].value/1000 # mol/m3 to M
    # Calculate species uptake
    c_V4_uptake = c_V4 * 0.1229  # M V4 / M SO3-
    c_V5_uptake = c_V5 * 0.0952  # M V5 / M SO3-
    # Calculate correction factor based on species uptake
    n_hyd_V4 = 5  # Hydration number of V4
    n_hyd_V5 = 3  # Hydration number of V5
    # Correction
    lam_corr = lam - n_hyd_V4 * c_V4_uptake - n_hyd_V5 * c_V5_uptake
    return lam_corr

def return_water_activity_cathode(self, SOC):
    """ Calculate water activity at cathode interface based on state of charge (SOC) and T.
    Simplified linear relationship based on experimental data, with a small temperature dependence.
    Ye, M.; Pino-Muñoz, C. A.; Cannon, C. G.; Sun, Y.; Kucernak, A.; Klusener, P. A. A.; Brandon, N. 
    Understanding H2-X Redox Flow Battery Performance Through Modelling and Design of Experiments. 
    SSRN 2025. https://doi.org/10.2139/ssrn.5126397.
    """
    T           = self.T
    a_w_cathode = -0.0114 + -0.0254*SOC + 0.0020*T
    return a_w_cathode

def lambda_298K(a):
    """Calculate lambda at 298K for 0 <= a <= 1"""
    return (86.7 * a**5 - 181.3 * a**4 + 145.7 * a**3 - 49.1 * a**2 + 10.4 * a + 1.4)

def lambda_363K(a):
    """Calculate lambda at 363K for 0 <= a <= 1"""
    return (215.0 * a**5 - 480.9 * a**4 + 393.5 * a**3 - 135.9 * a**2 + 24.0 * a + 0.8)
