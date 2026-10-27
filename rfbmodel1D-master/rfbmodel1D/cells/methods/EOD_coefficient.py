import numpy as np

def EOD_coefficient(self, lam):
    """
    Electro-osmotic drag coefficient ζ_w(λ) as a function of membrane water content and temperature.
    Works on both scalars and NumPy arrays.
    (1) Weber, A. Z.; Newman, J. Transport in Polymer-Electrolyte Membranes II. Mathematical Model. Journal of The Electrochemical Society 2004, 151 (2).
    """
    lam = np.asarray(lam, dtype=float)
    lam = np.clip(lam, 0, 22)  # Ensure non-negative water content
    R = self.R 
    T = self.T
    zeta = 2.5 * lam / 22 * np.exp(4000/R * (1/303.15-1/T))  # Adjust for temperature
    return zeta