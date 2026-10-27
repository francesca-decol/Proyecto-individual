import numpy as np
from rfbmodel1D.parameters.params import SimulationParams, Vars

def compute_overpotential(self, Phi_el, Phi_io, c, electrode):
    """
    Compute the overpotential for each node in the cathode mesh (nondimensional).
    The overpotential is calculated as the difference between the electrode potential,
    the ionic potential, and the open-circuit potential (OCP).
    eta = Phi_el - Phi_io - OCP
    Parameters
    ----------
    Phi_el : array_like
        Electrode potential at each node in the cathode mesh.
    Phi_io : array_like
        Open-circuit (or reference) potential at each node in the cathode mesh.
    c : array_like
        Concentration array containing concentrations for species.
    Returns
    -------
    eta : ndarray
        Overpotential values at each node in the electrode mesh (shape: (NA,)).
        Stored as instance attribute self.electrode.eta.
    """
    R  = self.R
    T  = self.T
    f  = self.F / (R * T) 
    if electrode == "cathode":
        NC    = self.mesh.cathode_mesh.n_x # Number of nodes in cathode
        c_V4  = c[:NC]                # Concentration of VO^+2  [mol/m3]
        c_V5  = c[NC:2*NC]            # Concentration of VO_2^+ [mol/m3]
        c_H   = c[2*NC:3*NC]          # Concentration of H+     [mol/m3]
        # Overpotential (vectorized cathode OCP)
        OCP = self.compute_OCP([c_V4, c_V5, c_H], electrode="cathode")
        eta = Phi_el[:NC] - Phi_io[:NC] - OCP
        self.cathode.eta = eta 
        