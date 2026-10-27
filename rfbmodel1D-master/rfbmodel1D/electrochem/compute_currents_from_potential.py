import numpy as np
from rfbmodel1D.parameters.params import SimulationParams, Vars

def compute_currents_from_potential(self, Phi_el, Phi_io, electrode):
    """ 
    Compute the electronic and ionic currents from the potential fields. 
    Considers finite volume discretisation, at the right side of each CV.
    First element is the right boundary of the first CV, last element is the right boundary of the last CV.
    Boundary condition is calculated only for the last node, which is the right boundary of the electrode.
    Parameters ----------
    Phi_el : array
        Electronic potential field (V) at the electrode.
    Phi_io : array
        Ionic potential field (V) at the electrode.
    electrode : str
        Electrode type: "cathode".
    Outputs -------
    i_el : array
        Electronic current density (A/m2) at the right side of each CV.
    i_io : array
        Ionic current density (A/m2) at the right side of each CV.
    """
    if electrode == "cathode":
        N = self.mesh.cathode_mesh.n_x
        dx_ca = np.asarray(self.mesh.cathode_mesh.dx) * self.cathode.thickness  # shape (N,)
        edges = np.concatenate([[0.0], np.cumsum(dx_ca)])
        x_center = 0.5 * (edges[:-1] + edges[1:])
        dx_face = x_center[1:] - x_center[:-1]  # shape (N-1,)
        # Electronic Current
        grad_phi_el = (Phi_el[1:] - Phi_el[:-1]) / dx_face
        i_el = -self.cathode.eff_conductivity * grad_phi_el
        bc_right_el = self.bc[Vars.phi_el]["cathode"]["right"]
        if bc_right_el.type == "dirichlet":
            i_el = np.append(i_el, -self.cathode.eff_conductivity * (Phi_el[-1] - Phi_el[-2]) / dx_ca[-1])
        elif bc_right_el.type == "neumann":
            i_el = np.append(i_el, -self.cathode.eff_conductivity * bc_right_el.value)
        # Ionic Current
        kappa = self.cathode.kappa_io
        # Compute harmonic mean for kappa at faces
        kappa = np.asarray(kappa)
        kappa_faces = 2 * kappa[:-1] * kappa[1:] / (kappa[:-1] + kappa[1:] + 1e-30)
        grad_phi_io = (Phi_io[1:] - Phi_io[:-1]) / dx_face
        i_io = -kappa_faces * grad_phi_io
        bc_right_io = self.bc[Vars.phi_io]["cathode"]["right"]
        # For the right boundary, use the last node value for kappa
        kappa_last = kappa[-1]
        if bc_right_io.type == "dirichlet":
            i_io = np.append(i_io, -kappa_last * (Phi_io[-1] - Phi_io[-2]) / dx_ca[-1])
        elif bc_right_io.type == "neumann":
            i_io = np.append(i_io, -kappa_last * bc_right_io.value)
    else:
        print("Not developed for this electrode")
        i_el, i_io = 0, 0
    return i_el, i_io