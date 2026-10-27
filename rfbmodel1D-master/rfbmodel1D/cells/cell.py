import matplotlib.pyplot as plt  
import numpy as np
from scipy.optimize import fsolve, least_squares, minimize
from scipy.optimize import approx_fprime
from scipy.sparse import csr_matrix
#Import functions created
from .DAE.water_transport_residual import water_transport_residual
from .DAE.dae_residual_full_cell import dae_residual_full_cell
# Import classes
from .subdomains.anode import Anode
from .subdomains.cathode import Cathode
from .subdomains.membrane import Membrane
from .subdomains.agdl import AGDL
from rfbmodel1D.meshes.grid_creation.fv_mesh import fv_mesh
from rfbmodel1D.meshes.grid_creation.submesh import submesh
from rfbmodel1D.parameters.params import boundary_condition
from rfbmodel1D.parameters.params import SimulationParams, Vars
#From folder
from .methods.initialize_properties import initialize_effective_diffusivities, initialize_effective_mobilities, initialize_cathode_viscosity, solve_velocity
from .methods.initialize_dae_state import initialize_dae_state
from .methods.membrane_water_diffusivity import membrane_water_diffusivity
from .methods.return_lambda_eq import return_lambda_eq, return_lambda_corr, return_lambda_cath_eq, return_water_activity_cathode
from .methods.update_kappa_io_m import update_kappa_io_m, update_kappa_io_cathode
from .methods.water_saturation_pressure import water_saturation_pressure
from .methods.EOD_coefficient import EOD_coefficient
from .methods.jac_matrix import _single_state_pattern, _representative_states, build_jac_sparsity
#From electrochem
from rfbmodel1D.electrochem.butler_volmer import butler_volmer, BV_lumped
from ..electrochem.compute_currents_from_potential import compute_currents_from_potential
from ..electrochem.compute_overpotential import compute_overpotential
from ..electrochem.compute_OCP import compute_OCP
from ..electrochem.decompose_overpotential import decompose_overpotential

class Cell:
    def __init__(self, params):
        self.anode           = Anode(params.AnodeParams)
        self.membrane        = Membrane(params.MembraneParams)
        self.cathode         = Cathode(params.CathodeParams)
        self.mesh            = fv_mesh(params.MeshParams)
        self.bc              = params.BCParams
        self.agdl            = AGDL(params.AGDL_Params)
        self.layout          = params.StateLayoutParams
        self.bc_mode         = params.CellParams.bc_mode
        self.vanadium_uptake = params.CellParams.vanadium_uptake
        # Properties
        self.T              = params.CellParams.T
        self.i              = 0                    # Current density [A/m2] #Positive for discharge, negative for charge. Goes from anode to cathode
        self.i_io           = 0                    # Ionic current density at membrane
        # Concentration profiles
        self.c  = []
        self.c0 = []
        # Potential profiles
        self.Phi_io = []
        self.Phi_el = []
        # Constants
        self.F   = params.CellParams.F     # Faraday constant [C/mol]
        self.R   = params.CellParams.R     # Universal gas constant [J/(mol·K)]
        self.K_KC = params.CellParams.K_KC # Kozeny–Carman constant []
        # Geometry
        self.d_f     = params.CellParams.d_f # Fiber mean diameter [m]
        self.sigma_BP     = params.CellParams.sigma_BP     # Conductivity of bipolar plate [S/m]
        self.sigma_CC     = params.CellParams.sigma_CC       # Conductivity of current collector [S/m]
        self.thickness_BP = params.CellParams.thickness_BP      # Thickness of bipolar plate [m]
        self.thickness_CC = params.CellParams.thickness_CC    # Thickness of current collector [m]
        # Kinetics
        self.alpha_c = params.CellParams.alpha_c
        self.dS_r   = params.CellParams.dS_r
        # Diffusion coefficients [m2/s]
        self.D_H2 = params.CellParams.D_H2
        self.D_vw = params.CellParams.D_vw
        self.D_V4 = params.CellParams.D_V4
        self.D_V5 = params.CellParams.D_V5
        self.D_H_plus     = params.CellParams.D_H_plus
        self.D_HSO4_minus = params.CellParams.D_HSO4_minus
        self.D_SO4_2minus = params.CellParams.D_SO4_2minus
        # Sulfate reaction
        self.k_d  = params.CellParams.k_d
        self.Beta = params.CellParams.Beta
        # Effective transport properties
        self.initialize_effective_diffusivities()
        self.initialize_effective_mobilities()
        self.initialize_cathode_viscosity()
        self.solve_velocity()
        self.p_sat = water_saturation_pressure(self)
        # Cathode Mesh
        self._dx_ca       = self.mesh.cathode_mesh.dx * self.cathode.thickness
        self._dx_face_ca  = (self.mesh.cathode_mesh.x_grid[1:] - self.mesh.cathode_mesh.x_grid[:-1]) * self.cathode.thickness
        self._dx_m        = self.mesh.membrane_mesh.dx * self.membrane.thickness
        self._dx_face_m   = (self.mesh.membrane_mesh.x_grid[1:] - self.mesh.membrane_mesh.x_grid[:-1]) * self.membrane.thickness
        self._Div_ca      = self.mesh.cathode_mesh.D / self.cathode.thickness  # divergence matrix
                
    def modify_BC(self, variable, subdomain, location, type, value):
        """ Creates boundary condition
        INPUTS
        type: "dirichlet" or "neumann", "robin"
        variable: Vars.key
        value: value of the BC
        subdomain: "anode", "membrane" or "cathode"
        location: "east"  "west" of the subdomain
        """
        self.bc[variable][subdomain][location] = boundary_condition(type, value)
    
    def get_boundary_flux_left(self, bc: boundary_condition, b):
        if bc.type == "neumann":
            b[0] += bc.value  # This is the total flux specified at the boundary
        return b
    def get_boundary_flux_right(self, bc: boundary_condition, b):
        if bc.type == "neumann":
            b[-1] += -bc.value  # type: ignore # This is the total flux specified at the boundary
        return b
    
    def apply_diffusion_left_BC_vec(self, c, bc, D, dx):
        bc_type  = bc.type
        bc_value = bc.value
        c_node   = c[0]
        if bc_type == "dirichlet":
            N = -D * (c_node-bc_value) / (dx[0]/2)
        else:  # Does nothing. bc_type == "neumann"
            N = 0
        return N
    def apply_diffusion_right_BC_vec(self, c, bc, D, dx):
        bc_type  = bc.type
        bc_value = bc.value
        c_node   = c[-1]
        if bc_type == "dirichlet":
            N = -D * (bc_value-c_node) / (dx[-1]/2)
        else:  # Does nothing. bc_type == "neumann"
            N = 0
        return N
    
    def apply_migration_operator(self, l, d, u, Phi_io):
        """Apply tridiagonal migration operator to Phi_io without building M."""
        mig = np.zeros_like(Phi_io)
        mig[1:-1] = l[1:-1] * Phi_io[:-2] + d[1:-1] * Phi_io[1:-1] + u[1:-1] * Phi_io[2:]
        return mig

    def apply_migration_left_electrode_BC_vec(self, mig, b, bc_c, bc_phi, c, Phi_io, dx, dx_face):
        """Apply left migration BC directly on migration vector and source term."""
        bc_type_c = bc_c.type
        bc_value_c = bc_c.value
        bc_type_phi = bc_phi.type
        bc_value_phi = bc_phi.value
        dx_e = dx_face[0]  
        if bc_type_c == "dirichlet": # Concentration BC is given
            c_w = bc_value_c
            c_e = (dx[1] * c[0] + dx[0] * c[1]) / (dx[0] + dx[1])
            if bc_type_phi == "neumann":     # dPhi/dx is given
               mig[0] = c_e * (Phi_io[1]-Phi_io[0])/dx_e - c_w * bc_value_phi 
            if bc_type_phi == "potentiostatic":     # dPhi/dx is given
               dPhi   = bc_value_phi / self.cathode.kappa_io[0] # type: ignore
               mig[0] = c_e * (Phi_io[1]-Phi_io[0])/dx_e - c_w * dPhi 
            elif bc_type_phi == "dirichlet": # Phi_io BC is given
                mig[0] = c_e * (Phi_io[1]-Phi_io[0])/dx_e - c_w * (Phi_io[0]-bc_value_phi)/(dx[0]/2)
        elif bc_type_c == "neumann": # Total flux is given (Generally N=0)
            if bc_type_phi == "neumann":     # dPhi/dx is given
                N_w    =  c[0]*bc_value_phi*dx[0]
                mig[0] =  ((c[0]+c[1])/2)*(Phi_io[1]-Phi_io[0])/(dx_e) 
            elif bc_type_phi == "dirichlet": # Phi_io BC is given
                N_w    = c[0]*(Phi_io[0]-bc_value_phi)*(2) 
                mig[0] = ((c[0]+c[1]))/2*(Phi_io[1]-Phi_io[0])/(dx_e)
            if bc_type_phi == "galvanostatic":     # i is given
                N_w    =  c[0]*bc_value_phi*dx[0]
                mig[0] =  ((c[0]+c[1])/2)*(Phi_io[1]-Phi_io[0])/(dx_e) 
        return mig, b

    def apply_migration_right_electrode_BC_vec(self, mig, b, bc_c, bc_phi, c, Phi_io, dx, dx_face):
        """Apply right migration BC directly on migration vector and source term."""
        bc_type_c = bc_c.type
        bc_value_c = bc_c.value
        bc_type_phi = bc_phi.type
        bc_value_phi = bc_phi.value
        # Distance between last two nodes (west face)
        dx_w = dx_face[-1]
        if bc_type_c == "dirichlet":  # Concentration BC is given
            # Distance-weighted average for non-uniform grid
            c_w = (dx[-1] * c[-2] + dx[-2] * c[-1]) / (dx[-2] + dx[-1])
            c_e = bc_value_c
            if bc_type_phi == "neumann":  # dPhi/dx is given
                mig[-1] = c_e * bc_value_phi - c_w * (Phi_io[-1] - Phi_io[-2]) / dx_w
            if bc_type_phi == "galvanostatic":  # i is given
                kappa   = self.cathode.kappa_io[-1] # type: ignore
                dPhi    = bc_value_phi / kappa
                mig[-1] = c_e * dPhi - c_w * (Phi_io[-1] - Phi_io[-2]) / dx_w
            elif bc_type_phi == "dirichlet":  # Phi_io BC is given
                mig[-1] = c_e * (Phi_io[-1] - bc_value_phi) / (dx[-1]/2) - c_w * (Phi_io[-1] - Phi_io[-2]) / dx_w
        elif bc_type_c == "neumann":  # Total flux is given
            if bc_type_phi == "neumann":
                # N_e = c[-1] * bc_value_phi * dx[-1] 
                mig[-1] = -((c[-1] + c[-2]) / 2) * (Phi_io[-1] - Phi_io[-2]) / dx_w
            elif bc_type_phi == "dirichlet":
                # N_e = c[-1] * (bc_value_phi - Phi_io[-1]) * 2
                mig[-1] = -((c[-1] + c[-2]) / 2) * (Phi_io[-1] - Phi_io[-2]) / dx_w
            elif bc_type_phi == "galvanostatic":
                mig[-1] = -((c[-1] + c[-2]) / 2) * (Phi_io[-1] - Phi_io[-2]) / dx_w
        return mig, b

    def return_migration_elements(self, c, dx_face): 
        c = np.asarray(c)
        n = c.size
        l = np.empty(n, dtype=c.dtype)
        d = np.empty(n, dtype=c.dtype)
        u = np.empty(n, dtype=c.dtype)
        l[0] = 0
        d[0] = 0
        u[0] = 0
        l[-1] = 0
        d[-1] = 0
        u[-1] = 0
        c_im1 = c[:-2]
        c_i = c[1:-1]
        c_ip1 = c[2:]
        l[1:-1] = 0.5 * (c_im1 + c_i) / dx_face[:-1]
        d[1:-1] = -0.5 * ((c_im1 + c_i)/dx_face[0:-1] + (c_i + c_ip1)/dx_face[1:])
        u[1:-1] = 0.5 * (c_i + c_ip1) / dx_face[1:]
        return l, d, u
    
    def apply_electronic_left_BC(self, Div, b, bc, sigma, dx, dx_face):
        bc_type  = bc.type
        bc_value = bc.value

        if bc_type == "dirichlet":
            Div[0,0] = -1/dx_face[0] - 1/(dx[0]/2)
            Div[0,1] = 1/dx_face[0]
            b[0]    += +2 * bc_value * sigma / dx[0]
        elif bc_type == "neumann":
            Div[0,0] =  -1/dx_face[0]
            Div[0,1] = 1/dx_face[0]
            b[0]    += -sigma * bc_value
        elif bc_type == "galvanostatic":
            Div[0,0] = -1/dx_face[0]
            Div[0,1] = 1/dx_face[0]
            b[0]    +=  bc_value    
        return Div, b
    
    def apply_electronic_right_BC(self, Div, b, bc, sigma, dx, dx_face):
        bc_type  = bc.type
        bc_value = bc.value
        if bc_type == "dirichlet":
            Div[-1,-1] = -1/dx_face[-1] - 1/(dx[-1]/2)
            Div[-1,-2] = 1/dx_face[-1]
            b[-1]    += + 2 * bc_value * sigma / dx[-1]
        elif bc_type == "neumann":
            Div[-1,-1] = -1/dx_face[-1]
            Div[-1,-2] =  1/dx_face[-1]
            b[-1]    += - sigma * bc_value
        elif bc_type == "galvanostatic":
            Div[-1,-1] = -1/dx_face[-1]
            Div[-1,-2] = 1/dx_face[-1]
            b[-1]    +=  bc_value
        else:
            print("error")
        return Div, b
    
    def apply_ionic_right_BC_vec(self, Phi_io, bc, kappa, dx):
        bc_type  = bc.type
        bc_value = bc.value
        if bc_type == "dirichlet": # Dirichlet BC for ionic potential
            flux_right = -kappa[-1]*(bc_value-Phi_io[-1])/(dx[-1]/2)
        elif bc_type == "neumann": # Neumann BC for ionic potential (gradient dPhi/dx)
            flux_right = -kappa[-1]*bc_value
        elif bc_type == "galvanostatic":
            flux_right = bc_value
        else:
            print("error")
            flux_right = 0
        return flux_right

    def apply_ionic_left_BC_vec(self, Phi_io, bc, kappa, dx):
        bc_type  = bc.type
        bc_value = bc.value
        if bc_type == "dirichlet": # Dirichlet BC for ionic potential
            flux_left = -kappa[0]*(Phi_io[0]-bc_value)/(dx[0]/2)
        elif bc_type == "neumann": # Neumann BC for ionic potential (gradient dPhi/dx)
            flux_left = -kappa[0]*bc_value
        elif bc_type == "galvanostatic":
            flux_left = bc_value
        else:
            print("error")
            flux_left = 0
        return flux_left

    def apply_ionic_left_BC(self, Div, b, bc, kappa, dx, dx_face):
        bc_type  = bc.type
        bc_value = bc.value
        if bc_type == "dirichlet": # Dirichlet BC for ionic potential
            Div[0,0] = -1/dx_face[0] -1/(dx[0]/2)
            Div[0,1] = 1/dx_face[0]
            b[0]    += 2 * bc_value * kappa[0] / dx[0]
        elif bc_type == "neumann": # Neumann BC for ionic potential (gradient dPhi/dx)
            Div[0,0] =  -1/dx_face[0]
            Div[0,1] = 1/dx_face[0]
            b[0]    += -kappa[0] * bc_value 
        elif bc_type == "galvanostatic":
            Div[0,0] =  -1/dx_face[0]
            Div[0,1] = 1/dx_face[0]
            b[0]    +=  bc_value 
        else:
            print("error")
        return Div, b
    
    def apply_ionic_right_BC(self, Div, b, bc, kappa, dx, dx_face):
        bc_type  = bc.type
        bc_value = bc.value
        if bc_type == "dirichlet":  # Phi is given
            Div[-1,-1] = -1/dx_face[-1] - 1/(dx[-1]/2)
            Div[-1,-2] = 1/dx_face[-1]
            b[-1]    += 2 * bc_value * kappa[-1] / dx[-1]
        elif bc_type == "neumann":  # dPhi is given
            Div[-1,-1] = -1/dx_face[-1]
            Div[-1,-2] =  1/dx_face[-1]
            b[-1]    += - kappa[-1] * bc_value
        elif bc_type == "galvanostatic":
            Div[-1,-1] = -1/dx_face[-1]
            Div[-1,-2] =  1/dx_face[-1]
            b[-1]     +=  bc_value
        else:
            print("error")
        return Div, b

    def initialize_dae_state(self, x, xtol=1e-6):
        """Initialize dae state for complete cell """
        u0, du0 = initialize_dae_state(self, x, xtol=1e-6)
        return u0, du0
    
    def butler_volmer(self, eta, c, electrode):
        j_n = butler_volmer(self, eta, c, electrode)
        return j_n
    
    def compute_overpotential(self, Phi_el, Phi_io, c, electrode):
        eta = compute_overpotential(self, Phi_el, Phi_io, c, electrode)
        return eta

    def decompose_overpotential(self, y):
        return decompose_overpotential(self, y)
    
    def dae_residual_full_cell(self, t, y, yp):
        f = dae_residual_full_cell(self, t, y, yp)
        return f
    
    def initialize_effective_diffusivities(self):
        initialize_effective_diffusivities(self)

    def initialize_effective_mobilities(self):
        initialize_effective_mobilities(self)

    def return_mobility(self,D):
        return D / (self.R * self.T)  # Mobility in the electrode [m2/s]
    def compute_currents_from_potential(self, Phi_el, Phi_io, electrode):
        i_el, i_io = compute_currents_from_potential(self, Phi_el, Phi_io, electrode)
        return i_el, i_io
    def membrane_water_diffusivity(self, lambda_w, model="piecewise"):
        D = membrane_water_diffusivity(self, lambda_w, model=model)
        return D
    
    def return_water_activity_cathode(self, SOC):
        a = return_water_activity_cathode(self, SOC)
        return a

    def return_lambda_eq(self, a):
        """For vapor-liquid interfaces"""
        lambda_eq = return_lambda_eq(self, a)
        return lambda_eq
    
    def return_lambda_cath_eq(self, a):
        """For cathode interface"""
        lambda_eq = return_lambda_cath_eq(self, a)
        return lambda_eq
    
    def return_lambda_corr(self, lambda_eq):
        lambda_corr = return_lambda_corr(self, lambda_eq)
        return lambda_corr
    
    def water_transport_residual(self, t, u, du, i_io, lam_eq_left, lam_eq_right):
        f = water_transport_residual(self, t, u, du, i_io,
                                     lam_eq_left, lam_eq_right)
        return f
    
    def update_kappa_io_m(self, lambda_w, Phi_io_membrane):
        update_kappa_io_m(self, lambda_w, Phi_io_membrane)
    
    def update_kappa_io_cathode(self, y_cathode):
        update_kappa_io_cathode(self, y_cathode)

    def BV_lumped(self, eta, c, electrode):
        j_n = BV_lumped(self, eta, c, electrode)
        return j_n  
    
    def capillary_pressure(self, s_r, domain):
        "Using Leverett J-function to calculate capillary pressure"
        "Goshtasbi, A.; Pence, B. L.; Ersal, T. Computationally Efficient Pseudo-2D Non-Isothermal Modeling of Polymer Electrolyte Membrane Fuel Cells with Two-Phase Phenomena. J. Electrochem. Soc. 2016, 163 (13), F1412–F1432. https://doi.org/10.1149/2.0871613jes."
        porosity = 0
        K_lw     = 0 
        theta_c  = 0
        if domain == "acl":
            porosity = self.anode.porosity
            K_lw     = self.anode.permeability
            theta_c  = self.anode.contact_angle
        elif domain == "agdl":
            porosity = self.agdl.porosity
            K_lw     = self.agdl.permeability
            theta_c  = self.agdl.contact_angle
        sigma_s = - 0.0001676*self.T + 0.1218 # Surface tension of water [N/m]
        p_cap   = sigma_s * np.cos(np.radians(theta_c)) * np.sqrt(porosity/K_lw) * (1.417*s_r - 2.12*s_r**2 + 1.262*s_r**3) # Capillary pressure [Pa]
        return p_cap
    
    def water_saturation_pressure(self):
        p_sat = water_saturation_pressure(self)
        return p_sat

    def compute_OCP(self, c, electrode):
        OCP = compute_OCP(self, c, electrode)
        return OCP
    
    def EOD_coefficient(self, lam):
        EOD = EOD_coefficient(self, lam)
        return EOD
    
    def initialize_cathode_viscosity(self):
        initialize_cathode_viscosity(self)

    def solve_velocity(self):
        solve_velocity(self)

    def build_jac_sparsity(self):
        return build_jac_sparsity(self)
    def _single_state_pattern(self, y, yp, t, eps, tol):
        return _single_state_pattern(self, y, yp, t, eps, tol)
    def _representative_states(self, base_tanks, n_jitter, jitter, seed):
        return _representative_states(self, base_tanks, n_jitter, jitter, seed)

    def compute_jac_sparsity(self, y0, yp0, t=0.0, eps=1e-7, tol=0):
        """Return (sparsity_y, sparsity_yp) for use as jac_sparsity in solve_dae."""
        res = lambda y, yp: self.dae_residual_full_cell(t, y, yp)
        J_y  = approx_fprime(y0.copy(),  lambda y:  res(y,  yp0), eps)
        J_yp = approx_fprime(yp0.copy(), lambda yp: res(y0, yp),  eps)
        return csr_matrix((np.abs(J_y) > tol).astype(int)), csr_matrix((np.abs(J_yp) > tol).astype(int)) #type: ignore

    def split_state(self, y):
        Nc   = self.mesh.cathode_mesh.n_x
        Nm   = self.mesh.membrane_mesh.n_x
        nv_t = self.layout.n_vars_tank
        nv_c = self.layout.n_vars_cathode
        nv_m = self.layout.n_vars_membrane
        nv_a = self.layout.n_vars_anode
        i0 = nv_t
        i1 = i0 + Nc * nv_c
        i2 = i1 + Nm * nv_m
        i3 = i2 + nv_a
        return {
            "tank":     y[:i0],
            "cathode":  y[i0:i1].reshape(Nc, nv_c),
            "membrane": y[i1:i2].reshape(Nm, nv_m),
            "anode":    y[i2:i3],
            "coupling": y[i3:],
        }


