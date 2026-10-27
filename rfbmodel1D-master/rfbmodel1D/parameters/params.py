from dataclasses import dataclass, field
from collections import defaultdict
from typing import Union, Tuple
import numpy as np
# ---------------------------------------------------------
# Cell-level parameters (global electrochemical properties)
# ---------------------------------------------------------
@dataclass
class cellparams:
    bc_mode         = "potentiostatic" # "galvanostatic" or "potentiostatic"
    vanadium_uptake = False            # Whether to include vanadium-uptake correction to proton mobility in the membrane
    i: float        = 0        # Current density [A/m2]
    T: float        = 322         # Temperature
    d_f: float      = 1.76e-5     # Fiber diameter
    alpha_c: float  = 0.5        # Cathodic transfer coefficient
    # Diffusion coefficients
    D_V4: float     = 3.9e-10
    D_V5: float     = 3.9e-10
    D_H_plus: float = 9.3e-9
    D_HSO4_minus: float = 1.3e-9
    D_SO4_2minus: float = 1.1e-9
    # Thermodynamic parameters
    dS_r: float = -86.9 # Entropic coefficient [J/mol/K]
    # Gas diffusion
    D_H2: float     = 1.055e-4           # Diffusion coefficient of H2  [m2/s]
    D_vw: float     = 1.055e-4           # Diffusion coefficient of water vapor [m2/s]
    # Sulfate reaction constants
    k_d: float      = 1.0e4
    Beta: float     = 0.25
    # Electrochemical constants
    F: float        = 96485.33212  # Faraday constant
    R: float        = 8.314        # Ideal gas constant
    K_KC: float     = 49.5         # Kozeny–Carman constant, calibrated to match CFD-resolved in-plane felt velocity (KKC=93 for 1.087 mm/s at 20 ml/min) (KKC=49.5 for same Q_cath)
    # Conductivity parameters
    sigma_BP        = 66000     # Conductivity of bipolar plate [S/m]
    sigma_CC        = 6e7       # Conductivity of current collector [S/m]
    thickness_BP    = 1e-4      # Thickness of bipolar plate [m]
    thickness_CC    = 0.0127    # Thickness of current collector [m]
# ---------------------------------------------------------
# Anode domain parameters
# ---------------------------------------------------------
@dataclass
class anodeparams:
    thickness: float = 1.9e-5  # Anode thickness [m]
    width: float = 0.05        # Channel width [m]
    # Electrode structural properties
    porosity: float = 0.4
    ionomer_volume_fraction: float = 0.2
    specific_surface_area: float = 3e4
    conductivity: float = 240   # [S/m]
    permeability: float = 1e-13 # [m2]
    contact_angle: float = 105         # Contact angle for water in ACL [degrees]
    # Kinetics
    reaction_rate_constant: float = 3e-9
    j_0: float = 12615 # A/m2 at 32°C
    # Water transport parameters
    rho_lw: float = 1000             # Density of liquid water [kg/m3]
    MW_w: float = 0.01801528         # Molecular weight of water [kg/mol]
    # Initial species concentrations
    c_V2_0: float = 950
    c_V3_0: float = 50
    c_H_plus_0: float = 3967
    c_HSO4_minus_0: float = 5989
    c_SO4_2minus_0: float = 16.6
    # Condensation/evaporation kinetics
    k_cond: float = 100  # Condensation rate constant [1/s]
    k_evap: float = 100  # Evaporation rate constant  [1/s/atm]
    # Immobile water in the anode (e.g., due to ionomer)
    s_im : float = 0.12  # Immobile saturation (volume fraction of pore space occupied by immobile water)
# ---------------------------------------------------------
# Membrane domain parameters
# ---------------------------------------------------------
@dataclass
class membraneparams:
    thickness: float  = 127e-6   # Membrane thickness [m]
    c_f: float        = 1390     # Fixed charge concentration [mol/m3]
    z_f: int          = -1       # Charge valence
    # Species diffusion coefficients
    D_V4: float = 5e-12
    D_V5: float = 1.17e-12
    D_H_plus: float = 3.35e-9
    D_HSO4_minus: float = 4e-11
    # Volume
    V_m: float = 5.5e-4   # Dry membrane molar volume [m3/mol]
    V_w: float = 1.809e-5 # Liquid water molar volume [m3/mol]
    # Scalling factor for diffusivity
    sf_D: float =  1#0.5
    # Water diffusivity model, ("bimodal", "trimodal", "piecewise", "motupally", "motupally_2024", "hwang", "exp")
    diffusivity_model: str = "hwang"
    # Anode side mass transfer (m/s)
    k_sorp_vw, k_desorp_vw, k_sorp_lw, k_desorp_lw = np.array([1.8969e-05, 9.8691e-05, 5e-5, 5e-4])  # k_vw_sorp, k_vw_desorp, k_lw_sorp, k_lw_desorp
    # Cathode side mass transfer
    k_ca_lw = 1e-3 # Mass transfer coefficient for liquid water at cathode interface (m/s)
# ---------------------------------------------------------
# Cathode domain parameters
# ---------------------------------------------------------
@dataclass
class cathodeparams:
    # Electrode geometry
    thickness: float = 0.0046       # Cathode thickness [m]
    width: float     = 0.05         # Cathode width [m]
    height: float    = 0.05         # Cathode height [m]
    # Channel geometry 
    channel_length: float = 0.56    # Transverse length of cathode channel [m]
    channel_width: float  = 0.8e-3  # Channel width [m]
    channel_depth: float  = 1e-3    # Channel height [m]
    # Channel flow parameters
    Q_total: float = 7.5e-7         # Total volumetric flow rate in the cathode [m3/s]
    # Electrode structural properties
    porosity: float               = 0.94     # Porosity of the cathode
    #specific_surface_area: float  = 1.44e5   # [m2/m3] (Pino et al. 2017)
    specific_surface_area: float  = 3e4   # [m2/m3] (Gandomi)
    conductivity: float           = 4400     # [S/m]
    reaction_rate_constant: float = 2.24e-6/5000  # [m/s] 
    # Electrolyte properties (catholyte)
    rho_ca: float = 1350                     # Density of catholyte [kg/m3] 
    # Channel pressure drop parameters
    n_bend: int      = 20                    # Number of bends in the channel
    zeta_bend: float = 1.1                   # Pressure drop coefficient for each bend
    n_channel: int   = 3                     # Number of parallel channels in the cathode
    # Through plane velocity
    u_tp: float      = 0                     # Through-plane velocity in the cathode [m/s] 
    F_y:float        = 25.0                  # Fitting parameter for cathode OCV (activity coefficient)
# ---------------------------------------------------------
# AGDL domain parameters
# ---------------------------------------------------------
@dataclass
class agdlparams:
    thickness: float = 171e-6     # AGDL thickness [m]
    width: float = 50e-3          # AGDL width [m]
    height: float = 50e-3         # AGDL height [m]
    # Channel geometry
    channel_width: float  = 0.8e-3 # Channel width [m]
    channel_height: float = 1e-3  # Channel height [m]
    # structural properties
    porosity: float     = 0.6
    permeability: float = 1e-12   # [m2]
    conductivity: float = 8700    # [S/m]
    contact_angle: float = 130    # Contact angle for water in AGDL [degrees]
    # Diffusion coefficients
    D_H2: float = 1.055e-4        # Diffusion coefficient of H2 in AGDL [m2/s]
    D_vw: float = 1.055e-4        # Diffusion coefficient of water vapor in AGDL [m2/s]
# ---------------------------------------------------------
# Mesh parameters
# ---------------------------------------------------------
@dataclass
class meshparams:
    NA: float = 1   # Number of nodes in Anode
    NM: float = 50 # Number of nodes in Membrane
    Uniform_M = True
    NC: float = 150 # Number of nodes in Cathode
    Uniform_C = False
# ---------------------------------------------------------
# Variable per domain
# ---------------------------------------------------------
@dataclass
class state_layout_params:
    n_vars_tank:    int = 5  # V4, V5, H+, HSO4, V  
    n_vars_cathode: int = 6  # V4, V5, H+, HSO4 Phi_el, Phi_io
    n_vars_membrane:int = 2  # lam, Phi_io
    n_vars_anode:   int = 7  # acl[c_vw, c_H, s, eta] agdl[c_vw, c_H, s]
    n_vars_coupling: int = 1 # c_H_ca_BC
# ---------------------------------------------------------
# Variable names
# ---------------------------------------------------------
from enum import Enum
class Vars(Enum):
    """ Create a Enum with all the variables of the system.
    9 variables for DAE. Everything with no capital letters"""
    c_v4   = "c_V4"
    c_v5   = "c_V5"
    c_h    = "c_H_plus"
    c_hso4 = "c_HSO4_minus"
    c_so4  = "c_SO4_2minus"
    phi_el = "Phi_el"
    phi_io = "Phi_io"
    a      = "water_activity"  # Related to RH. a=1 for RH=100%, a=0 for RH=0%
    s_r    = "reduced saturation"
    c_h2   = "c_H2"
    s      = "saturation"


# ---------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------
@dataclass
class boundary_condition:
    """
    Boundary condition definition.
    type  : "dirichlet" / "neumann" / "robin" / "ionic_current"
    value :
        - float for dirichlet, neumann, ionic_current
        - (alpha, beta, gamma) for robin
    """
    type: str
    value: Union[float, Tuple[float, float, float]]

bcs = defaultdict(               # var
        lambda: defaultdict(     # domain
            dict                 # side -> BoundaryCondition 
        ))
# Dict
bcs[Vars.c_v4]["cathode"]["left"]     = boundary_condition("dirichlet", 50)
bcs[Vars.c_v4]["cathode"]["right"]    = boundary_condition("neumann", 0)

bcs[Vars.c_v5]["cathode"]["left"]     = boundary_condition("dirichlet", 950)
bcs[Vars.c_v5]["cathode"]["right"]    = boundary_condition("neumann", 0)

bcs[Vars.c_h]["cathode"]["left"]      = boundary_condition("dirichlet", 6500)
bcs[Vars.c_h]["cathode"]["right"]     = boundary_condition("neumann", 0)

bcs[Vars.c_hso4]["cathode"]["left"]   = boundary_condition("dirichlet", 4000)
bcs[Vars.c_hso4]["cathode"]["right"]  = boundary_condition("neumann", 0)
bcs[Vars.c_hso4]["membrane"]["left"]  = boundary_condition("dirichlet", 970)

bcs[Vars.a]["membrane"]["left"]       = boundary_condition("dirichlet", 3)
bcs[Vars.a]["membrane"]["right"]      = boundary_condition("dirichlet", 0.8)

bcs[Vars.a]["anode"]["right"]         = boundary_condition("dirichlet", 1) # RH in anode channel
bcs[Vars.s_r]["anode"]["right"]       = boundary_condition("dirichlet", 0) # Sat in anode channel 
bcs[Vars.c_h2]["anode"]["right"]      = boundary_condition("dirichlet", 42)# H2 concentration in anode channel [mol/m3]

# Electronic Potential
# "dirichlet" / "neumann"
# For neumann BC, value is the gradient of the potential (dPhi/dx) at the boundary. For dirichlet BC, value is the potential itself.
bcs[Vars.phi_el]["cathode"]["left"]   = boundary_condition("dirichlet", 0.7) # Potentiostatic BC [V]
bcs[Vars.phi_el]["cathode"]["right"]  = boundary_condition("neumann", 0)

# Ionic Potential
# "dirichlet" / "neumann" 
# For neumann BC, value is the gradient of the potential (dPhi/dx) at the boundary. For dirichlet BC, value is the potential itself.
bcs[Vars.phi_io]["cathode"]["left"]   = boundary_condition("neumann", 0)

##
bcs[Vars.phi_io]["cathode"]["right"]  = boundary_condition("neumann", 0) # Should be the current density of the cell divided by the ionic conductivity of the cathode electrolyte.
bcs[Vars.phi_io]["anode"]["left"]     = boundary_condition("neumann", 0) # Should be the current density of the cell divided by the ionic conductivity of the anode electrolyte.

# ---------------------------------------------------------
# Simulation parameters SuperClass
# ---------------------------------------------------------
@dataclass
class SimulationParams:
    CellParams = cellparams
    AnodeParams = anodeparams
    MembraneParams = membraneparams
    CathodeParams = cathodeparams
    MeshParams = meshparams
    BCParams   = bcs
    AGDL_Params = agdlparams
    StateLayoutParams = state_layout_params

# Parameter fitting
# 
# In rfbmodel1D/parameters/params.py
D_params = [1.00000000e-11, 1.00000000e-11, 2.00000000e-10, 6.52071151e+00, 1.84887533e+00, 1.75629698e+01, 8.56748644e-01, 5.29543335e-01]
k_params = np.array([4.89690000e-06, 1.75706153e-04, 4.00000000e-04, 4.15300000e-04])  # k_vw_sorp, k_vw_desorp, k_lw_sorp, k_lw_desorp

 