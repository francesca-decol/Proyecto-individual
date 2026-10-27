import numpy as np
class submesh:
    """ Creates a finite volume mesh for 1D problems.
    This class represents a computational mesh defined on the interval [0, 1].
    It provides grid points, discretized differential operators, and a method
    for calculating field averages."""
    def __init__(self, n_x, Uniform=True) -> None:
        # n_x   : Number of nodes in the x direction
        # D1    : First derivative operator
        # D2    : Second derivative operator
        self.n_x = n_x
        if not Uniform:
            # Cosine-clustered non-uniform grid
            #self.x_nodes = 0.5 * (1 - np.cos(np.linspace(0, np.pi, self.n_x + 1)))  # n+1 nodes from 0 to 1
            alpha = 1  # 0 = uniform, 1 = strong clustering
            xi = np.linspace(0, 1, n_x + 1)
            # Tangent maps fine->center, so invert it to get fine->boundaries
            self.x_nodes = 0.5 * (1 - np.tanh(alpha * (1 - 2*xi)) / np.tanh(alpha))
            self.x_grid  = 0.5 * (self.x_nodes[:-1] + self.x_nodes[1:])             # cell centers
            self.dx      = np.diff(self.x_nodes)                                    # cell widths
            self.D       = self.build_diffusion_non_uniform_operator()
        else:
            # Uniform grid
            self.x_nodes = np.linspace(0, 1, self.n_x + 1)                          # n+1 nodes from 0 to 1
            self.x_grid  = 0.5 * (self.x_nodes[:-1] + self.x_nodes[1:])             # cell centers
            self.dx      = np.diff(self.x_nodes)                                    # cell widths (all equal)
            self.D         = self.build_diffusion_operator()
        self.C_central = self.build_convection_central()
        self.C_up_pos  = self.build_convection_upwind_pos()
        self.C_up_neg  = self.build_convection_upwind_neg()
        self.S_left    = self.build_left_shift_matrix()
        self.S_right   = self.build_right_shift_matrix()

    def build_diffusion_operator(self):
        """Returns the finite-volume diffusion flux divergence matrix (uniform mesh)."""
        n_x = self.n_x # Number of grid points
        D1 = np.zeros((n_x, n_x))
        for i in range(1, n_x-1):
            D1[i, i-1] =  1
            D1[i, i  ] = -2
            D1[i, i+1] =  1
        return D1

    def build_diffusion_non_uniform_operator(self):
        """
        Returns the finite-volume diffusion flux divergence matrix for a non-uniform mesh.
        Uses dx (cell widths) and dx_face (distances between nodes).
        Assumes Dirichlet BCs (zero flux at boundaries).
        """
        n_x = self.n_x
        D1 = np.zeros((n_x, n_x))
        dx = self.dx  # cell widths (n_x,)
        dx_face = self.x_nodes  # node positions (n_x+1,)
        # Compute face distances
        dx_faces = dx_face[1:] - dx_face[:-1]  # (n_x,)
        for i in range(1, n_x-1):
            dx_w = dx_faces[i-1]   # west face (between i-1 and i)
            dx_e = dx_faces[i]     # east face (between i and i+1)
            D1[i, i-1] = 1/dx_w
            D1[i, i] = -(1/dx_w + 1/dx_e)
            D1[i, i+1] = 1/dx_e
        # Optionally handle BCs here (e.g., Dirichlet or Neumann)
        return D1
    
    def build_convection_central(self):
        """ Returns the matrix for the convection term using central differencing, for Pe<2"""
        n_x    = self.n_x
        C_central = np.zeros((n_x, n_x))
        for i in range(1, n_x-1):
            C_central[i,i-1] = -1/2
            C_central[i,i]   = 0
            C_central[i,i+1] = 1/2
        return C_central
        
    def build_convection_upwind_pos(self):
        """ Returns the matrix for the convection term using upwind scheme, for positive velocity, and Pe>2"""
        n_x    = self.n_x
        C_up_pos = np.zeros((n_x, n_x))
        for i in range(1, n_x-1):
            C_up_pos[i,i-1] = -1
            C_up_pos[i,i]   = 1
            C_up_pos[i,i+1] = 0
        return C_up_pos

    def build_convection_upwind_neg (self):
        """ Returns the matrix for the convection term using upwind scheme, for negative velocity, and Pe>2"""
        n_x    = self.n_x
        C_up_neg  = np.zeros((n_x, n_x))
        for i in range(1, n_x-1):
            C_up_neg [i,i-1] = 0
            C_up_neg [i,i]   = -1
            C_up_neg [i,i+1] = 1
        return C_up_neg 
    
    def build_left_shift_matrix(self):
        n_x = self.n_x
        S_left  = np.zeros((n_x, n_x))
        for i in range(1,n_x-1):
            S_left[i,i-1] = 1
        return S_left

    def build_right_shift_matrix(self):
        n_x = self.n_x
        S_right  = np.zeros((n_x, n_x))
        for i in range(1,n_x-1):
            S_right[i,i+1] = 1
        return S_right

    def dx_array(self, thickness=1.0):
        """
        Return an array of cell widths.

        Parameters
        - thickness: physical scaling factor for the computational domain (default 1.0).

        Returns
        - numpy array of length `n_x` containing each cell's physical width.
        """
        dx_raw = self.dx
        if np.isscalar(dx_raw):
            return np.full(self.n_x, dx_raw * thickness) # type: ignore
        return np.asarray(dx_raw) * thickness
    


    