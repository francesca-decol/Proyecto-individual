from rfbmodel1D.meshes.grid_creation.submesh import submesh
class fv_mesh:
    """ Class which contains the finite volume mesh for three subdomains in cell class.
    """
    def __init__(self, MeshParams) -> None:
        self.anode_mesh    = submesh(MeshParams.NA)
        self.membrane_mesh = submesh(MeshParams.NM, Uniform = MeshParams.Uniform_M)
        self.cathode_mesh  = submesh(MeshParams.NC, Uniform = MeshParams.Uniform_C)