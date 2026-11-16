"""
iiq3843_utils.py
================
Utilidades del curso IIQ3843 para usar rfbmodel1D sin pelear con el vector de estado.

Todas las funciones reciben un objeto `Cell` (o una subclase suya) y vectores de estado `y`
tal como los entrega `solve_dae`. Nada de este archivo cambia la física del modelo: solo
arma corridas, extrae perfiles y calcula cantidades derivadas (voltaje, OCV, flujos de agua).

Uso típico (desde la carpeta tutoriales/):

    from iiq3843_utils import configurar, construir_celda, integrar, perfiles
    configurar(T_C=32, NC=100, NM=60)
    cell, y0, yp0 = construir_celda(soc=0.75, modo="galvanostatico", valor=-500)
    sol = integrar(cell, y0, yp0, t_final=600)
    p = perfiles(cell, sol.y[:, -1], t=sol.t[-1])
"""
import numpy as np
from solve_dae.integrate import solve_dae

from rfbmodel1D.parameters.params import SimulationParams, Vars
from rfbmodel1D.cells.cell import Cell
from rfbmodel1D.cells.methods.initialize_dae_state import solve_algebraic, solve_differential


# ------------------------------------------------------------------------------
# 1. Configuración y construcción de la celda
# ------------------------------------------------------------------------------

def configurar(T_C=32.0, Q_ml_min=45.0, NC=100, NM=60):
    """Fija temperatura [°C], caudal de catolito [ml/min] y mallas ANTES de crear la celda.

    Los parámetros de rfbmodel1D son atributos de clase: un cambio aquí vale para toda la
    sesión de Python. Llame a esta función antes de cada `construir_celda`.
    """
    P = SimulationParams
    P.CellParams.T          = T_C + 273.15     # [K]
    P.CathodeParams.Q_total = Q_ml_min * 1e-6 / 60  # [m3/s]
    P.MeshParams.NC         = NC               # volúmenes de control en el cátodo
    P.MeshParams.NM         = NM               # volúmenes de control en la membrana
    return P


def construir_celda(soc, modo="galvanostatico", valor=-500.0, V_tank=0.5e-3,
                    c_V=1000.0, c_H=4963.0, clase=Cell):
    """Crea la celda y un estado inicial consistente (y0, yp0).

    soc    : estado de carga inicial del tanque [-]
    modo   : "galvanostatico" (valor = densidad de corriente [A/m2], negativa = descarga)
             o "potenciostatico" (valor = potencial del colector catódico [V] vs. ánodo)
    V_tank : volumen de catolito en el tanque [m3]
    c_V    : vanadio total [mol/m3];  c_H : H+ en el tanque [mol/m3]
    clase  : Cell o una subclase suya (así se enganchan las modificaciones de cada proyecto)
    """
    cell = clase(params=SimulationParams)
    _fijar_operacion(cell, modo, valor)
    c_tank = np.array([(1 - soc) * c_V, soc * c_V, c_H, c_H * 3 / 5, V_tank])
    y0, yp0 = cell.initialize_dae_state(c_tank, xtol=1e-6)
    return cell, y0, yp0


def _fijar_operacion(cell, modo, valor):
    if modo == "galvanostatico":
        cell.bc_mode = "galvanostatic"
        cell.modify_BC(Vars.phi_el, "cathode", "left", "galvanostatic", valor)
    elif modo == "potenciostatico":
        cell.bc_mode = "potentiostatic"
        cell.modify_BC(Vars.phi_el, "cathode", "left", "dirichlet", valor)
    else:
        raise ValueError("modo debe ser 'galvanostatico' o 'potenciostatico'")


def cambiar_operacion(cell, y, modo, valor):
    """Cambia la corriente (o el voltaje) a mitad de una simulación.

    Re-resuelve las incógnitas algebraicas y las derivadas para que (y, yp) vuelvan a ser
    consistentes con el nuevo valor. Devuelve (y, yp) listos para otra llamada a `integrar`.
    Sirve para encadenar carga y descarga (ciclos).
    """
    _fijar_operacion(cell, modo, valor)
    y_new  = solve_algebraic(cell, np.asarray(y, dtype=float).copy(), xtol=1e-6)
    yp_new = solve_differential(cell, y_new, xtol=1e-6)
    return y_new, yp_new


def tolerancias(cell):
    """Tolerancias absolutas, una por incógnita, en el orden del vector de estado."""
    Nc, Nm = cell.mesh.cathode_mesh.n_x, cell.mesh.membrane_mesh.n_x
    return np.concatenate([
        [1e-1, 1e-1, 1e-1, 1e-1, 1e-6],                     # tanque
        np.tile([1e-1, 1e-1, 1e-1, 1e-1, 1e-3, 1e-3], Nc),  # cátodo
        np.tile([1e-2, 1e-3], Nm),                          # membrana
        [1e-3, 1e-3, 1e-3, 1e-4, 1e-3, 1e-3, 1e-3],         # ánodo (ACL + AGDL)
        [1e-1],                                             # acoplamiento
    ])


def integrar(cell, y0, yp0, t_final, rtol=1e-6, eventos=None, n_salidas=None):
    """Integra el DAE completo entre 0 y t_final [s] con BDF. Devuelve el objeto de solve_dae.

    n_salidas: si se da, entrega la solución interpolada en ese número de instantes
    equiespaciados (solo para graficar). NO la use para integrar flujos en el tiempo: la
    interpolación no respeta las relaciones algebraicas del DAE y en el balance de agua de la
    membrana introduce ~8 % de error, contra 0,01 % con los pasos propios del solver.
    """
    kw = dict(method="BDF", atol=tolerancias(cell), rtol=rtol)
    if eventos is not None:
        kw["events"] = eventos
    if n_salidas is not None:
        kw["t_eval"] = np.linspace(0.0, float(t_final), int(n_salidas))
    sol = solve_dae(cell.dae_residual_full_cell, (0.0, float(t_final)), y0, yp0, **kw)
    if not sol.success:
        print("ADVERTENCIA: solve_dae no terminó con éxito:", sol.message)
    return sol


# ------------------------------------------------------------------------------
# 2. Extracción de resultados
# ------------------------------------------------------------------------------

def refrescar(cell, y, t=0.0):
    """Evalúa el residuo en y para actualizar los atributos que el modelo recalcula en cada
    llamada (eta del cátodo, kappa, i_io, condiciones de interfaz, eta_proton del ánodo).
    Llámela antes de leer esos atributos para un instante dado."""
    y = np.asarray(y, dtype=float)
    cell.dae_residual_full_cell(t, y, np.zeros_like(y))


def bloques(cell, y):
    """Separa y en sus bloques con nombre (vistas, no copias)."""
    return cell.split_state(np.asarray(y, dtype=float))


def perfiles(cell, y, t=0.0):
    """Todos los perfiles y valores concentrados en un instante.

    Posiciones: x_ca [m] desde el colector catódico (0) hacia la membrana (L_ca);
                x_m  [m] desde la cara cátodo/membrana (0) hacia el ánodo (L_m).
    """
    refrescar(cell, y, t)
    b  = bloques(cell, y)
    ca, me, an, tk = b["cathode"], b["membrane"], b["anode"], b["tank"]
    Nc = cell.mesh.cathode_mesh.n_x
    c_eta = np.concatenate([ca[:, 0], ca[:, 1], ca[:, 2]])
    j_ca  = cell.butler_volmer(cell.cathode.eta, c_eta, "cathode")   # [A/m3]
    return {
        "x_ca": cell.mesh.cathode_mesh.x_grid * cell.cathode.thickness,
        "dx_ca": cell.mesh.cathode_mesh.dx * cell.cathode.thickness,
        "c_V4": ca[:, 0].copy(), "c_V5": ca[:, 1].copy(),
        "c_H": ca[:, 2].copy(),  "c_HSO4": ca[:, 3].copy(),
        "phi_el_ca": ca[:, 4].copy(), "phi_io_ca": ca[:, 5].copy(),
        "eta_ca": np.array(cell.cathode.eta, dtype=float).copy(),     # [V]
        "j_ca": np.asarray(j_ca, dtype=float).reshape(Nc).copy(),     # [A/m3], >0 oxidación
        "kappa_ca": np.asarray(cell.cathode.kappa_io, dtype=float).copy(),
        "x_m": cell.mesh.membrane_mesh.x_grid * cell.membrane.thickness,
        "dx_m": cell.mesh.membrane_mesh.dx * cell.membrane.thickness,
        "lam": me[:, 0].copy(), "phi_io_m": me[:, 1].copy(),
        "kappa_m": np.asarray(cell.membrane.kappa_io, dtype=float).copy(),
        "tanque": {"c_V4": tk[0], "c_V5": tk[1], "c_H": tk[2], "c_HSO4": tk[3],
                   "V": tk[4], "soc": tk[1] / (tk[0] + tk[1])},
        "anodo": {"c_vw_acl": an[0], "c_H2_acl": an[1], "s_acl": an[2], "eta_acl": an[3],
                  "c_vw_agdl": an[4], "c_H2_agdl": an[5], "s_agdl": an[6],
                  "eta_proton": float(cell.eta_proton)},
        "i_io": float(cell.i_io),      # corriente iónica en la membrana [A/m2], <0 descarga
    }


def voltaje(cell, y):
    """Voltaje de celda [V] = Phi_el en el colector catódico (el ánodo electrónico es 0 V)."""
    if cell.bc_mode == "potentiostatic":
        return float(cell.bc[Vars.phi_el]["cathode"]["left"].value)
    return float(bloques(cell, y)["cathode"][0, 4])


def ocv(cell, y):
    """Potencial de circuito abierto [V] con la composición del tanque y del ánodo en y.
    Requiere haber llamado a `refrescar` en y (usa la actividad de agua de la interfaz)."""
    b  = bloques(cell, y)
    tk, an = b["tank"], b["anode"]
    E_ca = cell.compute_OCP(np.array([tk[0], tk[1], tk[2]]), "cathode")
    E_an = cell.compute_OCP(np.array([an[1], cell.membrane.c_f + 970]), "anode")
    return float(E_ca - E_an)


def series_temporales(cell, sol):
    """Voltaje, OCV, corriente iónica y SOC del tanque en cada paso de tiempo de `sol`."""
    out = {k: np.empty(sol.t.size) for k in ("V", "OCV", "i_io", "soc")}
    for k, t in enumerate(sol.t):
        y = sol.y[:, k]
        refrescar(cell, y, t)
        out["V"][k]    = voltaje(cell, y)
        out["OCV"][k]  = ocv(cell, y)
        out["i_io"][k] = cell.i_io
        out["soc"][k]  = y[1] / (y[0] + y[1])
    out["t"] = sol.t
    return out


def flujos_agua_membrana(cell, y, t=0.0):
    """Flujos de agua en la membrana, separados por mecanismo.

    Devuelve flujos en [mol H2O / m2 / s], positivos hacia el ánodo (+x):
      x_caras            posición de las caras interiores [m]
      N_dif, N_eod       difusión y arrastre electro-osmótico en las caras interiores
      N_cat              flujo en la cara cátodo/membrana (>0: agua entra a la membrana)
      N_an               flujo en la cara membrana/ánodo  (>0: agua sale hacia el ánodo)
    Reproduce exactamente las expresiones de `membrane_residual`
    (rfbmodel1D/cells/DAE/dae_residual_full_cell.py).
    """
    refrescar(cell, y, t)
    lam   = bloques(cell, y)["membrane"][:, 0]
    m, F  = cell.membrane, cell.F
    c_f   = m.c_f
    dxf   = cell._dx_face_m
    i_io  = cell.i_io
    model = m.diffusivity_model
    # Caras interiores
    D_f   = cell.membrane_water_diffusivity((lam[:-1] + lam[1:]) / 2, model)
    lam_up = lam[:-1] if i_io > 0 else lam[1:]          # aguas arriba del arrastre
    xi_f  = cell.EOD_coefficient(lam_up)
    N_dif = -D_f * (lam[1:] - lam[:-1]) / dxf            # [m/s] (en unidades de lambda)
    N_eod = xi_f * i_io / (F * c_f)
    # Cara cátodo/membrana (izquierda)
    a_left  = cell.bc[Vars.a]["membrane"]["left"].value
    lam_eqL = cell.return_lambda_corr(cell.return_lambda_cath_eq(a_left))
    f_v_ca  = lam[0] * m.V_w / (lam[0] * m.V_w + m.V_m)
    k_ca    = m.k_ca_lw * f_v_ca
    blend   = 0.5 * (1.0 + np.tanh((lam[0] - lam_eqL) / 0.1))
    N_cat   = (blend * k_ca * 10 + (1 - blend) * k_ca) * (lam_eqL - lam[0])
    # Cara membrana/ánodo (derecha)
    a_w_an  = cell.bc[Vars.a]["membrane"]["right"].value
    s       = cell.bc[Vars.s]["membrane"]["right"].value
    s_im    = cell.anode.s_im
    lam_eqR = max(cell.return_lambda_eq(a_w_an), 2.01)
    s_r     = np.logaddexp(0.0, 1e5 * (s - s_im)) / (1e5 * (1 - s_im))
    f_v_an  = lam[-1] * m.V_w / (lam[-1] * m.V_w + m.V_m)
    blend   = 0.5 * (1.0 + np.tanh((lam[-1] - lam_eqR) / 0.1))
    k_vw    = blend * m.k_desorp_vw * f_v_an * (1 - s) + (1 - blend) * m.k_sorp_vw * f_v_an * (1 - s)
    k_lw    = blend * m.k_desorp_lw * f_v_an * s_r + (1 - blend) * m.k_sorp_lw * f_v_an * s_r
    blend22 = 0.5 * (1.0 + np.tanh((lam[-1] - 22) / 0.25))
    N_an    = (k_vw + k_lw + 10 * k_lw * blend22) * (lam[-1] - lam_eqR)
    x_c = cell.mesh.membrane_mesh.x_nodes[1:-1] * m.thickness
    return {"x_caras": x_c, "N_dif": c_f * N_dif, "N_eod": c_f * N_eod,
            "N_cat": float(c_f * N_cat), "N_an": float(c_f * N_an),
            "agua_membrana": float(c_f * np.sum(lam * cell._dx_m))}   # [mol/m2]


def area_celda(cell):
    """Área geométrica activa [m2] (alto × ancho del fieltro)."""
    return cell.cathode.height * cell.cathode.width
