import numpy as np
from scipy.interpolate import CubicSpline
from rfbmodel1D.parameters.params import D_params, membraneparams 
# Module-level cache: (model_key: str, T: float) -> CubicSpline
_D_spline_cache: dict = {}
_diffusivity_grid = 100

def D_hwang_raw(lam, params):
    """
    Fitted tri-regime model (Hwang desorption data for 119 µm membrane). Shared by _build_hwang_spline
    (production) and scripts/functions/membrane_model.py (fitting) so both always use the exact same formula.
    phase1: linear rise
    phase2: plateau
    phase3: exponential rise
    """
    a, b, D_plat, lam1, w1, lam2, w2, k = params
    lmbda  = np.asarray(lam, dtype=float)
    t1     = 0.5 * (1 + np.tanh((lmbda - lam1) / w1))
    t2     = 0.5 * (1 + np.tanh((lmbda - lam2) / w2))
    phase1 = a + b * lmbda
    phase2 = D_plat
    exponential_cap = np.clip(k * (lmbda - lam2), -7, 7)
    phase3 = D_plat * np.exp(exponential_cap)
    return (1 - t1) * phase1 + t1 * ((1 - t2) * phase2 + t2 * phase3)

def _build_hwang_spline(T: float) -> CubicSpline:
    """ Precompute a C2-smooth CubicSpline for D_hwang(λ, T) over λ ∈ [2, 22]."""
    lam_grid = np.linspace(2.0, 22.0, _diffusivity_grid)
    D = D_hwang_raw(lam_grid, D_params)
    D *= np.exp(2416 * (1 / 300 - 1 / T))  # Arrhenius scaling with literature obtained activation energy
    D *= membraneparams.sf_D               # Scaling factor correction.
    return CubicSpline(lam_grid, D, bc_type="natural", extrapolate=True)

def membrane_water_diffusivity(self, lam, model):
    """
    Calculate water diffusivity in the membrane as a function of water content.
    Uses smooth tanh-based transition between dry and wet diffusivity states.
    Args:
        lam: Water content lambda (array)
    Returns:
        D: Diffusivity in m2/s (scalar or array, same shape as lam)
    """
    lam = np.clip(lam, 2, 22)  
    if model == "hwang":
        # Build (or retrieve) the precomputed spline for this temperature.
        # Temperature is constant within a simulation, so the cache is hit
        # on every residual evaluation after the first.
        T = self.T
        key = ("hwang", float(T))
        if key not in _D_spline_cache:
            _D_spline_cache[key] = _build_hwang_spline(T)
        D = _D_spline_cache[key](lam)
    elif model == "bimodal":
        # Zhang, Z.; Promislow, K.; Martin, J.; Wang, H.; Balcom, B. J. Bi-Modal Water Transport Behavior across a Simple Nafion Membrane. 
        # Journal of Power Sources 2011, 196 (20), 8525–8530. https://doi.org/10.1016/j.jpowsour.2011.06.046.
        # Material parameters
        lambda_t = 16     # Transition water content threshold
        epsilon  = 0.5    # Transition zone (width)
        D_m      = 2.0509e-11 # Slope of diffusivity under dry conditions (m2/s)
        D_p      = 4.5e-9 # Diffusivity under wet conditions (m2/s)
        # Normalized transition parameter
        transition = (lam - lambda_t) / epsilon
        # Smooth weighting functions (interpolation from 0 to 1)
        weight_dry = (1 - np.tanh(transition)) / 2   # Decreases with λ
        weight_wet = (1 + np.tanh(transition)) / 2   # Increases with λ
        # Interpolate between dry and wet diffusivity
        D = D_m * lam * weight_dry + D_p * weight_wet
    elif model == "springer":
        # Springer, T. E.; Zawodzinski, T. A.; Gottesfeld, S. Polymer Electrolyte Fuel Cell Model. J. Electrochem. Soc. 1991, 138 (8), 2334–2342. https://doi.org/10.1149/1.2085971.
        # Ye, M.; Pino-Muñoz, C. A.; Cannon, C. G.; Sun, Y.; Kucernak, A.; Klusener, P. A. A.; Brandon, N. Understanding H2-X Redox Flow Battery Performance Through Modelling and Design of Experiments. SSRN 2025. https://doi.org/10.2139/ssrn.5126397.
        # Chinannai, M. F.; Ju, H. Effects of Water Transport on Performance Behaviors of Hydrogen Bromine Redox Flow Batteries. Journal of Energy Storage 2022, 49, 104151. https://doi.org/10.1016/j.est.2022.104151.
        T = self.T
        lam_arr = np.asarray(lam)
        D = np.zeros_like(lam_arr, dtype=float)
        # Vectorized piecewise evaluation to support array inputs.
        mask_2 = lam_arr <= 2                   # Below λ = 2, diffusivity is constant at D(2)
        mask_3 = (lam_arr > 2) & (lam_arr <= 3) # Between λ = 2 and λ = 3, linear interpolation between D(2) and D(3)
        mask_4 = (lam_arr > 3) & (lam_arr <= 4) # Between λ = 3 and λ = 4, linear interpolation between D(3) and D(4)
        mask_gt4 = lam_arr > 4                  # Above λ = 4, use the cubic polynomial fit
        # Below lambda = 2
        D[mask_2] = 2.69266e-10
        # Between lambda = 2 and 3
        D[mask_3] = 1e-10 * np.exp(2416 * (1 / 303 - 1 / T)) * (
            0.87 * (3 - lam_arr[mask_3]) + 2.95 * (lam_arr[mask_3] - 2)
        )
        # Between lambda = 3 and 4
        D[mask_4] = 1e-10 * np.exp(2416 * (1 / 303 - 1 / T)) * (
            2.95 * (4 - lam_arr[mask_4]) + 1.642 * (lam_arr[mask_4] - 3)
        )
        # Above lambda = 4
        D[mask_gt4] = 1e-10 * np.exp(2416 * (1 / 303 - 1 / T)) * (
            2.563
            - 0.33 * lam_arr[mask_gt4]
            + 0.0264 * lam_arr[mask_gt4] ** 2
            - 0.000671 * lam_arr[mask_gt4] ** 3
        )
        if np.isscalar(lam):
            D = float(D)
    elif model == "zawodzinski":
        # Motupally, S.; Becker, A. J.; Weidner, J. W. Diffusion of Water in Nafion 115 Membranes. J. Electrochem. Soc. 2000, 147 (9), 3171. https://doi.org/10.1149/1.1393879.
        # Zawodzinski, T. A. Jr.; Neeman, M.; Sillerud, L. O.; Gottesfeld, S. Determination of Water Diffusion Coefficients in Perfluorosulfonate Ionomeric Membranes. J. Phys. Chem. 1991, 95 (15), 6040–6044. https://doi.org/10.1021/j100168a060.
        # Obtained from the above references, smoothed with a tanh blend at λ = 3. Blending both branches, and returns D(17) for λ > 17
        lmbda  = np.asarray(lam, dtype=float)
        T      = self.T
        expT   = np.exp(-2436 / T)
        D1     = 3.10e-3 * lmbda * (-1 + np.exp(0.28 * lmbda))            * expT  # valid near λ ≤ 3
        D2     = 4.17e-4 * (1 + 161 * np.exp(-np.minimum(lmbda, 17.0)))   * expT  # capped at λ = 17
        w     = 0.5 * (1.0 + np.tanh((lmbda - 3.0) / 0.5))                        # Smooth blending
        D      = ((1.0 - w) * D1 + w * D2) * 1e-4                                 # Conversion from cm2/s -> m2/s
        if np.isscalar(lam):
            D = float(D)
    elif model == "motupally":
        # Motupally, S.; Mishra, L.; Thiagarajan, R. S.; Subramanian, V. R. Modeling Water Transport in Polymer Electrolyte Membrane Electrolyzers Using a One-Dimensional Transport Model. International Journal of Hydrogen Energy 2024, 62, 352–361. https://doi.org/10.1016/j.ijhydene.2024.02.318.
        lmbda = np.asarray(lam, dtype=float)
        D     = 1.4e-10 * np.log(lmbda) + 1e-9 # m2/s
        if np.isscalar(lam):
            D = float(D)
    elif model == "gerteisen":
        # Gerteisen, D.; Heilmann, T.; Ziegler, C. Modeling the Phenomena of Dehydration and Flooding of a Polymer Electrolyte Membrane Fuel Cell. Journal of Power Sources 2009, 187 (1), 165–181. https://doi.org/10.1016/j.jpowsour.2008.10.102.
        lmbda = np.asarray(lam, dtype=float)
        D     = 2.1e-7 * lmbda * np.exp(-2436 / self.T) # m2/s
        if np.isscalar(lam):
            D = float(D)
    elif model == "pino-munoz":
        # Pino-Muñoz, C. A.; Dewage, H. H.; Yufit, V.; Brandon, N. P. A Unit Cell Model of a Regenerative Hydrogen-Vanadium Fuel Cell. J. Electrochem. Soc. 2017, 164 (14), F1717. https://doi.org/10.1149/2.1431714jes.
        lmbda = np.asarray(lam, dtype=float)
        D     = 4.1e-10 * (lmbda/25)**0.15 * (1 + np.tanh((lmbda-2.5)/1.4))
        if np.isscalar(lam):
            D = float(D)
    elif model == "goshtabi":
        # Goshtasbi, A.; Pence, B. L.; Ersal, T. Computationally Efficient Pseudo-2D Non-Isothermal Modeling of Polymer Electrolyte Membrane Fuel Cells with Two-Phase Phenomena. J. Electrochem. Soc. 2016, 163 (13), F1412–F1432. https://doi.org/10.1149/2.0871613jes.
        lmbda = np.asarray(lam, dtype=float)
        D_no_swelling = 1e-4 * (0.0539) * (1 + 0.0027*lmbda**2) * (1 + np.tanh((lmbda-2.6225)/0.8729)) * np.exp(-3343/self.T) # m2/s
        D             = D_no_swelling * (1 + self.membrane.V_w * lmbda / self.membrane.V_m)**(-2) # m2/s
        if np.isscalar(lam):
            D = float(D)
    else:
        D = np.zeros_like(lam)  # Default to zero diffusivity if model is invalid
        raise ValueError("Invalid model type for water diffusion")
    return D # type: ignore