def water_saturation_pressure(self):
    """ Calculate the saturation pressure of water at the given temperature using the Antoine equation.
    (1) Springer, T. E.; Zawodzinski, T. A.; Gottesfeld, S. Polymer Electrolyte Fuel Cell Model. J. Electrochem. Soc. 1991, 138 (8), 2334–2342. https://doi.org/10.1149/1.2085971.
    (2) Ye, M.; Pino-Muñoz, C. A.; Cannon, C. G.; Sun, Y.; Kucernak, A.; Klusener, P. A. A.; Brandon, N. Understanding H2-X Redox Flow Battery Performance Through Modelling and Design of Experiments. SSRN 2025. https://doi.org/10.2139/ssrn.5126397.
"""
    T = self.T  # K
    logP_sat = 2.8263 + 0.02953*(T-273.15) - 9.1837e-5*(T-273.15)**2 + 1.4454e-7*(T-273.15)**3  
    P_sat = 10**logP_sat  # Convert from log10 to Pa
    return P_sat