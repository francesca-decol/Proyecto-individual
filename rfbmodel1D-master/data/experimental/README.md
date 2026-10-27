## LS_VS_Zhang.csv
### Description
Membrane water content as a function of normalized thickness for Nafion membranes.
Liquid saturated and vapor saturated. 
### Columns
- `normalized_thickness`: Dimensionless thickness parameter (0-1 range)
- `water_content`: Water content in percent
### Reference
[1] Z. Zhang, K. Promislow, J. Martin, H. Wang, and B. J. Balcom, "Bi-modal water transport behavior across a simple Nafion membrane," *Journal of Power Sources*, vol. 196, no. 20, pp. 8525–8530, Oct. 2011, doi: 10.1016/j.jpowsour.2011.06.046.
### Notes
- Data extracted from reference [1]
 
## DRY_VS_sorption_Hwang.csv.csv
### Description
Sorption curves (water content vs. normalized thickness) at different times during sorption experiments. Membrane water content as a function of normalized thickness for Nafion membranes, exposed to dry N2 at left side and saturated vapor at right side.
Tracks the evolution of water content distribution in the membrane at various time points.
Membrane thickness are 120, 122, 126, and 122 μm for 16.25, 36.25, 66.25, and 456.25 min  sorption times
### Columns
- `time_seconds`: Time elapsed in sorption experiment (seconds)
- `normalized_thickness`: Dimensionless thickness parameter (0-1 range)
- `water_content`: Water content
### Reference
[2] G. S. Hwang, D. Y. Parkinson, A. Kusoglu, A. A. MacDowell, and A. Z. Weber, “Understanding Water Uptake and Transport in Nafion Using X-ray Microtomography,” ACS Macro Lett., vol. 2, no. 4, pp. 288–291, Apr. 2013, doi: 10.1021/mz300651a.
### Notes
- Data extracted from reference [2]

 
## DRY_VS_desorption_Hwang.csv.csv
### Description
Desorption curves (water content vs. normalized thickness) at different times during sorption experiments. Membrane water content as a function of normalized thickness for Nafion membranes, exposed to dry N2 at left side and saturated vapor at right side.
Tracks the evolution of water content distribution in the membrane at various time points.
Membrane thickness are 120, 122, 126, and 122 μm for 16.25, 36.25, 66.25, and 456.25 min  sorption times
### Columns
- `time_seconds`: Time elapsed in sorption experiment (seconds)
- `normalized_thickness`: Dimensionless thickness parameter (0-1 range)
- `water_content`: Water content
### Reference
[2] G. S. Hwang, D. Y. Parkinson, A. Kusoglu, A. A. MacDowell, and A. Z. Weber, “Understanding Water Uptake and Transport in Nafion Using X-ray Microtomography,” ACS Macro Lett., vol. 2, no. 4, pp. 288–291, Apr. 2013, doi: 10.1021/mz300651a.
### Notes
- Data extracted from reference [2]

## DRY_LS_desorption_Hwang.csv.csv
### Description
Desorption curves (water content vs. normalized thickness) at different times during desorption experiments. Membrane water content as a function of normalized thickness for Nafion membranes, exposed to dry N2 at left side and liquid vapor at right side.
Tracks the evolution of water content distribution in the membrane at various time points.
### Columns
- `time_seconds`: Time elapsed in sorption experiment (seconds)
- `normalized_thickness`: Dimensionless thickness parameter (0-1 range)
- `water_content`: Water content
### Reference
[2] G. S. Hwang, D. Y. Parkinson, A. Kusoglu, A. A. MacDowell, and A. Z. Weber, “Understanding Water Uptake and Transport in Nafion Using X-ray Microtomography,” ACS Macro Lett., vol. 2, no. 4, pp. 288–291, Apr. 2013, doi: 10.1021/mz300651a.
### Notes
- Data extracted from reference [2]

## polarization.csv
### Description
Polarization curve from Minnan model.
### Columns
- `Current Density (A/m2)`: current density
- `Voltage (V)`: Applied voltage
- `SOC`: State of charge
- `mode`: charge or discharge
### Reference
(3) Ye, M.; Pino-Muñoz, C. A.; Cannon, C. G.; Sun, Y.; Kucernak, A.; Klusener, P. A. A.; Brandon, N. Understanding H2-X Redox Flow Battery Performance Through Modelling and Design of Experiments. SSRN 2025. https://doi.org/10.2139/ssrn.5126397.
### Notes
- Data extracted from reference [3]

## E_cell.csv

### Description
Cell potential (charge and discharge polarization curves) for a 25 cm² H₂-V redox flow battery under a 3³ full factorial design of experiments varying state of charge (SOC), catholyte flow rate (Q_v) and temperature (T). Data digitized from the experimental results of Ye et al. (2025).

### Columns
- `test_no`: Test number according to the full factorial design (1–29, see Table A.1 in reference)
- `SOC`: State of charge (–), values: 0.10, 0.50, 0.90
- `T (°C)`: Cell temperature in degrees Celsius, values: 24, 32, 40
- `Q_v (mL/min)`: Catholyte volumetric flow rate in mL/min, values: 20, 45, 70
- `mode`: Operating mode, either `charge` or `discharge`
- `j_appl (A/m2)`: Applied current density in A/m²
- `value (V)`: Cell potential in V

### Reference
Ye, M.; Pino-Muñoz, C. A.; Cannon, C. G.; Sun, Y.; Kucernak, A. R.; Klusener, P. A. A.; Brandon, N. P. Understanding H₂-X Redox Flow Battery Performance Through Modeling and Design of Experiments. *Journal of Energy Storage* **2025**, *132*, 117765. https://doi.org/10.1016/j.est.2025.117765.

### Notes
- Data extracted from Figs. S8, S9, S10 (full factorial) and Figs. 9, 12 (main paper) using WebPlotDigitizer
- Hydrogen inlet relative humidity was 100% for all tests
- Hydrogen flow rate was fixed at 250 mL/min for all tests
- Upper cut-off potential: 1.40 V; lower cut-off potential: 0.40 V
- The 9 tests used for model validation are: 5, 16, 25 (varying SOC), 18, 19, 20 (varying T), 22, 25, 28 (varying Q_v)
- Tests 14, 15, 16 are replicates of the central point (SOC=50%, T=32°C, Q_v=45 mL/min)

## HFR.csv

### Description
High-frequency resistance (HFR) measured during charge and discharge polarization for a 25 cm² H₂-V redox flow battery under a 3³ full factorial design of experiments varying state of charge (SOC), catholyte flow rate (Q_v) and temperature (T). The HFR reflects primarily the membrane ionic resistance and is closely related to the ionomer water content. Data digitized from the experimental results of Ye et al. (2025).

### Columns
- `test_no`: Test number according to the full factorial design (1–29, see Table A.1 in reference)
- `SOC`: State of charge (–), values: 0.10, 0.50, 0.90
- `T (°C)`: Cell temperature in degrees Celsius, values: 24, 32, 40
- `Q_v (mL/min)`: Catholyte volumetric flow rate in mL/min, values: 20, 45, 70
- `mode`: Operating mode, either `charge` or `discharge`
- `j_appl (A/m2)`: Applied current density in A/m²
- `value (Ω·cm²)`: High-frequency resistance in Ω·cm²

### Reference
Ye, M.; Pino-Muñoz, C. A.; Cannon, C. G.; Sun, Y.; Kucernak, A. R.; Klusener, P. A. A.; Brandon, N. P. Understanding H₂-X Redox Flow Battery Performance Through Modeling and Design of Experiments. *Journal of Energy Storage* **2025**, *132*, 117765. https://doi.org/10.1016/j.est.2025.117765.


## E_HFR_corr.csv

### Description
HFR-corrected cell potential during charge and discharge polarization for a 25 cm² H₂-V redox flow battery under a 3³ full factorial design of experiments varying state of charge (SOC), catholyte flow rate (Q_v) and temperature (T). The HFR-corrected potential is obtained by removing the ohmic contribution from the cell potential (E_HFR_corr = E_cell + HFR × j_appl), isolating the electrode kinetic overpotentials (cathode activation and concentration, anode hydrogen kinetics, and ACL protonic resistance). Data digitized from the experimental results of Ye et al. (2025).

### Columns
- `test_no`: Test number according to the full factorial design (1–29, see Table A.1 in reference)
- `SOC`: State of charge (–), values: 0.10, 0.50, 0.90
- `T (°C)`: Cell temperature in degrees Celsius, values: 24, 32, 40
- `Q_v (mL/min)`: Catholyte volumetric flow rate in mL/min, values: 20, 45, 70
- `mode`: Operating mode, either `charge` or `discharge`
- `j_appl (A/m2)`: Applied current density in A/m²
- `value (V)`: HFR-corrected cell potential in V

### Reference
Ye, M.; Pino-Muñoz, C. A.; Cannon, C. G.; Sun, Y.; Kucernak, A. R.; Klusener, P. A. A.; Brandon, N. P. Understanding H₂-X Redox Flow Battery Performance Through Modeling and Design of Experiments. *Journal of Energy Storage* **2025**, *132*, 117765. https://doi.org/10.1016/j.est.2025.117765.




