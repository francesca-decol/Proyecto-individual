import numpy as np
from sklearn.linear_model import LinearRegression

# Data
SOC = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9])
T = np.array([297.15, 305.15, 313.15]*3)
a_w = np.array([0.580, 0.594, 0.613, 0.572, 0.586, 0.605, 0.561, 0.574, 0.591])

# Features: intercept, SOC, T
X = np.column_stack([np.ones_like(SOC), SOC, T])
model = LinearRegression(fit_intercept=False)
model.fit(X, a_w)
c0, c1, c2 = model.coef_

print(f"a_w = {c0:.4f} + {c1:.4f}*SOC + {c2:.4f}*T")

# Results:
# a_w = -0.0114 + -0.0254*SOC + 0.0020*T
#
# Water activity was fitted as a linear function using nine experimental data points,
# covering the state of charge (SOC) range from 10% to 90%. 
# The data used for this fit were obtained from Ye 2025 experiments.