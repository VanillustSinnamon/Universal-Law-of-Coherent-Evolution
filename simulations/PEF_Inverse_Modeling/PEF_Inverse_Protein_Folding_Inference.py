"""
PEF_Inverse_Protein_Folding_Simulation.py
------------------------------------------
This simulation models a protein folding curve using the Universal Law of Coherent Evolution (PEF).
It starts disordered (low vitality) and gradually folds into a stable structure (high vitality).
Noise is added to simulate biological uncertainty.

The reverse PEF engine extracts hidden coherence parameters that govern folding.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 🧬 STEP 1: Simulate a Folding Vitality Curve (sigmoid shape + noise)
t_data = np.linspace(0, 100, 300)  # Time points from 0 to 100
P_data = 1 / (1 + np.exp(-0.1 * (t_data - 50)))  # Sigmoid folding curve

# Add small random noise to simulate real biological fluctuations
np.random.seed(42)
P_data += np.random.normal(0, 0.01, size=P_data.shape)

# Compute dP/dt (rate of change of vitality)
dP_obs = np.gradient(P_data, t_data)

# 🧪 STEP 2: Define the Reverse PEF Model
def reconstructed_dP_dt(P, t, alpha, beta, a, b, c, d, omega):
    C = a * P
    S = b * np.sqrt(np.maximum(P, 0))
    E = c * P**2
    eta = d * np.sin(omega * t) + 1
    numerator = C * np.log(1 + S)
    denominator = 1 + beta * E
    return alpha * (numerator / denominator) * eta

# Wrapper for curve_fit
def fit_func(t, alpha, beta, a, b, c, d, omega):
    return reconstructed_dP_dt(P_data, t, alpha, beta, a, b, c, d, omega)

# 🧠 STEP 3: Fit PEF Parameters
initial_guesses = [1.0, 0.1, 1.0, 1.0, 0.01, 0.5, 0.1]
bounds = ([0]*7, [10]*7)

popt, _ = curve_fit(
    fit_func,
    t_data,
    dP_obs,
    p0=initial_guesses,
    bounds=bounds,
    maxfev=10000
)

# Predict dP/dt using fitted parameters
dP_pred = fit_func(t_data, *popt)

# 🧬 STEP 4: Plot Protein Folding Curve
plt.figure(figsize=(10, 5))
plt.plot(t_data, P_data, label='Simulated Folding Vitality', color='green')
plt.xlabel('Time')
plt.ylabel('Folding Vitality (P)')
plt.title('Protein Folding: Vitality Evolution')
plt.grid(True)
plt.legend()
plt.show()

# Plot dP/dt Fit
plt.figure(figsize=(10, 5))
plt.plot(t_data, dP_obs, label='Observed dP/dt', color='blue')
plt.plot(t_data, dP_pred, label='Predicted dP/dt (Reverse PEF)', linestyle='--', color='orange')
plt.xlabel('Time')
plt.ylabel('Rate of Change (dP/dt)')
plt.title('Reverse PEF Fit — Protein Folding Coherence')
plt.grid(True)
plt.legend()
plt.show()

# 🧪 STEP 5: Display Fitted Parameters
print("🔬 Fitted PEF Parameters — Protein Folding")
param_names = ['alpha', 'beta', 'a', 'b', 'c', 'd', 'omega']
for name, value in zip(param_names, popt):
    print(f"{name}: {value:.6f}")
