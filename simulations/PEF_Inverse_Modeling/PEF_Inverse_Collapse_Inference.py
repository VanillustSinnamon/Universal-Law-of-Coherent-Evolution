import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Step 1: Simulate Collapse — vitality rises, peaks, then decays
t_data = np.linspace(0, 20, 200)
P_data = np.exp(-0.2 * (t_data - 10)**2)  # bell-shaped vitality curve

# Step 2: Compute Observed Rate of Change (dP/dt)
dP_obs = np.gradient(P_data, t_data)

# Step 3: Reverse PEF Functional Form
def reconstructed_dP_dt(P, t, alpha, beta, a, b, c, d, omega):
    C = a * P
    S = b * np.sqrt(np.maximum(P, 0))  # avoid sqrt of negatives
    E = c * P**2
    eta = d * np.sin(omega * t) + 1
    numerator = C * np.log(1 + S)
    denominator = 1 + beta * E
    return alpha * (numerator / denominator) * eta

# Step 4: Wrapper for Curve Fitting
def fit_func(t, alpha, beta, a, b, c, d, omega):
    return reconstructed_dP_dt(P_data, t, alpha, beta, a, b, c, d, omega)

# Step 5: Fit Parameters with Bounds and Extra Iterations
initial_guesses = [0.5, 0.1, 1.0, 1.0, 0.01, 0.5, 0.2]
lower_bounds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
upper_bounds = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

popt, _ = curve_fit(
    fit_func,
    t_data,
    dP_obs,
    p0=initial_guesses,
    bounds=(lower_bounds, upper_bounds),
    maxfev=10000  # allow more iterations
)

# Step 6: Predict dP/dt from Fitted Parameters
dP_pred = fit_func(t_data, *popt)

# Step 7: Plot Collapse Curve P(t)
plt.figure(figsize=(10, 5))
plt.plot(t_data, P_data, label='Collapse P(t)', color='crimson')
plt.xlabel('Time')
plt.ylabel('Vitality P(t)')
plt.title('Simulated Collapse of a Coherent System')
plt.grid(True)
plt.legend()
plt.show()

# Step 8: Plot Observed vs Predicted dP/dt
plt.figure(figsize=(10, 5))
plt.plot(t_data, dP_obs, label='Observed dP/dt', color='blue')
plt.plot(t_data, dP_pred, label='Predicted dP/dt (Reverse PEF)', linestyle='--', color='orange')
plt.xlabel('Time')
plt.ylabel('Rate of Change (dP/dt)')
plt.title('Collapse Profile — Reverse Engineered from PEF')
plt.grid(True)
plt.legend()
plt.show()

# Step 9: Show Fitted Parameters
print("📡 Fitted Parameters (Collapse Test):")
param_names = ['alpha', 'beta', 'a', 'b', 'c', 'd', 'omega']
for name, value in zip(param_names, popt):
    print(f"{name}: {value:.4f}")
