import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Step 1: Simulate ultra-stable vitality curve (Diamond)
t_data = np.linspace(0, 1e6, 1000)        # 1 million years
P_data = 1 - 1e-6 * t_data                # Very slow decay

# Step 2: Compute dP/dt from simulated data
dP_obs = np.gradient(P_data, t_data)

# Step 3: Define Reverse PEF Functional Form
def reconstructed_dP_dt(P, t, alpha, beta, a, b, c, d, omega):
    C = a * P
    S = b * np.sqrt(np.maximum(P, 0))
    E = c * P**2
    eta = d * np.sin(omega * t) + 1
    numerator = C * np.log(1 + S)
    denominator = 1 + beta * E
    return alpha * (numerator / denominator) * eta

# Step 4: Wrapper for fitting function
def fit_func(t, alpha, beta, a, b, c, d, omega):
    return reconstructed_dP_dt(P_data, t, alpha, beta, a, b, c, d, omega)

# Step 5: Fit parameters
initial_guesses = [0.01, 0.01, 1.0, 1.0, 0.01, 0.01, 0.0001]
bounds = ([0]*7, [10]*7)

popt, _ = curve_fit(
    fit_func,
    t_data,
    dP_obs,
    p0=initial_guesses,
    bounds=bounds,
    maxfev=10000
)

# Step 6: Predict dP/dt using fitted PEF parameters
dP_pred = fit_func(t_data, *popt)

# Step 7: Plot vitality curve
plt.figure(figsize=(10, 5))
plt.plot(t_data, P_data, label='Diamond Vitality Curve', color='black')
plt.xlabel('Time (years)')
plt.ylabel('Structural Integrity (P)')
plt.title('Simulated Longevity Curve of Diamond')
plt.grid(True)
plt.legend()
plt.show()

# Step 8: Plot observed vs predicted dP/dt
plt.figure(figsize=(10, 5))
plt.plot(t_data, dP_obs, label='Observed dP/dt', color='blue')
plt.plot(t_data, dP_pred, label='Predicted dP/dt (Reverse PEF)', linestyle='--', color='orange')
plt.xlabel('Time (years)')
plt.ylabel('Rate of Structural Decay (dP/dt)')
plt.title('Reverse PEF Fit — Diamond-Like Stability')
plt.grid(True)
plt.legend()
plt.show()

# Step 9: Print fitted parameters
print("🔬 Fitted PEF Parameters — Diamond Stability")
param_names = ['alpha', 'beta', 'a', 'b', 'c', 'd', 'omega']
for name, value in zip(param_names, popt):
    print(f"{name}: {value:.6f}")
