import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Step 1: Simulate a COVID-like vitality curve (rise, peak, decline)
t_data = np.linspace(0, 100, 300)  # 100 weeks
P_data = (
    1.5 * np.exp(-0.001 * (t_data - 30)**2) +  # First wave
    2.0 * np.exp(-0.001 * (t_data - 60)**2) +  # Second wave
    1.2 * np.exp(-0.001 * (t_data - 85)**2)    # Third wave
)

# Optional: Add noise to simulate real-world messiness
P_data += np.random.normal(0, 0.05, size=P_data.shape)

# Step 2: Compute observed dP/dt
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

# Step 4: Wrapper for curve fitting
def fit_func(t, alpha, beta, a, b, c, d, omega):
    return reconstructed_dP_dt(P_data, t, alpha, beta, a, b, c, d, omega)

# Step 5: Fit parameters
initial_guesses = [0.5, 0.1, 1.0, 1.0, 0.01, 0.5, 0.2]
bounds = ([0]*7, [10]*7)

popt, _ = curve_fit(
    fit_func,
    t_data,
    dP_obs,
    p0=initial_guesses,
    bounds=bounds,
    maxfev=10000
)

# Step 6: Predict dP/dt from fitted parameters
dP_pred = fit_func(t_data, *popt)

# Step 7: Plot vitality curve
plt.figure(figsize=(10, 5))
plt.plot(t_data, P_data, label='Simulated COVID Vitality Curve', color='darkred')
plt.xlabel('Time (weeks)')
plt.ylabel('P(t)')
plt.title('COVID-Like Vitality Curve')
plt.grid(True)
plt.legend()
plt.show()

# Step 8: Plot observed vs predicted dP/dt
plt.figure(figsize=(10, 5))
plt.plot(t_data, dP_obs, label='Observed dP/dt', color='blue')
plt.plot(t_data, dP_pred, label='Predicted dP/dt (Reverse PEF)', linestyle='--', color='orange')
plt.xlabel('Time (weeks)')
plt.ylabel('Rate of Change (dP/dt)')
plt.title('Reverse PEF Fit on COVID-Like Curve')
plt.grid(True)
plt.legend()
plt.show()

# Step 9: Display fitted parameters
print("📡 Fitted Parameters (COVID Curve):")
param_names = ['alpha', 'beta', 'a', 'b', 'c', 'd', 'omega']
for name, value in zip(param_names, popt):
    print(f"{name}: {value:.4f}")
