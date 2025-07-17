import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Step 1: Simulate P(t) with Equilibrium Behavior
t_data = np.linspace(0, 20, 200)
P_data = 1.5 / (1 + np.exp(-0.5 * (t_data - 10)))  # Slower logistic growth

# Step 2: Compute Observed dP/dt
dP_obs = np.gradient(P_data, t_data)

# Step 3: Define Guessed Functional Forms (Reverse PEF)
def reconstructed_dP_dt(P, t, alpha, beta, a, b, c, d, omega):
    C = a * P
    S = b * np.sqrt(P)
    E = c * P**2
    eta = d * np.sin(omega * t) + 1
    numerator = C * np.log(1 + S)
    denominator = 1 + beta * E
    return alpha * (numerator / denominator) * eta

# Step 4: Fitting Function Wrapper
def fit_func(t, alpha, beta, a, b, c, d, omega):
    return reconstructed_dP_dt(P_data, t, alpha, beta, a, b, c, d, omega)

# Step 5: Initial Parameter Guesses and Curve Fitting
initial_guesses = [1.0, 0.1, 1, 1, 0.01, 0.5, 0.1]
popt, _ = curve_fit(fit_func, t_data, dP_obs, p0=initial_guesses)

# Step 6: Compute Predicted dP/dt Using Fitted Parameters
dP_pred = fit_func(t_data, *popt)

# Step 7: Plot Results — Vitality Curve
plt.figure(figsize=(10, 6))
plt.plot(t_data, P_data, label='Equilibrium P(t)', color='green')
plt.xlabel('Time')
plt.ylabel('Vitality P(t)')
plt.title('Simulated Vitality Approaching Equilibrium')
plt.grid(True)
plt.legend()
plt.show()

# Step 8: Plot Observed vs Predicted dP/dt
plt.figure(figsize=(10, 6))
plt.plot(t_data, dP_obs, label='Observed dP/dt', color='purple')
plt.plot(t_data, dP_pred, label='Predicted dP/dt (Reverse PEF)', linestyle='--', color='orange')
plt.xlabel('Time')
plt.ylabel('Rate of Change (dP/dt)')
plt.title('Observed vs. Predicted dP/dt — Reverse Engineering the PEF')
plt.grid(True)
plt.legend()
plt.show()

# Step 9: Print Fitted Parameters
print("Fitted Parameters:")
param_names = ['alpha', 'beta', 'a', 'b', 'c', 'd', 'omega']
for name, value in zip(param_names, popt):
    print(f"{name}: {value:.4f}")
