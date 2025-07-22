import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# --- Added for reproducibility ---
np.random.seed(42) # Set a seed for reproducible random numbers for E(t)

# Define the derivative function P_dot(P, t, alpha, beta, C_t, S_t, E_t, eta_t)
def P_dot(P, t, alpha, beta, C_func, S_func, E_func, eta_func):
    C = C_func(t)
    S = S_func(t)
    E = E_func(t)
    eta = eta_func(t)

    log_S_term = np.log1p(S) # np.log1p(x) is log(1+x), handles S=0 gracefully
    denominator = 1 + beta * E

    dPdt = alpha * (C * log_S_term / denominator) * eta
    return dPdt

# --- Define time array ---
t = np.linspace(0, 200, 2001) # Time from 0 to 200, with 2001 points

# --- Define initial placeholder capacity ---
P0 = 0.1

# --- Define fixed model parameters ---
alpha = 1.0
beta = 0.5
eta_val = 1.0 # The constant value for eta

# --- Define C and E functions (these will remain consistent for this test) ---
C_t_func = lambda time: 1 + 0.5 * np.sin(0.1 * time)
random_E_values = 0.2 * np.random.rand(len(t))
E_t_func = interp1d(t, random_E_values, kind='nearest', fill_value='extrapolate')
eta_t_func = lambda time: np.full_like(time, eta_val)

# --- Stress Test Configuration for S(t) Growth Rate ---
# We'll vary the 'k' in 2 * (1 - np.exp(-k * time)) to change growth speed
# Current/Baseline k is 0.01
complexity_growth_rates = [0.005, 0.01, 0.02, 0.05] # Slower, Original, Faster, Much Faster
results_P = {} # Dictionary to store P(t) results for each S(t) configuration
results_S = {} # Dictionary to store S(t) values for each configuration

print("Starting S(t) Structural Complexity Growth Rate Stress Test...")

for rate in complexity_growth_rates:
    # Define S_t_func with the current growth rate
    S_t_func_current = lambda time: 2 * (1 - np.exp(-rate * time))

    # Solve the ODE for the current S(t) configuration
    sol = odeint(P_dot, P0, t, args=(alpha, beta, C_t_func, S_t_func_current, E_t_func, eta_t_func))
    results_P[f'S Growth Rate {rate}'] = sol[:, 0]
    results_S[f'S Growth Rate {rate}'] = S_t_func_current(t)

    print(f"  - Simulation complete for S Growth Rate: {rate}")

print("\nPlotting results...")

# --- Plotting P(t) for different S(t) configurations ---
plt.figure(figsize=(12, 7))
for label, P_values in results_P.items():
    plt.plot(t, P_values, label=label)

plt.xlabel('Time')
plt.ylabel('P(t) - Placeholder Expansion')
plt.title('PEF Simulation: P(t) with Varying S(t) Growth Rate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plotting S(t) for different configurations (to show the change) ---
plt.figure(figsize=(12, 7))
for label, S_values in results_S.items():
    plt.plot(t, S_values, label=label, alpha=0.7)

plt.xlabel('Time')
plt.ylabel('S(t) - Structural Complexity')
plt.title('S(t) Signal for Different Growth Rates')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nStress test complete. Observe the graphs to see the impact of S(t) growth rate on P(t).")


