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

# --- Define C, S, E functions (these will remain consistent for this test) ---
C_t_func = lambda time: 1 + 0.5 * np.sin(0.1 * time)
random_E_values = 0.2 * np.random.rand(len(t)) # Using original noise level for E(t)
# CORRECTED LINE BELOW: Changed 'random_E_E_values' to 'random_E_values'
E_t_func = interp1d(t, random_E_values, kind='nearest', fill_value='extrapolate')
S_t_func = lambda time: 2 * (1 - np.exp(-0.01 * time)) # Using original growth rate for S(t)

# --- Stress Test Configuration for η(t) Temporal Pacing ---
# Define different constant values for eta_val to test different pacing rates
# 0.5: Slower pacing
# 1.0: Current/Baseline pacing
# 2.0: Faster pacing
eta_factors = [0.5, 1.0, 2.0]
results_P = {} # Dictionary to store P(t) results for each eta configuration
results_eta = {} # Dictionary to store eta(t) values for each configuration

print("Starting η(t) Temporal Pacing Stress Test...")

for factor in eta_factors:
    # Define eta_t_func with the current constant factor
    eta_t_func_current = lambda time: np.full_like(time, factor)

    # Solve the ODE for the current eta(t) configuration
    sol = odeint(P_dot, P0, t, args=(alpha, beta, C_t_func, S_t_func, E_t_func, eta_t_func_current))
    results_P[f'η Factor {factor}'] = sol[:, 0]
    results_eta[f'η Factor {factor}'] = eta_t_func_current(t)

    print(f"  - Simulation complete for η Factor: {factor}")

print("\nPlotting results...")

# --- Plotting P(t) for different η(t) configurations ---
plt.figure(figsize=(12, 7))
for label, P_values in results_P.items():
    plt.plot(t, P_values, label=label)

plt.xlabel('Time')
plt.ylabel('P(t) - Placeholder Expansion')
plt.title('PEF Simulation: P(t) with Varying η(t) Temporal Pacing')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plotting η(t) for different configurations (to show the change) ---
plt.figure(figsize=(12, 7))
for label, eta_values in results_eta.items():
    plt.plot(t, eta_values, label=label, alpha=0.7)

plt.xlabel('Time')
plt.ylabel('η(t) - Temporal Pacing')
plt.title('η(t) Signal for Different Pacing Factors')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nStress test complete. Observe the graphs to see the impact of η(t) temporal pacing on P(t).")


