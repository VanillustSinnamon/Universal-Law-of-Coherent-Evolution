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

    log_S_term = np.log1p(S)
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

# --- Define S and E functions (these will remain consistent for this test) ---
S_t_func = lambda time: 2 * (1 - np.exp(-0.01 * time))
# Use the original E(t) noise level for consistency unless specified otherwise
random_E_values = 0.2 * np.random.rand(len(t))
E_t_func = interp1d(t, random_E_values, kind='nearest', fill_value='extrapolate')
eta_t_func = lambda time: np.full_like(time, eta_val)

# --- Stress Test Configuration for C(t) Oscillating Baseline ---
# We'll test different amplitudes for the *baseline oscillation* of C(t)
# Original C(t) was: 1 + 0.5 * np.sin(0.1 * time)
# Now, the '1' (baseline) will also oscillate.
# baseline_amplitudes:
# 0.0: No additional baseline oscillation (original behavior)
# 0.5: Moderate baseline oscillation
# 1.0: Strong baseline oscillation
c_baseline_oscillation_amplitudes = [0.0, 0.5, 1.0] # Amplitude of the *additional* sine wave for the baseline
results_P = {} # Dictionary to store P(t) results for each C(t) configuration
results_C = {} # Dictionary to store C(t) values for each configuration

print("Starting C(t) Oscillating Baseline Stress Test...")

for amp in c_baseline_oscillation_amplitudes:
    # Define C_t_func with an oscillating baseline
    # The original C_t was `1 + 0.5 * np.sin(0.1 * time)`
    # Now, the '1' (its baseline) will become `1 + amp * np.sin(0.05 * time)`
    # We use a slightly slower frequency (0.05) for the baseline oscillation to make it distinct
    C_t_func_current = lambda time: (1 + amp * np.sin(0.05 * time)) + 0.5 * np.sin(0.1 * time)

    # Solve the ODE for the current C(t) configuration
    sol = odeint(P_dot, P0, t, args=(alpha, beta, C_t_func_current, S_t_func, E_t_func, eta_t_func))
    results_P[f'C Baseline Oscillation Amp {amp}'] = sol[:, 0]
    results_C[f'C Baseline Oscillation Amp {amp}'] = C_t_func_current(t)

    print(f"  - Simulation complete for C Baseline Oscillation Amplitude: {amp}")

print("\nPlotting results...")

# --- Plotting P(t) for different C(t) configurations ---
plt.figure(figsize=(12, 7))
for label, P_values in results_P.items():
    plt.plot(t, P_values, label=label)

plt.xlabel('Time')
plt.ylabel('P(t) - Placeholder Expansion')
plt.title('PEF Simulation: P(t) with Varying C(t) Baseline Oscillation')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plotting C(t) for different configurations (to show the change) ---
plt.figure(figsize=(12, 7))
for label, C_values in results_C.items():
    plt.plot(t, C_values, label=label, alpha=0.7)

plt.xlabel('Time')
plt.ylabel('C(t) - System Coherence')
plt.title('C(t) Signal for Different Baseline Oscillation Amplitudes')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nStress test complete. Observe the graphs to see the impact of C(t) baseline oscillation on P(t).")


