import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# --- Added for reproducibility as discussed ---
np.random.seed(42) # Set a seed for reproducible random numbers

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

# --- Define C, S, eta as functions of time (these remain constant for this test) ---
C_t_func = lambda time: 1 + 0.5 * np.sin(0.1 * time)
S_t_func = lambda time: 2 * (1 - np.exp(-0.01 * time))
eta_t_func = lambda time: np.full_like(time, eta_val)

# --- Stress Test Configuration for E(t) Noise Amplitude ---
# Define different noise amplitude factors to test
# 0.1: Lower noise
# 0.2: Current/Baseline noise (from your original code)
# 0.5: Higher noise
# 1.0: Very high noise (significant dissonance)
noise_amplitude_factors = [0.05, 0.2, 0.5, 1.0]
results_P = {} # Dictionary to store P(t) results for each noise level
results_E = {} # Dictionary to store E(t) results for each noise level

print("Starting E(t) Noise Amplitude Stress Test...")

for factor in noise_amplitude_factors:
    # Generate random E values scaled by the current factor
    # We use a new seed for each run to ensure different random noise patterns,
    # but the overall amplitude is controlled by the factor.
    # If you want the *exact same noise pattern* scaled, you'd set the seed once outside the loop.
    # For stress testing, different patterns at different amplitudes are usually better.
    np.random.seed(int(factor * 1000) + 42) # Vary seed to get different noise patterns for each factor
    random_E_values = factor * np.random.rand(len(t)) # Scale the random values
    E_t_func = interp1d(t, random_E_values, kind='nearest', fill_value='extrapolate')

    # Solve the ODE for the current E(t) configuration
    sol = odeint(P_dot, P0, t, args=(alpha, beta, C_t_func, S_t_func, E_t_func, eta_t_func))
    results_P[f'Noise Factor {factor}'] = sol[:, 0]
    results_E[f'Noise Factor {factor}'] = E_t_func(t) # Store the actual E(t) used

    print(f"  - Simulation complete for Noise Factor: {factor}")

print("\nPlotting results...")

# --- Plotting P(t) for different noise levels ---
plt.figure(figsize=(12, 7))
for label, P_values in results_P.items():
    plt.plot(t, P_values, label=label)

plt.xlabel('Time')
plt.ylabel('P(t) - Placeholder Expansion')
plt.title('PEF Simulation: P(t) with Varying E(t) Noise Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plotting E(t) for different noise levels (to show the change) ---
plt.figure(figsize=(12, 7))
for label, E_values in results_E.items():
    plt.plot(t, E_values, label=label, alpha=0.7) # Use alpha for transparency

plt.xlabel('Time')
plt.ylabel('E(t) - Entropic Dissonance (Noise)')
plt.title('E(t) Signal for Different Noise Amplitudes')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nStress test complete. Observe the graphs to see the impact of E(t) noise on P(t).")


