import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# --- Added for reproducibility ---
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

# --- Fractal Conflict World Configuration ---
# Setting parameters to create a highly challenging, multi-paradox environment
fractal_conflict_params = {
    'E_noise_amp': 1.5, # Even higher noise than previous "Chaotic" (was 1.0)
    # C(t) will oscillate with a baseline that itself oscillates, and can dip very low
    # This creates "inverse coherence dips" or periods of severe coherence challenge
    'C_baseline_osc_amp': 1.5, # Stronger baseline oscillation
    'C_osc_amplitude': 1.0, # Stronger regular oscillation
    'S_growth_rate': 0.1, # Explosive growth (was 0.05 for Ideal)
    'eta_factor': 3.0 # Hyperfast pacing (was 2.0 for Ideal)
}

print("Starting Sixth Stress Test: Fractal Conflict Simulation...")

# Generate E(t) for extreme noise
np.random.seed(999) # A new, distinct seed for this extreme test
random_E_values = fractal_conflict_params['E_noise_amp'] * np.random.rand(len(t))
E_t_func = interp1d(t, random_E_values, kind='nearest', fill_value='extrapolate')

# Define C(t) with inverse coherence dips (oscillating baseline and large amplitude)
# The baseline will oscillate significantly, and the main sine wave will be strong
C_t_func = lambda time: (1 + fractal_conflict_params['C_baseline_osc_amp'] * np.sin(0.04 * time)) + \
                        fractal_conflict_params['C_osc_amplitude'] * np.sin(0.15 * time)
# Ensure C(t) doesn't go negative, as it represents coherence
C_t_func_clamped = lambda time: np.maximum(0.01, C_t_func(time)) # Clamp minimum at a small positive value

# Define S(t) for explosive growth
S_t_func = lambda time: 2 * (1 - np.exp(-fractal_conflict_params['S_growth_rate'] * time))

# Define eta(t) for hyperfast pacing
eta_t_func = lambda time: np.full_like(time, fractal_conflict_params['eta_factor'])

# Solve the ODE for the Fractal Conflict configuration
sol_fractal_conflict = odeint(P_dot, P0, t, args=(alpha, beta, C_t_func_clamped, S_t_func, E_t_func, eta_t_func))
P_fractal_conflict = sol_fractal_conflict[:, 0]

print("  - Simulation complete for Fractal Conflict World.")

print("\nPlotting results...")

# --- Plotting P(t) for the Fractal Conflict World ---
plt.figure(figsize=(12, 7))
plt.plot(t, P_fractal_conflict, label='P(t) - Fractal Conflict World', color='red', linewidth=2)

plt.xlabel('Time')
plt.ylabel('P(t) - Placeholder Expansion')
plt.title('PEF Simulation: P(t) in a Fractal Conflict World (Extreme Stress Test)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Optional: Plot the components for this extreme world to visualize the inputs ---
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(t, C_t_func_clamped(t), color='blue')
plt.ylabel('C(t)')
plt.title('Components for Fractal Conflict World')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, S_t_func(t), color='green')
plt.ylabel('S(t)')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, E_t_func(t), color='red')
plt.ylabel('E(t)')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, eta_t_func(t), color='purple')
plt.ylabel('η(t)')
plt.xlabel('Time')
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nSixth Stress Test complete. Observe how P(t) behaves under maximum paradox load.")


