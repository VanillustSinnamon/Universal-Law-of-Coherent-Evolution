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
paradigm_shift_time = 100.0 # The time point at which the paradigm shift occurs

# --- Define initial placeholder capacity ---
P0 = 0.1

# --- Define initial model parameters (Balanced World baseline) ---
alpha_initial = 1.0
beta_initial = 0.5
eta_val_initial = 1.0

# --- Define parameters AFTER the paradigm shift ---
# We'll change alpha and beta to simulate a shift in how the system processes
# its internal dynamics and dissonance.
alpha_shifted = 0.8 # Slightly less aggressive expansion
beta_shifted = 0.2 # Dissonance (E) has less dampening effect (system becomes more tolerant/efficient at resolving it)

# --- Define C, S, E, eta functions (using Balanced World parameters) ---
C_t_func = lambda time: (1 + 0.5 * np.sin(0.05 * time)) + 0.5 * np.sin(0.1 * time) # Moderate oscillation
random_E_values = 0.5 * np.random.rand(len(t)) # Moderate noise
E_t_func = interp1d(t, random_E_values, kind='nearest', fill_value='extrapolate')
S_t_func = lambda time: 2 * (1 - np.exp(-0.01 * time)) # Medium growth
eta_t_func = lambda time: np.full_like(time, eta_val_initial) # Normal pacing

# --- Custom ODE function to handle the paradigm shift ---
# This function will switch the alpha and beta values at the paradigm_shift_time
def P_dot_with_shift(P, t_current, C_func, S_func, E_func, eta_func, shift_time, alpha_init, beta_init, alpha_shifted, beta_shifted):
    # Determine which alpha and beta to use based on current time
    if t_current < shift_time:
        current_alpha = alpha_init
        current_beta = beta_init
    else:
        current_alpha = alpha_shifted
        current_beta = beta_shifted

    # Call the original P_dot with the selected parameters
    return P_dot(P, t_current, current_alpha, current_beta, C_func, S_func, E_func, eta_func)

print("Starting Paradigm Shift (DLR-style Reconfiguration) Test...")

# Solve the ODE using the custom function with shift
sol_paradigm_shift = odeint(P_dot_with_shift, P0, t,
                            args=(C_t_func, S_t_func, E_t_func, eta_t_func,
                                  paradigm_shift_time, alpha_initial, beta_initial,
                                  alpha_shifted, beta_shifted))
P_paradigm_shift = sol_paradigm_shift[:, 0]

print(f"  - Simulation complete for Paradigm Shift at t={paradigm_shift_time}.")

print("\nPlotting results...")

# --- Plotting P(t) for the Paradigm Shift ---
plt.figure(figsize=(12, 7))
plt.plot(t, P_paradigm_shift, label='P(t) - Paradigm Shift Simulation', color='purple', linewidth=2)
# Add a vertical line to indicate the shift time
plt.axvline(x=paradigm_shift_time, color='gray', linestyle='--', label=f'Paradigm Shift at t={paradigm_shift_time}')

plt.xlabel('Time')
plt.ylabel('P(t) - Placeholder Expansion')
plt.title('PEF Simulation: P(t) with a Paradigm Shift (DLR-style Reconfiguration)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nParadigm Shift Stress Test complete. Observe how P(t) adapts after the fundamental change.")


