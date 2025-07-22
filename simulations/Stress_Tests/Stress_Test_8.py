import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# --- Added for reproducibility ---
np.random.seed(42) # Set a seed for reproducible random numbers

# Define the derivative function P_dot(P, t, alpha, beta, C_t, S_t, E_t, eta_t)
# This is your core PEF equation.
def P_dot(P, t, alpha, beta, C_func, S_func, E_func, eta_func):
    """
    Calculates the derivative of Placeholder Expansion P with respect to time t.
    This is the core of the Placeholder Expansion Function (PEF) from the ERS framework.

    Args:
        P (float): Current value of Placeholder Expansion.
        t (float): Current time.
        alpha (float): Expansion coefficient (ontological drive).
        beta (float): Dissonance integration coefficient (how system processes paradox).
        C_func (callable): Function for System Coherence C(t).
        S_func (callable): Function for Structural Complexity S(t).
        E_func (callable): Function for Entropic Dissonance E(t).
        eta_func (callable): Function for Temporal Pacing η(t).

    Returns:
        float: dP/dt, the rate of change of Placeholder Expansion.
    """
    # Get current values of input functions
    C = C_func(t)
    S = S_func(t)
    E = E_func(t)
    eta = eta_func(t)

    # Ensure S is non-negative for log1p
    # In the ERS framework, S represents structural complexity, which is always positive or zero.
    S_clamped = max(0, S)

    # Calculate the logarithmic term for Structural Complexity
    # np.log1p(x) calculates log(1+x), which handles S=0 gracefully and
    # reflects the diminishing returns of raw structural complexity on expansion.
    log_S_term = np.log1p(S_clamped)

    # Calculate the denominator, which incorporates Entropic Dissonance
    # The (1 + beta * E) in the denominator is crucial. It ensures that as E (dissonance)
    # increases, it initially dampens expansion, but because it's a positive term,
    # it prevents division by zero and allows the system to 'integrate' paradox
    # rather than collapse, turning it into fuel at extreme levels.
    denominator = 1 + beta * E

    # Calculate dP/dt based on the PEF equation
    # This equation brings together all the components:
    # alpha: overall expansion drive
    # C: coherence (oscillatory, guides expansion)
    # log_S_term: structural complexity (drives growth, with diminishing returns)
    # denominator: entropic dissonance (integrates paradox)
    # eta: temporal pacing (controls the speed of expansion)
    dPdt = alpha * (C * log_S_term / denominator) * eta
    return dPdt

# --- Define time array ---
t = np.linspace(0, 200, 2001) # Time from 0 to 200, with 2001 points

# --- Define initial placeholder capacity ---
P0 = 0.1 # Starting point for Placeholder Expansion

# --- Define model parameters for Extreme Fractal Conflict ---
alpha_val = 1.0  # Ontological drive - standard
beta_val = 0.5   # Dissonance integration coefficient - standard
eta_val = 1.0    # Temporal pacing - standard

# --- Define C, S, E, eta functions for Extreme Fractal Conflict ---
# C(t): Highly oscillatory coherence, with periods of near-zero coherence.
# This simulates extreme internal conflict or existential doubt.
C_t_func = lambda time: (1 + 0.5 * np.sin(0.05 * time)) + 0.5 * np.sin(0.1 * time) - 0.5 * np.cos(0.02 * time)

# S(t): Exponentially growing structural complexity, pushing the system's limits.
# This represents rapid, potentially overwhelming, increase in internal structure.
S_t_func = lambda time: 5 * (1 - np.exp(-0.02 * time)) + 0.5 * np.sin(0.07 * time)

# E(t): EXTREMELY high random entropic dissonance (noise).
# This is the key change for this test, pushing the noise amplitude to 2.0.
# This simulates maximum, chaotic external and internal paradox.
E_noise_amp = 2.0 # Increased from 1.5 in previous Fractal Conflict
random_E_values = E_noise_amp * np.random.rand(len(t)) # Pure random noise
E_t_func = interp1d(t, random_E_values, kind='nearest', fill_value='extrapolate')

# eta(t): Hyper-pacing, pushing the system to evolve very quickly.
# This adds another layer of stress by demanding rapid processing.
eta_t_func = lambda time: np.full_like(time, 1.5) # Increased from 1.0

print("Starting Extreme Fractal Conflict World Test (E_noise_amp = 2.0)...")

# Solve the ODE
sol_extreme_fractal_conflict = odeint(P_dot, P0, t,
                                      args=(alpha_val, beta_val,
                                            C_t_func, S_t_func, E_t_func, eta_t_func))
P_extreme_fractal_conflict = sol_extreme_fractal_conflict[:, 0]

print(f" - Simulation complete for Extreme Fractal Conflict World.")

print("\nPlotting results...")

# --- Plotting P(t) and its components for Extreme Fractal Conflict ---
plt.figure(figsize=(14, 10))

# Plot P(t)
plt.subplot(2, 1, 1) # Two rows, one column, first plot
plt.plot(t, P_extreme_fractal_conflict, label='P(t) - Placeholder Expansion', color='red', linewidth=2)
plt.xlabel('Time')
plt.ylabel('P(t) - Placeholder Expansion')
plt.title('PEF Simulation: Extreme Fractal Conflict World (E_noise_amp = 2.0)')
plt.grid(True)
plt.legend()

# Plot the components C(t), S(t), E(t), eta(t)
plt.subplot(2, 1, 2) # Two rows, one column, second plot
plt.plot(t, C_t_func(t), label='C(t) - System Coherence', linestyle='--', color='blue', alpha=0.7)
plt.plot(t, S_t_func(t), label='S(t) - Structural Complexity', linestyle=':', color='green', alpha=0.7)
plt.plot(t, E_t_func(t), label='E(t) - Entropic Dissonance (Noise)', linestyle='-.', color='orange', alpha=0.7)
plt.plot(t, eta_t_func(t), label='η(t) - Temporal Pacing', linestyle='-', color='purple', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Component Values')
plt.title('Input Components for Extreme Fractal Conflict World')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print("\nExtreme Fractal Conflict Stress Test complete. Observe P(t)'s continued coherent growth despite even higher, chaotic dissonance.")
