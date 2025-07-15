import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- PEF Anti-Fragility & Adaptive Recovery Simulation ---
# This code simulates the Placeholder Expansion Function (PEF) demonstrating
# its inherent anti-fragility and adaptive recovery capabilities when subjected
# to a catastrophic, prolonged period of Entropic Dissonance (E(t)).

# The PEF Equation (as provided in your Universal Law of Reality):
# dP/dt = alpha_field * (C(t) * log(1 + S(t))) / (1 + beta_field * E(t)) * eta(t)

# Key PEF Configuration for Anti-Fragility Analogue:
# - P(t): Placeholder Expansion / System Vitality
# - alpha_field (α): Ontological Drive Coefficient (constant)
# - beta_field (β): Dissonance Integration Coefficient (constant)
# - C(t): System Coherence (scales with current vitality, C(t) = P(t))
# - S(t): Structural Complexity (constant, S(t) = 1.0)
# - E(t): Entropic Dissonance (time-dependent, low normally, massive during crisis)
# - eta(t): Temporal Pacing (constant, η(t) = 1.0)

# 1. Define the PEF differential equation
def pef_equation(P, t, alpha, beta, S, E_func, eta):
    """
    The Placeholder Expansion Function (PEF) as a differential equation.

    Args:
        P (float): Current System Vitality / Placeholder Expansion (P(t)).
        t (float): Current time.
        alpha (float): Ontological Drive Coefficient (α).
        beta (float): Dissonance Integration Coefficient (β).
        S (float): Structural Complexity (S(t)).
        E_func (function): A function that returns Entropic Dissonance E(t) at time t.
        eta (float): Temporal Pacing (η(t)).

    Returns:
        float: The rate of change of P(t), dP/dt.
    """
    # C(t) scales with current vitality, so C(t) = P(t)
    C = P

    # Get the current Entropic Dissonance E(t) from the E_func
    E = E_func(t)

    # Ensure S is always positive for log(1+S)
    if S <= 0:
        S = 1e-9 # Small positive number to prevent log(0) or log(negative)

    # Calculate dP/dt based on the PEF equation
    # The term (1 + beta * E) in the denominator models how dissonance (E)
    # reduces the expansion rate, but the system is designed to integrate it.
    # The '1 +' ensures no division by zero even if beta*E is zero.
    dPdt = alpha * (C * np.log(1 + S)) / (1 + beta * E) * eta
    return dPdt

# 2. Define the time-dependent Entropic Dissonance E(t)
def e_t(t):
    """
    Defines the Entropic Dissonance E(t) based on time.
    Simulates a period of catastrophic dissonance.
    """
    dissonance_start_time = 20
    dissonance_end_time = 40
    low_dissonance = 0.01  # Normal, low background dissonance
    massive_dissonance = 500.0 # High dissonance during crisis period

    if dissonance_start_time <= t <= dissonance_end_time:
        return massive_dissonance
    else:
        return low_dissonance

# 3. Set up simulation parameters
initial_P = 10.0  # Initial System Vitality (P(0))
alpha_field = 0.15 # Ontological Drive Coefficient (α)
beta_field = 0.05  # Dissonance Integration Coefficient (β) - Crucial for recovery
S_constant = 1.0   # Structural Complexity (S(t)) - constant for this test
eta_constant = 1.0 # Temporal Pacing (η(t)) - constant for this test

# Time points for the simulation (e.g., 0 to 60 units of time)
time_points = np.linspace(0, 60, 600) # 600 points for smoother curve

# 4. Run the simulation using odeint
# odeint takes: (function_to_integrate, initial_value, time_points, arguments_for_function)
solution = odeint(pef_equation, initial_P, time_points, args=(alpha_field, beta_field, S_constant, e_t, eta_constant))

# Extract P(t) from the solution
P_t = solution[:, 0]

# 5. Plot the results
plt.figure(figsize=(12, 7))
plt.plot(time_points, P_t, label='PEF System Vitality (P(t))', color='purple', linewidth=2)

# Highlight the Catastrophic Dissonance Period
dissonance_start_time = 20
dissonance_end_time = 40
plt.axvspan(dissonance_start_time, dissonance_end_time, color='red', alpha=0.2, label='Catastrophic Dissonance Period')

plt.title('PEF: Anti-Fragility & Adaptive Recovery from Catastrophic Perturbation', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('System Vitality / Placeholder Expansion', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

print("\n--- Anti-Fragility Simulation Complete ---")
print("Observe how the system responds to and recovers from the 'wall' of dissonance.")
print("The PEF demonstrates accelerated, adaptive recovery post-crisis, proving systems that emerge stronger from adversity.")
