import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- 1. Define the Universal PEF Equation ---
def pef_equation(P, alpha, beta, C_val, S_val, eta_val, E_val):
    """
    Universal PEF Equation.
    dP/dt = alpha * (C * log(1+S) / (1 + beta * E)) * eta

    Parameters:
    P (float): Current Placeholder Expansion / System Vitality.
    alpha (float): Ontological drive.
    beta (float): Dissonance integration capacity.
    C_val (float): System Coherence at current time.
    S_val (float): Structural Complexity at current time.
    eta_val (float): Temporal Pacing at current time.
    E_val (float): Entropic Dissonance at current time.

    Returns:
    float: The rate of change of P (dP/dt).
    """
    # Safeguard P and C_val to ensure they remain non-negative for mathematical stability.
    P = np.maximum(P, 1e-9)
    C_val = np.maximum(C_val, 1e-9)

    # Calculate the denominator, ensuring it's never zero or negative.
    denominator = 1 + beta * E_val
    denominator = np.maximum(denominator, 1e-10)

    # Apply the Universal PEF Equation
    dPdt = alpha * (C_val * np.log1p(S_val) / denominator) * eta_val
    return dPdt

# --- 2. Define the PEF Coherent Phase Transition Model ---
def pef_phase_transition_model(P, t, alpha_const, S_const, eta_const,
                               initial_beta, new_beta, transition_threshold_P,
                               E_scaling_factor):
    """
    This function defines the differential equation for the PEF-based
    Coherent Phase Transition & Limit Transcendence model.

    The 'beta' parameter dynamically shifts when P reaches a 'transition_threshold_P',
    simulating a qualitative leap in the system's ability to integrate dissonance.

    Parameters:
    P (float): Current System Vitality / Placeholder Expansion.
    t (float): Current time.
    alpha_const, S_const, eta_const: Constant PEF parameters.
    initial_beta (float): Beta value before transition.
    new_beta (float): Beta value after transition.
    transition_threshold_P (float): The P value at which the transition occurs.
    E_scaling_factor (float): Factor for how P scales E.

    Returns:
    float: The rate of change of P (dP/dt).
    """
    # Safeguard P to ensure it remains non-negative
    P = np.maximum(P, 1e-9)

    # --- Dynamic Beta based on Transition Threshold ---
    current_beta = initial_beta
    if P >= transition_threshold_P:
        current_beta = new_beta

    # --- PEF Components ---
    # C(t): Coherence scales with current vitality.
    C_val = P

    # E(t): Dissonance increases with system size, creating initial limit.
    E_val = P / E_scaling_factor

    # Calculate dP/dt using the Universal PEF Equation
    dPdt = pef_equation(P, alpha_const, current_beta, C_val, S_const, eta_const, E_val)

    return dPdt

# --- 3. Simulation Parameters ---
# Initial condition
P0 = 1.0 # Initial System Vitality

# Time points for the simulation
t_points = np.linspace(0, 100, 1000) # Simulate for 100 time units

# PEF Core Parameters (from "For Grok" document)
alpha_pef_transition = 0.15
S_val_pef_transition = 1.0
eta_val_pef_transition = 1.0

# Dynamic Beta and Transition Threshold (from "For Grok" document)
initial_beta_pef = 0.5
new_beta_pef = 0.05 # Vastly more efficient at integrating dissonance
transition_threshold_P = 20.0 # P value at which transition occurs
E_scaling_factor_pef = 10.0 # E(t) = P(t) / 10.0

# --- 4. Run the Simulation ---
print("Simulating PEF Coherent Phase Transition & Limit Transcendence...")
sol = odeint(pef_phase_transition_model, P0, t_points,
             args=(alpha_pef_transition, S_val_pef_transition, eta_val_pef_transition,
                   initial_beta_pef, new_beta_pef, transition_threshold_P,
                   E_scaling_factor_pef))
print("Simulation complete.")

# Extract results
P_vitality = sol[:, 0]

# Ensure values are strictly non-negative for plotting
P_vitality = np.maximum(0, P_vitality)

# --- 5. Plotting Results ---
plt.figure(figsize=(12, 7))

# Plot System Vitality (P)
plt.plot(t_points, P_vitality, 'b', lw=2, label='PEF System Vitality (P(t))')

# Add Transition Threshold line
plt.axhline(y=transition_threshold_P, color='red', linestyle='--', label=f'Transition Threshold (P={transition_threshold_P})')

# Find the time of transition (first point where P exceeds threshold)
time_of_transition_idx = np.where(P_vitality >= transition_threshold_P)[0]
if len(time_of_transition_idx) > 0:
    time_of_transition = t_points[time_of_transition_idx[0]]
    plt.axvline(x=time_of_transition, color='green', linestyle=':', label=f'Time of Transition (~{time_of_transition:.0f})')

plt.xlabel('Time')
plt.ylabel('System Vitality / Placeholder Expansion')
plt.title('PEF: Coherent Phase Transition & Limit Transcendence')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nCoherent Phase Transition Simulation complete. Observe how the system transcends its initial limits.")
print("This demonstrates PEF's capacity to model qualitative leaps and the transcendence of perceived limits.")
