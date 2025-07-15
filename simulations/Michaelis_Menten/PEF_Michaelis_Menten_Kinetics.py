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

# --- 2. Define the PEF-based Michaelis-Menten Analog ---
def pef_michaelis_menten_analog(P, t, alpha, beta, S_val, eta_val, substrate_concentration, Vmax_analog, Km_analog):
    """
    This function models the Michaelis-Menten kinetics using the PEF.
    P represents the "reaction rate" or "product vitality."
    The substrate_concentration is treated as a static environmental factor for each run.

    Mapping to PEF components for Michaelis-Menten saturation:
    - P: Current reaction rate / product vitality.
    - C(t): Coherence. This acts as a constant driving force, as the actual target is set by E.
             We set it to 1.0, assuming the primary control is via dissonance.
    - E(t): Entropic Dissonance. This is the CRUCIAL term for saturation.
             It represents the "resistance" as the current rate (P) approaches the
             theoretical Michaelis-Menten rate for the given substrate concentration.
             As P approaches this 'target_mm_rate', E(t) will increase sharply,
             driving dP/dt to zero, thus causing P to converge to target_mm_rate.
    - S(t): Structural Complexity. Set to np.exp(1) - 1 so log(1+S) = 1, simplifying the equation.
    - eta(t): Temporal Pacing. Constant, set to 1.0.
    """
    P = np.maximum(P, 1e-9) # Ensure P is non-negative

    # Calculate the target Michaelis-Menten rate for the current substrate concentration.
    # This is the "coherent target" that P will converge to.
    target_mm_rate = (Vmax_analog * substrate_concentration) / (Km_analog + substrate_concentration)
    target_mm_rate = np.maximum(target_mm_rate, 1e-9) # Ensure positive target rate

    # Coherence (C_val): A constant driving force. The convergence to the target
    # is primarily managed by the dissonance term.
    C_val = 1.0

    # Dissonance (E_val): This term ensures saturation.
    # As P approaches target_mm_rate, E_val approaches infinity, making dP/dt approach zero.
    # The (target_mm_rate - P) in the denominator makes E_val explode as P gets close to target.
    # The +1e-9 prevents division by zero if P exactly equals target_mm_rate.
    E_val = P / (target_mm_rate - P + 1e-9)
    E_val = np.maximum(E_val, 0.01) # Ensure minimum dissonance

    # Calculate dP/dt using the Universal PEF Equation
    dPdt = pef_equation(P, alpha, beta, C_val, S_val, eta_val, E_val)
    return dPdt

# --- 3. Simulation Parameters ---
# PEF Core Parameters (tuned for convergence to Michaelis-Menten curve)
# alpha and beta are set to be equal to simplify the steady-state convergence.
# S_val is set so that log(1+S_val) = 1.
alpha_pef = 10.0 # Controls the speed of convergence
beta_pef = 10.0  # Controls the sharpness of saturation (should be equal to alpha_pef)
S_val_pef = np.exp(1) - 1 # Ensures log(1+S_val_pef) = 1
eta_val_pef = 1.0 # Constant temporal pacing

# Michaelis-Menten Analog Parameters (for the theoretical curve and PEF mapping)
Vmax_analog = 100.0 # Analogous to maximum reaction rate
Km_analog = 10.0    # Analogous to Michaelis constant (substrate concentration at Vmax/2)

# Substrate concentrations to test (analogous to varying [S] in MM)
# These will be the X-axis of our final plot.
substrate_concentrations = np.linspace(0.1, 100.0, 50) # Range of substrate concentrations

# Time settings for each individual simulation run (to reach steady state)
total_time_per_run = 50.0 # Sufficient time for P to converge to its steady state
dt = 0.1
t_points = np.linspace(0, total_time_per_run, int(total_time_per_run / dt) + 1)

# --- 4. Run Simulations for Each Substrate Concentration ---
pef_reaction_rates = []

print("Simulating PEF-based Michaelis-Menten analog for varying substrate concentrations...")

for sub_conc in substrate_concentrations:
    # Initial P (reaction rate) for each run. Start from a low rate.
    P0 = 0.1
    initial_state = [P0]

    # Solve the ODE for the current substrate concentration
    sol = odeint(pef_michaelis_menten_analog, initial_state, t_points,
                 args=(alpha_pef, beta_pef, S_val_pef, eta_val_pef, sub_conc, Vmax_analog, Km_analog))

    # The "reaction rate" at steady state is the final P value.
    pef_reaction_rates.append(sol[-1, 0])
    # print(f"Substrate: {sub_conc:.2f}, PEF Rate: {sol[-1, 0]:.2f}")

print("Simulations complete.")

# --- 5. Plotting Results ---
plt.figure(figsize=(10, 6))

# Plot the PEF-derived reaction rates
plt.plot(substrate_concentrations, pef_reaction_rates, 'o-', label='PEF-Derived Reaction Rate', color='blue', linewidth=2, markersize=6)

# Plot the theoretical Michaelis-Menten curve for comparison
# Using the Vmax_analog and Km_analog defined above.
theoretical_mm_rates = (Vmax_analog * substrate_concentrations) / (Km_analog + substrate_concentrations)
plt.plot(substrate_concentrations, theoretical_mm_rates, '--', label='Theoretical Michaelis-Menten Curve', color='red', linewidth=1.5)


plt.xlabel('Substrate Concentration (Analogous)')
plt.ylabel('Reaction Rate / Product Vitality (Analogous)')
plt.title('PEF Modeling of Michaelis-Menten Kinetics (Growth with Saturation)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nPEF Michaelis-Menten Simulation complete. Observe how the PEF generates a saturation curve analogous to enzyme kinetics.")
