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

# --- 2. PEF-driven Rates for Chemical Reaction (A <=> B) ---
def pef_chemical_reaction_rates(concentrations, t, alpha_fwd, beta_fwd, S_val_fwd, eta_val_fwd,
                                                alpha_rev, beta_rev, S_val_rev, eta_val_rev):
    """
    This function defines the differential equations for a simple reversible chemical reaction (A <=> B),
    where the forward and reverse reaction rates are governed by instances of the PEF.

    Parameters:
    concentrations (array): Current concentrations [A, B].
    t (float): Current time.
    alpha_fwd, beta_fwd, S_val_fwd, eta_val_fwd: PEF parameters for the forward reaction (A -> B).
    alpha_rev, beta_rev, S_val_rev, eta_val_rev: PEF parameters for the reverse reaction (B -> A).

    Returns:
    array: The rates of change [dA/dt, dB/dt].
    """
    A, B = concentrations

    # Ensure concentrations are non-negative for PEF calculations
    A = np.maximum(A, 1e-9)
    B = np.maximum(B, 1e-9)

    # --- PEF for Forward Reaction Rate (A -> B) ---
    # P_fwd_pef: Represents the current "vitality" of the forward reaction.
    # We can use A as the P for the forward PEF, as it's the reactant driving the forward process.
    P_fwd_pef = A

    # C_fwd (Coherence for Forward Reaction): Proportional to reactant A's concentration.
    # Higher A means more "coherence" for the forward reaction to proceed.
    C_fwd = A

    # E_fwd (Dissonance for Forward Reaction): Represents "resistance" to the forward reaction,
    # which increases as product B accumulates (reverse pressure) or as A depletes.
    # We'll use B as dissonance, as it represents the "back pressure" from the product.
    E_fwd = B # Dissonance increases with product B concentration

    # S_val_fwd, eta_val_fwd are constant PEF parameters for forward reaction.

    # Calculate the rate of change of the forward reaction potential
    forward_rate_pef = pef_equation(P_fwd_pef, alpha_fwd, beta_fwd, C_fwd, S_val_fwd, eta_val_fwd, E_fwd)


    # --- PEF for Reverse Reaction Rate (B -> A) ---
    # P_rev_pef: Represents the current "vitality" of the reverse reaction.
    # We use B as the P for the reverse PEF, as it's the reactant driving the reverse process.
    P_rev_pef = B

    # C_rev (Coherence for Reverse Reaction): Proportional to reactant B's concentration.
    # Higher B means more "coherence" for the reverse reaction to proceed.
    C_rev = B

    # E_rev (Dissonance for Reverse Reaction): Represents "resistance" to the reverse reaction,
    # which increases as product A accumulates (forward pressure) or as B depletes.
    # We'll use A as dissonance, as it represents the "back pressure" from the product.
    E_rev = A # Dissonance increases with product A concentration

    # S_val_rev, eta_val_rev are constant PEF parameters for reverse reaction.

    # Calculate the rate of change of the reverse reaction potential
    reverse_rate_pef = pef_equation(P_rev_pef, alpha_rev, beta_rev, C_rev, S_val_rev, eta_val_rev, E_rev)


    # --- Chemical Reaction Differential Equations ---
    # Net change for A is (reverse rate - forward rate)
    dAdt = reverse_rate_pef - forward_rate_pef
    # Net change for B is (forward rate - reverse rate)
    dBdt = forward_rate_pef - reverse_rate_pef

    # Ensure concentrations remain non-negative (odeint will usually handle this, but good practice)
    # This is a simple check to prevent negative values from propagating if numerical issues arise.
    if (A + dAdt * (t_points[1] - t_points[0])) < 0:
        dAdt = -A / (t_points[1] - t_points[0])
    if (B + dBdt * (t_points[1] - t_points[0])) < 0:
        dBdt = -B / (t_points[1] - t_points[0])

    return [dAdt, dBdt]

# --- 3. Simulation Parameters ---
# Initial concentrations
A0 = 100.0 # Start with all A
B0 = 0.0   # Start with no B
initial_concentrations = [A0, B0]

# Time points for the simulation
t_points = np.linspace(0, 50, 500) # Simulate for 50 time units

# PEF Parameters for Forward Reaction (A -> B)
# Tuned to drive A towards B
alpha_fwd_pef = 0.1
beta_fwd_pef = 0.5
S_val_fwd_pef = np.exp(1) - 1 # log(1+S) = 1
eta_val_fwd_pef = 1.0

# PEF Parameters for Reverse Reaction (B -> A)
# Tuned to drive B towards A, note alpha_rev is slightly lower for equilibrium favoring B
alpha_rev_pef = 0.08 # Slightly lower alpha for reverse to favor forward equilibrium
beta_rev_pef = 0.5
S_val_rev_pef = np.exp(1) - 1 # log(1+S) = 1
eta_val_rev_pef = 1.0

# --- 4. Run the Simulation ---
print("Simulating PEF-based Chemical Reaction Dynamics (A <=> B)...")
sol = odeint(pef_chemical_reaction_rates, initial_concentrations, t_points,
             args=(alpha_fwd_pef, beta_fwd_pef, S_val_fwd_pef, eta_val_fwd_pef,
                   alpha_rev_pef, beta_rev_pef, S_val_rev_pef, eta_val_rev_pef))
print("Simulation complete.")

# Extract results
A_conc, B_conc = sol.T # Transpose solution to get A and B concentration arrays

# Ensure concentrations are strictly non-negative for plotting
A_conc = np.maximum(0, A_conc)
B_conc = np.maximum(0, B_conc)


# --- 5. Plotting Results ---
plt.figure(figsize=(12, 7))
plt.plot(t_points, A_conc, 'b', alpha=0.7, lw=2, label='Concentration of A')
plt.plot(t_points, B_conc, 'r', alpha=0.7, lw=2, label='Concentration of B')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('PEF Modeling of Chemical Reaction Dynamics (A <=> B)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nPEF Chemical Reaction Dynamics Simulation complete. Observe how the concentrations evolve towards equilibrium.")
