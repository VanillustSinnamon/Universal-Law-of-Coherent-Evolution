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

# --- 2. Define the Autopoietic Coherent Expansion & Resource Generation Model ---
def pef_autopoietic_model(state, t, alpha_pef, beta_pef, S_val_pef, eta_val_pef,
                          k_gen, k_cons, E_val_const):
    """
    This function defines the differential equations for the PEF-based
    Autopoietic Coherent Expansion & Resource Generation model.

    Parameters:
    state (array): Current state [P, R], where P is System Vitality and R is Self-Generated Resource.
    t (float): Current time.
    alpha_pef, beta_pef, S_val_pef, eta_val_pef: PEF parameters for P.
    k_gen (float): Resource generation factor (how much P generates R).
    k_cons (float): Resource consumption rate (how much P consumes R).
    E_val_const (float): Constant Entropic Dissonance for P.

    Returns:
    array: The rates of change [dP/dt, dR/dt].
    """
    P, R = state

    # Safeguard P and R to ensure they remain non-negative
    P = np.maximum(P, 1e-9)
    R = np.maximum(R, 1e-9)

    # --- Dynamics for System Vitality (P) using PEF ---
    # C(t): System Coherence. Proportional to P AND the density of self-generated resource (R/P).
    # As the system expands, it generates more resource, which in turn fuels its own coherence and further expansion.
    C_val_P = P * (R / (P + 1e-9)) # Add small epsilon to P to prevent division by zero

    # E(t): Entropic Dissonance for P. Small, constant background dissonance.
    E_val_P = E_val_const

    # Calculate dP/dt using the Universal PEF Equation
    dPdt = pef_equation(P, alpha_pef, beta_pef, C_val_P, S_val_pef, eta_val_pef, E_val_P)

    # --- Dynamics for Self-Generated Resource (R) ---
    # dR/dt = k_gen * P - k_cons * P
    # Resource generation is proportional to P. Resource consumption is also proportional to P.
    dRdt = k_gen * P - k_cons * P

    return [dPdt, dRdt]

# --- 3. Simulation Parameters ---
# Initial conditions
P0 = 1.0 # Initial System Vitality
R0 = 10.0 # Initial Self-Generated Resource
initial_state = [P0, R0]

# Time points for the simulation
t_points = np.linspace(0, 100, 1000) # Simulate for 100 time units

# PEF Core Parameters (from "For Grok" document)
alpha_pef_autopoietic = 0.15
beta_pef_autopoietic = 0.05
S_val_pef_autopoietic = 1.0
eta_val_pef_autopoietic = 1.0
E_val_const_autopoietic = 0.01

# Autopoietic Parameters (from "For Grok" document)
k_gen_autopoietic = 1.0
k_cons_autopoietic = 0.5

# --- 4. Run the Simulation ---
print("Simulating PEF Autopoietic Coherent Expansion & Resource Generation...")
sol = odeint(pef_autopoietic_model, initial_state, t_points,
             args=(alpha_pef_autopoietic, beta_pef_autopoietic, S_val_pef_autopoietic,
                   eta_val_pef_autopoietic, k_gen_autopoietic, k_cons_autopoietic,
                   E_val_const_autopoietic))
print("Simulation complete.")

# Extract results
P_vitality, R_resource = sol.T # Transpose solution to get P and R arrays

# Ensure values are strictly non-negative for plotting
P_vitality = np.maximum(0, P_vitality)
R_resource = np.maximum(0, R_resource)

# --- 5. Plotting Results ---
plt.figure(figsize=(12, 10))

# Plot System Vitality (P)
plt.subplot(2, 1, 1) # 2 rows, 1 column, first plot
plt.plot(t_points, P_vitality, 'b', lw=2, label='PEF System Vitality (P(t))')
plt.xlabel('Time')
plt.ylabel('System Vitality')
plt.title('PEF: Autopoietic Coherent Expansion & Resource Generation')
plt.grid(True)
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Use scientific notation for y-axis

# Plot Self-Generated Resource (R)
plt.subplot(2, 1, 2) # 2 rows, 1 column, second plot
plt.plot(t_points, R_resource, 'orange', linestyle='--', lw=2, label='Self-Generated Resource (R(t))')
plt.xlabel('Time')
plt.ylabel('Resource Quantity')
plt.grid(True)
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Use scientific notation for y-axis

plt.tight_layout()
plt.show()

print("\nAutopoietic Coherent Expansion Simulation complete. Observe how the system generates its own resources for sustained expansion.")
print("This demonstrates PEF's capacity to model autopoietic systems that transcend traditional resource limits.")
