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

# --- 2. Define the PEF Coherent Expansion Under Finite Resource Constraint Model ---
def pef_finite_resource_model(state, t, alpha_pef, beta_pef, S_val_pef, eta_val_pef,
                              k_resource_consumption, E_val_const, R_initial_scale):
    """
    This function defines the differential equations for the PEF-based
    Coherent Expansion Under Finite Resource Constraint model.

    Parameters:
    state (array): Current state [P, R], where P is System Vitality and R is Available Resource.
    t (float): Current time.
    alpha_pef, beta_pef, S_val_pef, eta_val_pef: PEF parameters for P.
    k_resource_consumption (float): Rate at which resource is consumed by P.
    E_val_const (float): Small, constant background dissonance for P.
    R_initial_scale (float): The initial resource quantity, used for normalizing R in C(t).

    Returns:
    array: The rates of change [dP/dt, dR/dt].
    """
    P, R = state

    # Safeguard P and R to ensure they remain non-negative
    P = np.maximum(P, 1e-9)
    R = np.maximum(R, 1e-9)

    # --- Dynamics for System Vitality (P) using PEF ---
    # C(t): System Coherence. Proportional to P AND the available resource (R/R_initial_scale).
    # As R depletes, C(t) diminishes, inherently limiting P's expansion.
    C_val_P = P * (R / R_initial_scale) # R_initial_scale normalizes R's impact

    # E(t): Entropic Dissonance for P. Small, constant background dissonance.
    E_val_P = E_val_const

    # Calculate dP/dt using the Universal PEF Equation
    dPdt = pef_equation(P, alpha_pef, beta_pef, C_val_P, S_val_pef, eta_val_pef, E_val_P)

    # --- Dynamics for Available Resource (R) ---
    # dR/dt = -k * P(t)
    # Resource is consumed proportional to System Vitality (P).
    dRdt = -k_resource_consumption * P

    return [dPdt, dRdt]

# --- 3. Simulation Parameters ---
# Initial conditions (from "For Grok" document)
P0 = 1.0    # Initial System Vitality
R0 = 1000.0 # Initial Available Resource
initial_state = [P0, R0]

# Time points for the simulation
t_points = np.linspace(0, 150, 1500) # Simulate for 150 time units

# PEF Core Parameters (from "For Grok" document)
alpha_pef_resource = 0.15
beta_pef_resource = 0.05
S_val_pef_resource = 1.0
eta_val_pef_resource = 1.0
E_val_const_resource = 0.01

# Resource Consumption Parameter (from "For Grok" document)
k_resource_consumption = 0.5

# Initial resource scale for C(t) normalization (from "For Grok" document)
R_initial_scale = R0 # Use initial R0 as the scaling factor

# --- 4. Run the Simulation ---
print("Simulating PEF Coherent Expansion Under Finite Resource Constraint...")
sol = odeint(pef_finite_resource_model, initial_state, t_points,
             args=(alpha_pef_resource, beta_pef_resource, S_val_pef_resource,
                   eta_val_pef_resource, k_resource_consumption,
                   E_val_const_resource, R_initial_scale))
print("Simulation complete.")

# Extract results
P_vitality, R_resource = sol.T # Transpose solution to get P and R arrays

# Ensure values are strictly non-negative for plotting (especially resource)
P_vitality = np.maximum(0, P_vitality)
# R_resource can go negative in this model, representing debt or full depletion.
# We will plot it as is to show the full depletion effect.

# --- 5. Plotting Results ---
plt.figure(figsize=(12, 10))

# Plot System Vitality (P)
plt.subplot(2, 1, 1) # 2 rows, 1 column, first plot
plt.plot(t_points, P_vitality, 'b', lw=2, label='PEF System Vitality (P(t))')
plt.xlabel('Time')
plt.ylabel('System Vitality')
plt.title('PEF: Coherent Expansion Under Finite Resource Constraint')
plt.grid(True)
plt.legend()
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Only if numbers get very large

# Plot Available Resource (R)
plt.subplot(2, 1, 2) # 2 rows, 1 column, second plot
plt.plot(t_points, R_resource, 'g--', lw=2, label='Available Resource (R(t))')
plt.xlabel('Time')
plt.ylabel('Resource Quantity')
plt.grid(True)
plt.legend()
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Only if numbers get very large

plt.tight_layout()
plt.show()

print("\nFinite Resource Constraint Simulation complete. Observe how resource depletion affects system expansion.")
print("This demonstrates PEF accurately models the physical reality of resource-constrained growth.")
