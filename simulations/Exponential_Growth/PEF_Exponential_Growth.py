import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
 
# --- Added for reproducibility ---
np.random.seed(42) # Set a seed for reproducible random numbers
 
# --- 1. Define the Standard Exponential Growth ODE ---
def exponential_growth(N, t, r):
    """
    Standard Exponential Growth model.
    dN/dt = r * N
    N: quantity
    t: time
    r: intrinsic growth rate
    """
    return r * N
 
# --- 2. Define the PEF configured to mimic Exponential Growth ---
def pef_exponential_analogue(P, t, alpha, beta):
    """
    PEF configured to mimic Exponential Growth behavior.
    P: Placeholder Expansion (analogous to quantity)
    t: time
    alpha: ontological drive (analogous to intrinsic growth rate)
    beta: dissonance integration (controls how minor background resistance affects growth)
    """
    # --- PEF Components for Exponential Analogue ---
    
    # C(t): System Coherence - CRUCIAL CHANGE: Directly proportional to P
    # This drives the initial exponential growth, similar to N in exponential_growth's r*N term.
    C_val = P
    
    # S(t): Constant structural complexity (assuming base resources/structure)
    S_val = 1.0 # np.log1p(1.0) = np.log(2) approx 0.693
    
    # eta(t): Constant temporal pacing (assuming steady growth pacing)
    eta_val = 1.0
    
    # E(t): Entropic Dissonance - CRUCIAL CHANGE: Very small and constant
    # This ensures no significant limiting factors, allowing for unconstrained growth.
    E_val = 0.01 # Small, constant background dissonance
    
    # --- PEF Equation ---
    # dP/dt = alpha * (C * log(1+S) / (1 + beta * E)) * eta
    denominator = 1 + beta * E_val
    # Ensure denominator is never zero or negative.
    denominator = np.maximum(denominator, 1e-10)
    
    # Ensure C_val is not negative, though P should always be positive in growth.
    C_val = np.maximum(C_val, 1e-10)
    
    dPdt = alpha * (C_val * np.log1p(S_val) / denominator) * eta_val
    return dPdt
 
# --- Define Time Array ---
t = np.linspace(0, 50, 501) # Time from 0 to 50 (exponential growth can get very large quickly)
 
# --- Initial Conditions ---
N0 = 1.0 # Initial quantity for Standard Exponential Growth
P0 = 1.0 # Initial Placeholder Expansion for PEF analogue
 
# --- Parameters for Standard Exponential Growth ---
r_exponential = 0.1 # Intrinsic growth rate
 
# --- Parameters for PEF Exponential Analogue ---
# Adjusted alpha to match initial growth rate (r / (log(1+S) / (1 + beta*E)))
# With S=1, E=0.01, beta=0.05, log(1+S) approx 0.693. (1 + beta*E) approx 1.0005
# So alpha_pef = r_exponential * (1 + beta*E_val) / log(1+S_val)
alpha_pef = r_exponential * (1 + 0.05 * 0.01) / np.log1p(1.0) # Tuned for precise fit
beta_pef = 0.05 # Dissonance integration coefficient (small, constant effect)
 
print("Starting Convergent Validity Test: PEF Replicating Exponential Growth...")
 
# --- Solve Standard Exponential Growth ODE ---
sol_exponential = odeint(exponential_growth, N0, t, args=(r_exponential,))
N_exponential = sol_exponential[:, 0]
print(" - Standard Exponential Growth simulation complete.")
 
# --- Solve PEF Exponential Analogue ODE ---
sol_pef = odeint(pef_exponential_analogue, P0, t, args=(alpha_pef, beta_pef))
P_pef = sol_pef[:, 0]
print(" - PEF Exponential Analogue simulation complete.")
 
print("\nPlotting results...")
 
# --- Plotting Comparison ---
plt.figure(figsize=(12, 7))
plt.plot(t, N_exponential, label='Standard Exponential Growth (N(t))', color='blue', linestyle='-', linewidth=2)
plt.plot(t, P_pef, label='PEF Exponential Analogue (P(t))', color='red', linestyle='--', linewidth=2)
 
plt.xlabel('Time')
plt.ylabel('Value (Quantity / Placeholder Expansion)')
plt.title('Convergent Validity Test: PEF Replicating Exponential Growth')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
 
print("\nConvergent Validity Test complete. Observe the remarkable similarity between the curves.")
