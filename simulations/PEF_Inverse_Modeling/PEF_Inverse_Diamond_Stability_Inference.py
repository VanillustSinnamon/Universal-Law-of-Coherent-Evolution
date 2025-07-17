import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. Define the Universal PEF Equation ---
# This is the core equation, it operates on scalar P values.
def pef_equation_scalar(P_val, alpha, beta, C_val, S_val, eta_val, E_val):
    P_val = np.maximum(P_val, 1e-9)
    C_val = np.maximum(C_val, 1e-9)
    denominator = 1 + beta * E_val
    denominator = np.maximum(denominator, 1e-10)
    dPdt = alpha * (C_val * np.log1p(S_val) / denominator) * eta_val
    return dPdt

# Step 1: Simulate ultra-stable vitality curve (Diamond - Billions of Years)
t_data = np.linspace(0, 10 * 1e9, 1000) # 10 billion years
P_data = np.exp(-1e-12 * t_data) # Extremely slow exponential decay

# Step 2: Compute dP/dt from simulated data
dP_obs = np.gradient(P_data, t_data)

# Step 3: Define Reverse PEF Functional Form for curve_fit
# For an ultra-stable system with constant C, S, E, eta, the PEF predicts a constant dP/dt.
# This function will be passed to `curve_fit`.
# It must take `t` as the first argument, and then the parameters to be fitted.
# It will return a *scalar* value, which `curve_fit` will then broadcast to match `t_data`.
def reconstructed_constant_dP_dt_for_fit(t_dummy, alpha, beta, C_const, S_const, E_const, eta_const):
    # `t_dummy` is required by `curve_fit` signature but not used in calculation
    # because we are fitting a constant dP/dt.
    
    # Calculate the constant dP/dt that these PEF parameters would produce.
    # We use a representative P value (e.g., 1.0) for the PEF equation itself,
    # as the vitality is nearly constant in this simulation.
    constant_rate = pef_equation_scalar(1.0, alpha, beta, C_const, S_const, eta_const, E_const)
    
    return constant_rate # Return a single scalar value

# Step 4: Wrapper for Curve Fitting - Not needed, use reconstructed_constant_dP_dt_for_fit directly

# Step 5: Fit parameters
# Initial guesses reflect expected values for extreme stability:
# alpha (intrinsic change) should be very small, and *must be negative* for decay.
# beta (dissonance integration) should be very large
initial_guesses = [-1e-12, 1000.0, 1.0, 1.0, 0.01, 1.0] # Adjusted alpha to be negative initially
# Bounds: alpha *must* be negative, beta very large
lower_bounds = [-1e-10, 100.0, 0.01, 0.01, 0.001, 0.01] # Alpha can be negative
upper_bounds = [-1e-13, 1e5, 10.0, 10.0, 1.0, 10.0] # ***CRITICAL CHANGE: Upper bound for alpha is now negative***

popt, _ = curve_fit(
    reconstructed_constant_dP_dt_for_fit, # Directly pass the function to fit
    t_data,
    dP_obs,
    p0=initial_guesses,
    bounds=(lower_bounds, upper_bounds),
    maxfev=20000 # Increased iterations for potentially harder fit
)

# Step 6: Predict dP/dt using fitted PEF parameters
# Call the same function with the fitted parameters.
# Since it returns a scalar, we need to explicitly create an array of that scalar
# for plotting against t_data.
predicted_constant_rate = reconstructed_constant_dP_dt_for_fit(t_data[0], *popt) # Pass any t_dummy value
dP_pred = np.full_like(t_data, predicted_constant_rate) # Create an array of the constant rate

# Step 7: Plot vitality curve
plt.figure(figsize=(10, 5))
plt.plot(t_data / 1e9, P_data, label='Diamond Vitality Curve', color='black') # Plot in Billions of Years
plt.xlabel('Time (billions of years)')
plt.ylabel('Structural Integrity (P)')
plt.title('Simulated Longevity Curve of Diamond (Billions of Years)')
plt.grid(True)
plt.legend()
plt.ticklabel_format(style='plain', axis='x') # Prevent scientific notation on x-axis if desired
plt.show()

# Step 8: Plot observed vs predicted dP/dt
plt.figure(figsize=(10, 5))
plt.plot(t_data / 1e9, dP_obs, label='Observed dP/dt', color='blue') # Plot in Billions of Years
plt.plot(t_data / 1e9, dP_pred, label='Predicted dP/dt (Reverse PEF)', linestyle='--', color='orange') # Plot in Billions of Years
plt.xlabel('Time (billions of years)')
plt.ylabel('Rate of Structural Decay (dP/dt)')
plt.title('Reverse PEF Fit — Diamond-Like Stability (Billions of Years)')
plt.grid(True)
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Use scientific notation for y-axis
plt.ticklabel_format(style='plain', axis='x') # Prevent scientific notation on x-axis if desired
plt.show()

# Step 9: Print fitted parameters
print("🔬 Fitted PEF Parameters — Diamond Stability (Billions of Years)")
param_names = ['alpha', 'beta', 'C_const', 'S_const', 'E_const', 'eta_const']
for name, value in zip(param_names, popt):
    print(f"{name}: {value:.15f}") # More precision for very small/large numbers
