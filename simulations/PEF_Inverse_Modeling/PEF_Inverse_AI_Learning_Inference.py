import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.stats import bootstrap # New import for bootstrapping

# --- 1. Define the Universal PEF Equation for ODE Integration ---
def pef_ode(P, t, alpha, beta, a, c, S_const, eta_const):
    P_val = np.maximum(P, 1e-9) 
    
    C = a * P_val                          
    S = S_const                            
    E = c * P_val 
    eta = eta_const 

    numerator = C * np.log1p(S) 
    denominator = 1 + beta * E
    denominator = np.maximum(denominator, 1e-10) 

    dPdt = alpha * (numerator / denominator) * eta
    dPdt *= (1 - P_val) # Saturation Cap: Forces dPdt to zero as P_val approaches 1
    
    return dPdt

# 🧠 STEP 1: Simulate AI Learning Vitality Curve (REDUCED NOISE)
t_data = np.linspace(0, 100, 300) 
P_data_clean = 0.1 + 0.8 * (1 - np.exp(-0.05 * t_data)) 
np.random.seed(42) 
P_data = P_data_clean + np.random.normal(0, 0.005, size=t_data.shape) 
P_data = np.clip(P_data, 0.0, 1.0) 

# 🧠 STEP 2: Define Wrapper for Curve Fitting (Integrate PEF)
# ***CRITICAL CHANGE: integrated_pef_model now accepts P_data_local as an argument***
def integrated_pef_model(t, alpha, beta, a, c, S_const, eta_const, P_data_local):
    P0 = P_data_local[0] # Initial condition from the specific P_data being used for this fit
    P_integrated = odeint(pef_ode, P0, t, args=(alpha, beta, a, c, S_const, eta_const))
    return P_integrated.T[0]

# 🧠 STEP 3: Fit Parameters (Initial Fit)
initial_guesses = [0.1, 1.0, 1.0, 1.0, 1.0, 1.0] 
lower_bounds = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
upper_bounds = [100.0, 1000.0, 100.0, 100.0, 100.0, 100.0] 

# ***CRITICAL CHANGE: Pass P_data explicitly as an argument to curve_fit's args***
popt, pcov = curve_fit(
    lambda t, alpha, beta, a, c, S_const, eta_const: integrated_pef_model(t, alpha, beta, a, c, S_const, eta_const, P_data),
    t_data,
    P_data, 
    p0=initial_guesses,
    bounds=(lower_bounds, upper_bounds),
    maxfev=100000 
)

# 🧠 STEP 4: Predict P and dP/dt using fitted PEF parameters
P_pred = odeint(pef_ode, P_data[0], t_data, args=tuple(popt)).T[0]
dP_pred = np.gradient(P_pred, t_data)

# --- Calculate Confidence Intervals using Bootstrapping ---
rng = np.random.default_rng(seed=42) # For reproducibility of bootstrap
bootstrap_samples = []
n_resamples = 100 # Number of bootstrap resamples (can increase for more precision, but takes longer)

print(f"Calculating {n_resamples} bootstrap resamples for confidence intervals... This may take a moment.")
for _ in range(n_resamples):
    # Resample indices with replacement
    resample_indices = rng.integers(0, len(t_data), size=len(t_data))
    t_resampled = t_data[resample_indices]
    P_resampled = P_data[resample_indices]
    
    # Sort resampled data by time to ensure odeint works correctly
    sort_indices = np.argsort(t_resampled)
    t_resampled = t_resampled[sort_indices]
    P_resampled = P_resampled[sort_indices]

    try:
        # Fit model to resampled data
        popt_resample, _ = curve_fit(
            # ***CRITICAL CHANGE: Pass P_resampled explicitly to the lambda function***
            lambda t, alpha, beta, a, c, S_const, eta_const: integrated_pef_model(t, alpha, beta, a, c, S_const, eta_const, P_resampled),
            t_resampled,
            P_resampled,
            p0=popt, # Use previously fitted popt as initial guess for speed
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000 # Reduced maxfev for bootstrap fits to speed up
        )
        # Predict P_data using the resampled parameters over the original t_data
        # ***CRITICAL CHANGE: Use P_data[0] from original data for initial condition here***
        bootstrap_samples.append(odeint(pef_ode, P_data[0], t_data, args=tuple(popt_resample)).T[0])
    except RuntimeError as e:
        print(f"Warning: Bootstrap fit failed for a resample: {e}")
        continue

if len(bootstrap_samples) > 0: # ***CRITICAL FIX: Check length of list before converting to array***
    bootstrap_samples = np.array(bootstrap_samples)
    # Calculate 2.5th and 97.5th percentiles for 95% confidence interval
    ci_lower = np.percentile(bootstrap_samples, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_samples, 97.5, axis=0)
else:
    print("Warning: No successful bootstrap samples. Confidence interval will not be plotted.")
    ci_lower = P_pred # Fallback to predicted line if no samples
    ci_upper = P_pred


# 🧠 STEP 5: Plot Learning Curve (Observed vs Predicted P with CI)
plt.figure(figsize=(10, 5))
plt.plot(t_data, P_data, label='Simulated AI Learning Vitality (Observed)', color='purple')
plt.plot(t_data, P_pred, label='Predicted AI Learning Vitality (PEF Fit)', linestyle='--', color='orange')
if len(bootstrap_samples) > 0: # ***CRITICAL FIX: Check length again before plotting CI***
    plt.fill_between(t_data, ci_lower, ci_upper, color='orange', alpha=0.2, label='95% CI')
plt.xlabel('Epochs / Training Steps')
plt.ylabel('Accuracy (P)')
plt.title('AI Learning: Vitality Trajectory (Integral Fit with Saturation Cap & CI)')
plt.grid(True)
plt.legend()
plt.show()

# 🧠 STEP 6: Plot Observed vs Predicted dP/dt
dP_obs_from_data = np.gradient(P_data, t_data) 
plt.figure(figsize=(10, 5))
plt.plot(t_data, dP_obs_from_data, label='Observed dP/dt (from data)', color='blue')
plt.plot(t_data, dP_pred, label='Predicted dP/dt (from PEF Fit)', linestyle='--', color='orange')
plt.xlabel('Epochs / Training Steps')
plt.ylabel('Rate of Change (dP/dt)')
plt.title('Reverse PEF Fit — AI Learning Dynamics (Integral Fit - Saturation Cap)')
plt.grid(True)
plt.legend()
plt.show()

# 🧠 STEP 7: Display Fitted Parameters
print("🔬 Fitted PEF Parameters — AI Learning (Integral Fit - Saturation Cap):")
param_names = ['alpha', 'beta', 'a', 'c', 'S_const', 'eta_const'] 
for name, value in zip(param_names, popt):
    print(f"{name}: {value:.6f}")
