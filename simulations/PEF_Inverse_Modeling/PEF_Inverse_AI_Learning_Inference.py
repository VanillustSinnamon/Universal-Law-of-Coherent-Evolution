import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint

# --- 1. Define the Universal PEF Equation for ODE Integration ---
def pef_ode(P, t, alpha, beta, a, c, S_const, eta_const):
    P_val = np.maximum(P, 1e-9) 
    
    C = a * P_val                          
    S = S_const                            
    E = c * P_val # E now models linear dissonance with P
    eta = eta_const 

    numerator = C * np.log1p(S) 
    denominator = 1 + beta * E
    denominator = np.maximum(denominator, 1e-10) 

    dPdt = alpha * (numerator / denominator) * eta
    # ***CRITICAL CHANGE: Added saturation cap***
    dPdt *= (1 - P_val) # This term forces dPdt to zero as P_val approaches 1
    
    return dPdt

# 🧠 STEP 1: Simulate AI Learning Vitality Curve (REDUCED NOISE)
t_data = np.linspace(0, 100, 300) 
P_data_clean = 0.1 + 0.8 * (1 - np.exp(-0.05 * t_data)) 
np.random.seed(42) 
P_data = P_data_clean + np.random.normal(0, 0.005, size=t_data.shape) 
P_data = np.clip(P_data, 0.0, 1.0) 

# 🧠 STEP 2: Define Wrapper for Curve Fitting (Integrate PEF)
def integrated_pef_model(t, alpha, beta, a, c, S_const, eta_const):
    P0 = P_data[0] 
    P_integrated = odeint(pef_ode, P0, t, args=(alpha, beta, a, c, S_const, eta_const))
    return P_integrated.T[0]

# 🧠 STEP 3: Fit Parameters
# Adjust initial guesses and bounds for a smoother, better fit.
initial_guesses = [0.1, 1.0, 1.0, 1.0, 1.0, 1.0] 
lower_bounds = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001] 
# ***CRITICAL CHANGE: Expanded upper bounds as per Copilot's suggestion***
upper_bounds = [100.0, 1000.0, 100.0, 100.0, 100.0, 100.0] 

popt, _ = curve_fit(
    integrated_pef_model, 
    t_data,
    P_data, 
    p0=initial_guesses,
    bounds=(lower_bounds, upper_bounds),
    maxfev=100000 
)

# 🧠 STEP 4: Predict P and dP/dt using fitted PEF parameters
P_pred = odeint(pef_ode, P_data[0], t_data, args=tuple(popt)).T[0]
dP_pred = np.gradient(P_pred, t_data)

# 🧠 STEP 5: Plot Learning Curve (Observed vs Predicted P)
plt.figure(figsize=(10, 5))
plt.plot(t_data, P_data, label='Simulated AI Learning Vitality (Observed)', color='purple')
plt.plot(t_data, P_pred, label='Predicted AI Learning Vitality (PEF Fit)', linestyle='--', color='orange')
plt.xlabel('Epochs / Training Steps')
plt.ylabel('Accuracy (P)')
plt.title('AI Learning: Vitality Trajectory (Integral Fit - Saturation Cap)')
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
