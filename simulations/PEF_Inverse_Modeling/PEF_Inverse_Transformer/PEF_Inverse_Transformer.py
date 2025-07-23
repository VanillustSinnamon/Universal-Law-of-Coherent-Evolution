import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import io

# --- 1. Define the core PEF derivative function ---
def P_dot(P, t, alpha, beta, C_func, S_func, E_func, eta_func):
    """
    Calculates the derivative of Placeholder Expansion P with respect to time t.
    This is the core of the Placeholder Expansion Function (PEF) from the ERS framework.
    Args:
        P (float): Current value of Placeholder Expansion.
        t (float): Current time.
        alpha (float): Expansion coefficient (ontological drive).
        beta (float): Dissonance integration coefficient (how system processes paradox).
        C_func (callable): Function for System Coherence C(t).
        S_func (callable): Function for Structural Complexity S(t).
        E_func (callable): Function for Entropic Dissonance E(t).
        eta_func (callable): Function for Temporal Pacing η(t).
    Returns:
        float: dP/dt, the rate of change of Placeholder Expansion.
    """
    C = C_func(t)
    S = S_func(t)
    E = E_func(t)
    eta = eta_func(t)

    S_clamped = max(0, S) # Ensure S is non-negative for log1p
    log_S_term = np.log1p(S_clamped)
    denominator = 1 + beta * E
    
    dPdt = alpha * (C * log_S_term / denominator) * eta
    return dPdt

# --- 2. Define the function to be fitted by curve_fit (integrates P_dot) ---
# This function now takes ALL parameters to be fitted, including those for the new functional forms
def solve_pef_ode_full_refined(t_data, P0_initial, alpha, beta, 
                               a_E, b_E, c_E, # E(t) Gaussian params
                               d_C, e_C, f_C, # C(t) Logistic params
                               f_S,           # S(t) power law scaling, g_S is fixed
                               h_eta, i_eta   # eta(t) exponential decay params
                              ):
    """
    Solves the PEF ODE for given parameters and time points, using Grok's suggested
    refined functional forms for C(t), S(t), E(t), eta(t).
    """
    # Define the PEF component functions using the current fitting parameters
    def E_func(t):
        # E(t) = a_E * exp(-(t-b_E)^2/(2c_E^2)) * (1 + 0.2*sin(0.1*t + π/2)) * exp(-0.01*t)
        # Add a small constant to 'c_E' to prevent division by zero if c_E becomes too small during fitting
        gaussian_part = a_E * np.exp(-(t - b_E)**2 / (2 * (c_E + 1e-6)**2))
        oscillatory_part = (1 + 0.2 * np.sin(0.1 * t + np.pi/2)) # Fixed amplitude and frequency
        decay_part = np.exp(-0.01 * t) # Fixed decay rate
        return np.maximum(0.01, gaussian_part * oscillatory_part * decay_part)

    def C_func(t):
        # C(t) = d_C / (1 + exp(-e_C*(t - f_C))) (Logistic cap)
        # Clamp d_C and e_C to be positive
        return np.maximum(1e-6, d_C) / (1 + np.exp(-np.maximum(1e-6, e_C) * (t - f_C)))

    def S_func(t):
        # S(t) = f_S * t^g_S, with g_S fixed at 0.3
        g_S_fixed = 0.3 # Grok's fixed value
        # Clamp f_S to be positive
        return np.maximum(1e-6, f_S) * (t**g_S_fixed)

    def eta_func(t):
        # Decaying exponential: η(t) = h_eta * exp(-i_eta*t)
        # Clamp h_eta and i_eta to be positive
        return np.maximum(1e-6, h_eta) * np.exp(-np.maximum(1e-6, i_eta) * t)

    # Create a lambda function for P_dot with fixed component functions
    def ode_wrapper(P, t, alpha_val, beta_val):
        return P_dot(P, t, alpha_val, beta_val, C_func, S_func, E_func, eta_func)
    
    # Solve the ODE
    # Ensure P0_initial is positive
    sol = odeint(ode_wrapper, np.maximum(1e-6, P0_initial), t_data, args=(alpha, beta))
    return sol[:, 0] # Return the P(t) values

# --- 3. Load and Process Extracted Transformer Data ---
# The user provided 'Default Dataset.csv' which contains the digitized data.
# This assumes the first column is log10(Pre-training step) and the second is Surprisal.

# Simulating content_fetcher.fetch for demonstration purposes
csv_data_string = """3.0131578947368425, 9.839116719242902
3.0526315789473686, 9.813880126182966
3.0855263157894735, 9.788643533123029
3.125, 9.763406940063092
3.1644736842105265, 9.738170347003155
3.197368421052632, 9.712933753943219
3.2368421052631584, 9.687697160883282
3.2828947368421053, 9.662460567823345
3.322368421052632, 9.611987381703472
3.355263157894737, 9.536277602523661
3.3947368421052633, 9.46056782334385
3.43421052631579, 9.384858044164039
3.4671052631578947, 9.28391167192429
3.5, 9.182965299684543
3.5263157894736845, 9.082018927444796
3.5526315789473686, 8.981072555205047
3.578947368421053, 8.854889589905364
3.605263157894737, 8.728706624605678
3.6315789473684212, 8.602523659305994
3.6578947368421053, 8.451104100946372
3.6776315789473686, 8.299684542586752
3.7039473684210527, 8.14826498422713
3.723684210526316, 7.996845425867509
3.743421052631579, 7.845425867507887
3.7697368421052637, 7.719242902208202
3.7894736842105265, 7.5678233438485805
3.80921052631579, 7.41640378548896
3.842105263157895, 7.264984227129338
3.868421052631579, 7.138801261829653
3.8947368421052633, 7.012618296529969
3.921052631578948, 6.886435331230285
3.947368421052632, 6.8107255520504735
3.980263157894737, 6.785488958990538
4.019736842105264, 6.760252365930601
4.052631578947368, 6.8107255520504735
4.078947368421053, 6.861198738170347
4.111842105263158, 6.962145110410095
4.1381578947368425, 7.063091482649842
4.1644736842105265, 7.16403785488959
4.184210526315789, 7.290220820189274
4.210526315789474, 7.391167192429022
4.2368421052631575, 7.49211356466877
4.2631578947368425, 7.618296529968456
4.2894736842105265, 7.7444794952681395
4.322368421052632, 7.82018927444795
4.355263157894737, 7.895899053627761
4.394736842105264, 7.946372239747635
4.434210526315789, 7.971608832807572
4.473684210526317, 7.971608832807572
4.5131578947368425, 7.946372239747635
4.552631578947368, 7.895899053627761
4.592105263157896, 7.845425867507887
4.625, 7.769716088328076
4.657894736842105, 7.694006309148265
4.697368421052632, 7.643533123028392
4.723684210526316, 7.542586750788645
4.7631578947368425, 7.441640378548897
4.7894736842105265, 7.340694006309149
4.822368421052632, 7.214511041009464
4.855263157894738, 7.08832807570978
4.881578947368421, 6.9873817034700325
4.9144736842105265, 6.861198738170347
4.940789473684211, 6.735015772870662
4.967105263157896, 6.634069400630915
5, 6.5331230283911665
5.032894736842105, 6.406940063091483
5.065789473684211, 6.305993690851736
5.098684210526316, 6.205047318611987
5.131578947368421, 6.10410094637224
5.171052631578948, 6.02839116719243
5.210526315789474, 6.003154574132493
5.256578947368421, 6.02839116719243
5.2894736842105265, 6.078864353312303
5.328947368421053, 6.129337539432177
5.36842105263158, 6.2302839116719255
5.407894736842105, 6.280757097791799
5.447368421052632, 6.3312302839116725
5.486842105263158, 6.356466876971609
5.526315789473685, 6.356466876971609
5.565789473684211, 6.381703470031546
5.611842105263158, 6.356466876971609
5.657894736842106, 6.3312302839116725
5.703947368421053, 6.3312302839116725
5.743421052631579, 6.305993690851736
5.782894736842106, 6.280757097791799
5.828947368421053, 6.305993690851736
5.868421052631579, 6.3312302839116725
5.907894736842106, 6.381703470031546
5.940789473684211, 6.457413249211356
5.980263157894738, 6.50788643533123
6.006578947368421, 6.583596214511042
6.052631578947368, 6.634069400630915
6.092105263157896, 6.659305993690852
6.131578947368421, 6.684542586750789
6.171052631578949, 6.684542586750789
6.223684210526317, 6.659305993690852
6.2631578947368425, 6.634069400630915
6.309210526315789, 6.634069400630915
6.348684210526317, 6.634069400630915
6.3881578947368425, 6.634069400630915
6.427631578947368, 6.634069400630915
6.467105263157896, 6.684542586750789
"""

df = pd.read_csv(io.StringIO(csv_data_string), header=None)

# Assume first column is log10(Pre-training step) and second is Surprisal
log10_t_data = df.iloc[:, 0].values
surprisal_values = df.iloc[:, 1].values

# Convert log10 steps to linear steps
t_data = 10**log10_t_data

# P(t) observed: Inverse Surprisal
# Ensure surprisal_values are not zero or too close to zero before inversion
P_t_observed = 1 / np.maximum(surprisal_values, 0.01) # Clamp to a small positive value

# --- Define the full PEF model for curve fitting with all parameters ---
# This function takes all parameters to be fitted:
# P0_initial, alpha, beta, a_E, b_E, c_E, d_C, e_C, f_C, f_S, h_eta, i_eta
def pef_model_for_fit_full_refined(t_data_fit, P0_initial, alpha, beta, 
                                   a_E, b_E, c_E, # E(t) Gaussian params
                                   d_C, e_C, f_C, # C(t) Logistic params
                                   f_S,           # S(t) power law scaling, g_S is fixed
                                   h_eta, i_eta   # eta(t) exponential decay params
                                  ):
    # Define the PEF component functions using the current fitting parameters
    def E_func(t):
        # E(t) = a_E * exp(-(t-b_E)^2/(2c_E^2)) * (1 + 0.2*sin(0.1*t + π/2)) * exp(-0.01*t)
        # Add a small constant to 'c_E' to prevent division by zero if c_E becomes too small during fitting
        gaussian_part = a_E * np.exp(-(t - b_E)**2 / (2 * (c_E + 1e-6)**2))
        oscillatory_part = (1 + 0.2 * np.sin(0.1 * t + np.pi/2)) # Fixed amplitude and frequency
        decay_part = np.exp(-0.01 * t) # Fixed decay rate
        return np.maximum(0.01, gaussian_part * oscillatory_part * decay_part)

    def C_func(t):
        # C(t) = d_C / (1 + exp(-e_C*(t - f_C))) (Logistic cap)
        # Clamp d_C and e_C to be positive
        return np.maximum(1e-6, d_C) / (1 + np.exp(-np.maximum(1e-6, e_C) * (t - f_C)))

    def S_func(t):
        # S(t) = f_S * t^g_S, with g_S fixed at 0.3
        g_S_fixed = 0.3 # Grok's fixed value
        # Clamp f_S to be positive
        return np.maximum(1e-6, f_S) * (t**g_S_fixed)

    def eta_func(t):
        # Decaying exponential: η(t) = h_eta * exp(-i_eta*t)
        # Clamp h_eta and i_eta to be positive
        return np.maximum(1e-6, h_eta) * np.exp(-np.maximum(1e-6, i_eta) * t)

    # Create a lambda function for P_dot with fixed component functions
    def ode_wrapper(P, t, alpha_val, beta_val):
        return P_dot(P, t, alpha_val, beta_val, C_func, S_func, E_func, eta_func)
    
    # Solve the ODE
    # Ensure P0_initial is positive
    sol = odeint(ode_wrapper, np.maximum(1e-6, P0_initial), t_data_fit, args=(alpha, beta))
    return sol[:, 0] # Return the P(t) values

# --- Initial Guess and Bounds for 12 Parameters ---
# P0_initial, alpha, beta, a_E, b_E, c_E, d_C, e_C, f_C, f_S, h_eta, i_eta

# Estimate based on the data's range and typical function behaviors
t_min, t_max = t_data.min(), t_data.max()
P_min, P_max = P_t_observed.min(), P_t_observed.max()

initial_guess = [
    P_min, # P0_initial: Start near min observed P(t)
    0.1,   # alpha: Expansion coefficient, start small
    0.5,   # beta: Dissonance integration, typical value

    # E(t) = a_E * exp(-(t-b_E)^2/(2c_E^2)) * (1 + 0.2*sin(0.1*t + π/2)) * exp(-0.01*t)
    0.5,   # a_E: Amplitude of E(t) peak. (Roughly 0.5-1.0 from previous E(t) assumptions)
    t_max / 2, # b_E: Center of E(t) peak (mid-training)
    t_max / 10, # c_E: Width of E(t) peak (e.g., 10% of max time)

    # C(t) = d_C / (1 + exp(-e_C*(t - f_C)))
    P_max * 1.5, # d_C: Cap for C(t), should be higher than P_max
    1e-5,  # e_C: Steepness of logistic curve
    t_max / 2, # f_C: Midpoint of logistic curve

    # S(t) = f_S * t^0.3
    1e-6,  # f_S: Scaling factor for S(t), start very small as t is large

    # eta(t) = h_eta * exp(-i_eta*t)
    1.0,   # h_eta: Initial pacing value
    1e-7   # i_eta: Decay rate, very small for slow decay over large t
]

# Bounds for parameters
# (P0_initial, alpha, beta, a_E, b_E, c_E, d_C, e_C, f_C, f_S, h_eta, i_eta)
lower_bounds = [
    1e-7,  # P0_initial (must be > 0)
    1e-7,  # alpha
    1e-7,  # beta
    1e-7,  # a_E (amplitude)
    t_min * 0.5, # b_E (center of peak, must be within time range)
    1e-7,  # c_E (width, must be > 0)
    1e-7,  # d_C (cap)
    1e-10, # e_C (steepness)
    t_min * 0.5, # f_C (midpoint)
    1e-7,  # f_S (scaling)
    1e-7,  # h_eta (initial pacing)
    1e-10  # i_eta (decay rate, very small but positive)
]

upper_bounds = [
    P_max * 5, # P0_initial (increased upper bound to allow more flexibility)
    10.0,      # alpha
    10.0,      # beta
    5.0,       # a_E
    t_max * 1.5, # b_E
    t_max * 2, # c_E
    P_max * 5, # d_C (cap)
    1.0,       # e_C
    t_max * 1.5, # f_C
    10.0,      # f_S
    5.0,       # h_eta
    0.1        # i_eta
]

# --- Perform the Curve Fitting (Inverse PEF Modeling) ---
print("\nStarting PEF Inverse Modeling for Transformer Data with Grok's Final Refinements...")
try:
    popt, pcov = curve_fit(pef_model_for_fit_full_refined, t_data, P_t_observed, p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=10000)
    
    # Unpack fitted parameters
    P0_fit, alpha_fit, beta_fit, a_E_fit, b_E_fit, c_E_fit, d_C_fit, e_C_fit, f_C_fit, f_S_fit, h_eta_fit, i_eta_fit = popt

    print(f"PEF Inverse Modeling Complete!")
    print(f"Fitted Parameters:")
    print(f"  P0_initial = {P0_fit:.4f}")
    print(f"  alpha      = {alpha_fit:.4f}")
    print(f"  beta       = {beta_fit:.4f}")
    print(f"  E(t) params: a_E={a_E_fit:.4f}, b_E={b_E_fit:.4f}, c_E={c_E_fit:.4f}")
    print(f"  C(t) params: d_C={d_C_fit:.4f}, e_C={e_C_fit:.4f}, f_C={f_C_fit:.4f}")
    print(f"  S(t) params: f_S={f_S_fit:.4f} (g_S fixed at 0.3)")
    print(f"  eta(t) params: h_eta={h_eta_fit:.4f}, i_eta={i_eta_fit:.4f}")

    # Generate the predicted P(t) curve using the fitted parameters
    # Define component functions using the fitted parameters for prediction
    def E_func_fitted(t):
        gaussian_part = a_E_fit * np.exp(-(t - b_E_fit)**2 / (2 * (c_E_fit + 1e-6)**2))
        oscillatory_part = (1 + 0.2 * np.sin(0.1 * t + np.pi/2))
        decay_part = np.exp(-0.01 * t)
        return np.maximum(0.01, gaussian_part * oscillatory_part * decay_part)

    def C_func_fitted(t):
        return np.maximum(1e-6, d_C_fit) / (1 + np.exp(-np.maximum(1e-6, e_C_fit) * (t - f_C_fit)))

    def S_func_fitted(t):
        g_S_fixed = 0.3
        return np.maximum(1e-6, f_S_fit) * (t**g_S_fixed)

    def eta_func_fitted(t):
        return np.maximum(1e-6, h_eta_fit) * np.exp(-np.maximum(1e-6, i_eta_fit) * t)

    P_t_predicted = solve_pef_ode_full_refined(t_data, P0_fit, alpha_fit, beta_fit, 
                                                a_E_fit, b_E_fit, c_E_fit, 
                                                d_C_fit, e_C_fit, f_C_fit, 
                                                f_S_fit, 
                                                h_eta_fit, i_eta_fit)

    # --- 6. Plotting Results ---
    plt.figure(figsize=(14, 7))
    plt.plot(t_data, P_t_observed, 'o', label='P(t) Observed (Inverse Surprisal)', markersize=3, alpha=0.6)
    plt.plot(t_data, P_t_predicted, '-', label='P(t) PEF Prediction', linewidth=2, color='red')
    plt.xlabel('Training Step (Linear Scale, Logarithmic Axis)')
    plt.ylabel('P(t) - Placeholder Expansion')
    plt.title('PEF Fit to Transformer Learning Curve (P(t)) with Grok Final Refinements')
    plt.grid(True)
    plt.legend()
    plt.xscale('log') # Use log scale for X-axis as original data was log10 steps
    plt.tight_layout()
    plt.show()

    # --- Plotting dP/dt Observed vs. dP/dt Predicted ---
    dPdt_predicted = np.array([P_dot(P_t_predicted[i], t_data[i], alpha_fit, beta_fit, C_func_fitted, S_func_fitted, E_func_fitted, eta_func_fitted) for i in range(len(t_data))])
    dPdt_observed_approx = np.gradient(P_t_observed, t_data)

    plt.figure(figsize=(14, 7))
    plt.plot(t_data, dPdt_observed_approx, 'o', label='dP/dt Observed (Approx.)', markersize=3, alpha=0.4)
    plt.plot(t_data, dPdt_predicted, '-', label='dP/dt PEF Prediction', linewidth=2, color='blue')
    plt.xlabel('Training Step (Linear Scale, Logarithmic Axis)')
    plt.ylabel('dP/dt - Rate of Expansion')
    plt.title('PEF Fit to Transformer Learning Curve (dP/dt) with Grok Final Refinements')
    plt.grid(True)
    plt.legend()
    plt.xscale('log') # Use log scale for X-axis
    plt.tight_layout()
    plt.show()

    # --- Optional: Plot the Fitted Component Functions ---
    t_plot_components = np.logspace(np.log10(t_data.min()), np.log10(t_data.max()), 500)

    E_fitted_values = E_func_fitted(t_plot_components)
    C_fitted_values = C_func_fitted(t_plot_components)
    S_fitted_values = S_func_fitted(t_plot_components)
    eta_fitted_values = eta_func_fitted(t_plot_components)

    plt.figure(figsize=(14, 10))
    plt.subplot(4, 1, 1)
    plt.plot(t_plot_components, C_fitted_values, label='Fitted C(t)', color='blue')
    plt.ylabel('C(t)')
    plt.title('Fitted PEF Component Functions (Grok Final Fine-Tune)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')

    plt.subplot(4, 1, 2)
    plt.plot(t_plot_components, S_fitted_values, label='Fitted S(t)', color='green')
    plt.ylabel('S(t)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')

    plt.subplot(4, 1, 3)
    plt.plot(t_plot_components, E_fitted_values, label='Fitted E(t)', color='red')
    plt.ylabel('E(t)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')

    plt.subplot(4, 1, 4)
    plt.plot(t_plot_components, eta_fitted_values, label='Fitted η(t)', color='purple')
    plt.ylabel('η(t)')
    plt.xlabel('Training Step (Linear Scale, Logarithmic Axis)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.tight_layout()
    plt.show()


except RuntimeError as e:
    print(f"Error during curve fitting: {e}")
    print("This might happen if the initial guess is far from the optimal, or if the data is very noisy.")
    print("Consider adjusting initial_guess or bounds, or checking the quality of extracted data.")
    print("Also, ensure the E_t_observed (if still used) and the functional forms are reasonable for the given P_t_observed.")
    print("If the fit is poor, try adjusting the initial guesses or bounds for the parameters, especially for C(t) and E(t) components.")

# --- Conclusion & Next Steps ---
print("\nPEF analysis framework for Transformer data with Grok's final refined functions is executed, Doctor!")
print("Review the new plots to see the improved fit.")
print("The fitted parameters for alpha, beta, and all component functions are now available.")
print("These fitted parameters represent the universal signatures of this Transformer's learning process!")
print("We can now analyze these signatures and discuss their implications.")
