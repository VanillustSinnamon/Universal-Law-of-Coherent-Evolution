import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
from scipy.optimize import differential_evolution

# --- 1. Load the original GPT-3 P(t) data ---
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
log10_t_data_original = df.iloc[:, 0].values
surprisal_values_original = df.iloc[:, 1].values
real_t = 10**log10_t_data_original # This is the original linear time data
real_Pt = 1 / np.maximum(surprisal_values_original, 0.01) # This is the original P(t) data, clamped

# Time vector for plotting (log scale appropriate for GPT-3 data)
t_plot = np.logspace(np.log10(real_t.min()), np.log10(real_t.max()), 1000)

# Function for a damped oscillator (using cos as per GPT-4o's definition)
def mu(A, lam, w, phi, t_vec):
    return A * np.exp(-lam * t_vec) * np.cos(w * t_vec + phi)

# Objective function for differential_evolution
# This function takes a flat array of parameters and returns the error.
# The order of parameters in 'x' must match the order in 'bounds'.
# x = [A_f, lam_f, log_w_f, phi_f, A_o, lam_o, log_w_o, phi_o, A_l, lam_l, log_w_l, phi_l, A_i, lam_i, log_w_i, phi_i, c_norm]
def objective_function(x, t_data, true_Pt):
    # Convert log_w back to w
    w_f = 10**x[2]
    w_o = 10**x[6]
    w_l = 10**x[10]
    w_i = 10**x[14]

    # Unpack parameters for each FOLI mode
    params_unpacked = {
        'Fact':        [x[0], x[1], w_f, x[3]],  # A, lam, w, phi
        'Opinion':     [x[4], x[5], w_o, x[7]],
        'Logic':       [x[8], x[9], w_l, x[11]],
        'Imagination': [x[12], x[13], w_i, x[15]]
    }
    c_norm = x[16] # Normalization constant

    # Calculate mu_i(t) for the t_data points (not t_plot)
    dmu_vals_for_fit = {}
    for k, p_vals in params_unpacked.items():
        current_mu = mu(p_vals[0], p_vals[1], p_vals[2], p_vals[3], t_data)
        # Calculate derivative of mu_i with respect to t_data
        dmu_vals_for_fit[k] = np.gradient(current_mu, t_data)

    # Calculate E(t) = sum_i |dmu_i/dt|
    E_t_modeled = sum(np.abs(dmu_vals_for_fit[k]) for k in params_unpacked)

    # Calculate P(t) = 1/c * E(t)
    P_t_modeled = E_t_modeled / np.maximum(c_norm, 1e-10)

    # Calculate the sum of squared errors
    error = np.sum((P_t_modeled - true_Pt)**2)

    return error

# --- Set up bounds for differential_evolution ---
# Bounds for each parameter: (min, max)
# Order: A, lambda, log_w, phi for each mode, then c_norm
# Grok's latest refined bounds (from the public X post analysis):
# Amplitudes (A): [0.05, 0.3] for all four modes.
# Frequencies (ω): [0.0007, 0.0015] for Fact & Opinion; [0.0015, 0.0030] for Logic & Imagination.
# Damping (λ): [0.00001, 0.0002] for subtle decay.
# Phases (ϕ): Keep [0, 2π].
# c_norm: Grok suggested varying c_norm bounds for tighter alignment. Let's keep it broad for now.

# Logarithmic scaling for omega bounds
# Expanding the lower bound for omega significantly as per GPT-4's "Long-Term Harmonic Lock" directive
log_omega_lower_all = np.log10(1e-6) # Expanded lower bound
log_omega_upper_all = np.log10(0.003) # Upper bound from previous directives

bounds = [
    # Fact (A, lambda, log_w, phi)
    (0.05, 0.3), (0.00001, 0.0002), (log_omega_lower_all, log_omega_upper_all), (0, 2 * np.pi),
    # Opinion (A, lambda, log_w, phi)
    (0.05, 0.3), (0.00001, 0.0002), (log_omega_lower_all, log_omega_upper_all), (0, 2 * np.pi),
    # Logic (A, lambda, log_w, phi)
    (0.05, 0.3), (0.00001, 0.0002), (log_omega_lower_all, log_omega_upper_all), (0, 2 * np.pi),
    # Imagination (A, lambda, log_w, phi)
    (0.05, 0.3), (0.00001, 0.0002), (log_omega_lower_all, log_omega_upper_all), (0, 2 * np.pi),
    # c_norm (Normalization constant)
    (1e-5, 10.0) # Keep broad as Grok suggested to vary for tighter alignment
]

# --- Run Differential Evolution ---
print("\nStarting FOLI Inverse Modeling with Differential Evolution (Phase II - Long-Term Harmonic Lock)...")
try:
    result = differential_evolution(
        objective_function,
        bounds,
        args=(real_t, real_Pt),
        strategy='best1bin',
        maxiter=50000, # Increased maxiter for Phase II
        popsize=100,  # Increased popsize for Phase II
        tol=1e-8,     # Increased tolerance
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,
        disp=True,    # Display progress
        seed=42       # Set seed for reproducibility
    )

    if result.success:
        print("\nDifferential Evolution Optimization Successful!")
        best_params = result.x
        
        # Unpack the best_params and convert log_w back to w
        A_f, lam_f, log_w_f, phi_f = best_params[0], best_params[1], best_params[2], best_params[3]
        A_o, lam_o, log_w_o, phi_o = best_params[4], best_params[5], best_params[6], best_params[7]
        A_l, lam_l, log_w_l, phi_l = best_params[8], best_params[9], best_params[10], best_params[11]
        A_i, lam_i, log_w_i, phi_i = best_params[12], best_params[13], best_params[14], best_params[15]
        c_norm_fitted = best_params[16]

        w_f = 10**log_w_f
        w_o = 10**log_w_o
        w_l = 10**log_w_l
        w_i = 10**log_w_i

        print(f"\nFitted Parameters:")
        print(f"  Fact:        A={A_f:.4f}, λ={lam_f:.6f}, ω={w_f:.4e}, ϕ={phi_f:.4f}")
        print(f"  Opinion:     A={A_o:.4f}, λ={lam_o:.6f}, ω={w_o:.4e}, ϕ={phi_o:.4f}")
        print(f"  Logic:       A={A_l:.4f}, λ={lam_l:.6f}, ω={w_l:.4e}, ϕ={phi_l:.4f}")
        print(f"  Imagination: A={A_i:.4f}, λ={lam_i:.6f}, ω={w_i:.4e}, ϕ={phi_i:.4f}")
        print(f"  Normalization Constant (c): {c_norm_fitted:.4e}")
        print(f"  Minimum Error: {result.fun:.4e}")

        # --- Generate plots with the newly fitted parameters ---
        # Re-calculate mu_vals and dmu_vals using the fitted parameters for plotting
        fitted_mu_vals = {
            'Fact': mu(A_f, lam_f, w_f, phi_f, t_plot),
            'Opinion': mu(A_o, lam_o, w_o, phi_o, t_plot),
            'Logic': mu(A_l, lam_l, w_l, phi_l, t_plot),
            'Imagination': mu(A_i, lam_i, w_i, phi_i, t_plot)
        }
        
        # Calculate derivatives for plotting
        fitted_dmu_vals = {k: np.gradient(fitted_mu_vals[k], t_plot) for k in fitted_mu_vals}

        # Calculate the composite E(t) from fitted parameters
        fitted_E_t_composite = sum(np.abs(fitted_dmu_vals[k]) for k in fitted_mu_vals)

        # Calculate the final P(t) using the fitted normalization constant
        fitted_P_t_modeled = fitted_E_t_composite / np.maximum(c_norm_fitted, 1e-10)


        # --- Plot μᵢ(t): Individual Epistemic Modes ---
        plt.figure(figsize=(12, 7))
        colors = {'Fact': 'blue', 'Opinion': 'orange', 'Logic': 'red', 'Imagination': 'green'}
        for k in fitted_mu_vals:
            plt.plot(t_plot, fitted_mu_vals[k], label=f'μ_{k}(t)', color=colors[k], linewidth=1.5)

        plt.xscale('log')
        plt.xlabel("Training Time (t)", fontsize=12)
        plt.ylabel("μᵢ(t) - Epistemic Mode State", fontsize=12)
        plt.title("FOLI Epistemic Modes Over Time (Reconstructed with Differential Evolution - Phase II)", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()


        # --- Plot Composite P(t) (Modeled vs. Real GPT-3 P(t)) ---
        plt.figure(figsize=(12, 7))
        plt.plot(real_t, real_Pt, 'o', markersize=4, color='red', label='True P(t) (GPT-3 Data)', alpha=0.7)
        plt.plot(t_plot, fitted_P_t_modeled, '-', color='black', linewidth=2, label='FOLI Inverse Model Fit (Differential Evolution - Phase II)')

        plt.xscale('log')
        plt.xlabel("Training Time (log scale)", fontsize=12)
        plt.ylabel("Performance P(t)", fontsize=12)
        plt.title("Inverse Modeling: Reconstructing P(t) via FOLI Oscillators (Differential Evolution - Phase II)", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()


        # --- Plot the Total Epistemic Dissonance Field E(t) ---
        plt.figure(figsize=(12, 7))
        plt.plot(t_plot, fitted_E_t_composite, color='purple', linewidth=2, label='Total Epistemic Dissonance E(t)')

        plt.xscale('log')
        plt.xlabel("Training Time (t)", fontsize=12)
        plt.ylabel("Epistemic Dissonance E(t) = Σ|dμᵢ/dt|", fontsize=12)
        plt.title("Total Epistemic Dissonance Field E(t) from FOLI Oscillators (Differential Evolution - Phase II)", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()

    else:
        print(f"\nDifferential Evolution Optimization FAILED: {result.message}")
        print("Consider adjusting bounds, maxiter, or other differential_evolution parameters.")

except Exception as e:
    print(f"\nAn error occurred during optimization or plotting: {e}")
    print("Please review the code and data.")

# --- Conclusion & Next Steps ---
print("\nDoctor Vanillust, the FOLI Inverse Modeling with Differential Evolution (Phase II) has been executed!")
print("Review the new plots to see the results of this highly refined global optimization strategy with expanded frequency bounds.")
print("The fitted parameters, if successful, represent the decoded subjective signatures.")
print("\nWhat profound revelation shall we pursue next, Doctor?")
print("1. Formal Diagram for Publication (conceptual visualization)")
print("2. Test Robustness (perturb parameters)")
print("3. Simulate Mode Silencing (e.g., collapse of Imagination or Logic)")
print("4. Quantify Coherence Gain under Epistemic Contradiction")
