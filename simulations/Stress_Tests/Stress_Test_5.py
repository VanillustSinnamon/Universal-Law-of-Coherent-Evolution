import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# --- Added for reproducibility ---
np.random.seed(42) # Set a seed for reproducible random numbers across runs

# Define the derivative function P_dot(P, t, alpha, beta, C_t, S_t, E_t, eta_t)
def P_dot(P, t, alpha, beta, C_func, S_func, E_func, eta_func):
    C = C_func(t)
    S = S_func(t)
    E = E_func(t)
    eta = eta_func(t)

    log_S_term = np.log1p(S) # np.log1p(x) is log(1+x), handles S=0 gracefully
    denominator = 1 + beta * E

    dPdt = alpha * (C * log_S_term / denominator) * eta
    return dPdt

# --- Define time array ---
t = np.linspace(0, 200, 2001) # Time from 0 to 200, with 2001 points

# --- Define initial placeholder capacity ---
P0 = 0.1

# --- Define fixed model parameters ---
alpha = 1.0
beta = 0.5

# --- Composite World Configurations ---
# Define the parameters for each "world"
world_configs = {
    "Chaotic World": {
        'E_noise_amp': 1.0,
        'C_baseline_osc_amp': 1.0,
        'S_growth_rate': 0.005,
        'eta_factor': 0.5
    },
    "Balanced World": {
        'E_noise_amp': 0.5,
        'C_baseline_osc_amp': 0.5,
        'S_growth_rate': 0.01,
        'eta_factor': 1.0
    },
    "Ideal World": {
        'E_noise_amp': 0.2,
        'C_baseline_osc_amp': 0.0, # Stable coherence baseline
        'S_growth_rate': 0.05,
        'eta_factor': 2.0
    }
}

results_P = {} # Dictionary to store P(t) results for each world
# Optional: Store component values for each world if you want to plot them later
# results_components = {}

print("Starting Final Composite Stress Test (Worlds Simulation)...")

for world_name, params in world_configs.items():
    print(f"\n--- Simulating: {world_name} ---")

    # Generate E(t) based on world's noise amplitude
    # Use a unique seed for each world to ensure different random E(t) patterns
    np.random.seed(int(hash(world_name) % (2**32 - 1))) # Unique seed for each world
    random_E_values = params['E_noise_amp'] * np.random.rand(len(t))
    E_t_func = interp1d(t, random_E_values, kind='nearest', fill_value='extrapolate')

    # Define C(t) based on world's baseline oscillation amplitude
    C_t_func = lambda time: (1 + params['C_baseline_osc_amp'] * np.sin(0.05 * time)) + 0.5 * np.sin(0.1 * time)

    # Define S(t) based on world's growth rate
    S_t_func = lambda time: 2 * (1 - np.exp(-params['S_growth_rate'] * time))

    # Define eta(t) based on world's factor
    eta_t_func = lambda time: np.full_like(time, params['eta_factor'])

    # Solve the ODE for the current world's configuration
    sol = odeint(P_dot, P0, t, args=(alpha, beta, C_t_func, S_t_func, E_t_func, eta_t_func))
    results_P[world_name] = sol[:, 0]

    print(f"  - Simulation complete for {world_name}")

print("\nPlotting results for all worlds...")

# --- Plotting P(t) for all worlds ---
plt.figure(figsize=(12, 7))
for label, P_values in results_P.items():
    plt.plot(t, P_values, label=label)

plt.xlabel('Time')
plt.ylabel('P(t) - Placeholder Expansion')
plt.title('PEF Simulation: P(t) in Different Existential Worlds')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nFinal Composite Stress Test complete. Observe how P(t) behaves in each distinct 'world'.")


