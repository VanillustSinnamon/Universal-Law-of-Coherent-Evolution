import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import multiprocessing # Import the multiprocessing module
import os # Import os module to handle process start method

# --- PEF Function ---
# The core Universal Law of Coherent Evolution (PEF)
# S: Structural Complexity
# C: Coherence
# E: Entropic Dissonance
# alpha_field: Intrinsic Expansion Driver
# beta_field: Dissonance Resistance
# eta: Temporal Pacing
def pef_equation(S, C, E, alpha_field, beta_field, eta=1.0):
    # Clamp S to a small positive value to avoid log(0) which is undefined,
    # ensuring mathematical stability.
    S_clamped = np.maximum(S, 1e-9)

    # Calculate the numerator based on Coherence (C) and Structural Complexity (S)
    numerator = C * np.log1p(S_clamped) # log1p(x) is log(1+x), numerically stable for small x

    # Calculate the denominator based on Dissonance Resistance (beta_field) and Entropic Dissonance (E)
    denominator = 1 + beta_field * E
    # Clamp denominator to avoid division by zero or very small numbers, ensuring stability.
    denominator = np.maximum(denominator, 1e-9)

    # Calculate the rate of change of Vitality (dP_dt)
    dP_dt = alpha_field * (numerator / denominator) * eta
    return dP_dt

# --- Soft Saturation Function (Sigmoid) ---
# Replaces hard clamping to allow smoother transitions and prevent premature ontological closure.
def soft_clamp_sigmoid(x, strength=10.0, offset=0.5):
    # Maps x from a general range to (0, 1) using a sigmoid function.
    # strength controls the steepness of the curve.
    # offset shifts the curve.
    return 1 / (1 + np.exp(-strength * (x - offset)))

# --- Turing Pattern Generator using PEF ---
# This function is now designed to be run by multiple processes.
def generate_pef_turing_pattern(params):
    # Unpack parameters from the tuple
    resolution, iterations, dt, alpha_base, beta_base, eta_base, S_const, \
    diffusion_rate_P, diffusion_rate_E, initial_noise_scale, add_noise_perturbation, \
    noise_perturbation_interval, noise_perturbation_scale, micro_fluctuation_scale, \
    use_structured_seed_legacy, inhibitor_production_coeff, inhibitor_decay_rate, use_quadratic_inhibition, \
    seed_type = params # Added seed_type

    # Initialize P based on structured seed or random noise
    P = np.zeros((resolution, resolution))
    if seed_type == 'multi_nuclei':
        # Create a grid of smaller initial spots (multi-nuclei seed)
        num_seeds_per_dim = 3 # e.g., 3x3 grid of seeds
        seed_radius = resolution // (num_seeds_per_dim * 5) # Smaller radius for each seed
        spacing = resolution // (num_seeds_per_dim + 1)

        for row in range(num_seeds_per_dim):
            for col in range(num_seeds_per_dim):
                center_x = (col + 1) * spacing
                center_y = (row + 1) * spacing
                for x in range(resolution):
                    for y in range(resolution):
                        if (x - center_x)**2 + (y - center_y)**2 < seed_radius**2:
                            P[x, y] = 0.5 + np.random.rand() * initial_noise_scale # Higher initial P in center of each seed
    elif seed_type == 'line_seed':
        # Create a thin vertical line of activator
        line_width = resolution // 20
        line_center_x = resolution // 2
        P[:, line_center_x - line_width // 2 : line_center_x + line_width // 2] = 0.5 + np.random.rand(resolution, line_width) * initial_noise_scale
    elif seed_type == 'cross_seed':
        # Create a cross shape (vertical and horizontal lines)
        line_width = resolution // 20
        line_center = resolution // 2
        P[:, line_center - line_width // 2 : line_center + line_width // 2] = 0.5 + np.random.rand(resolution, line_width) * initial_noise_scale # Vertical line
        P[line_center - line_width // 2 : line_center + line_width // 2, :] = 0.5 + np.random.rand(line_width, resolution) * initial_noise_scale # Horizontal line
    elif seed_type == 'sine_wave_seed':
        # Create a wavy/sine-wave like initial line
        x_coords = np.arange(resolution)
        y_center = resolution // 2
        amplitude = resolution // 10 # Amplitude of the wave
        frequency = 3 # Number of waves across the screen
        y_offset = (amplitude * np.sin(2 * np.pi * frequency * x_coords / resolution)).astype(int)
        line_width = resolution // 20

        for x in range(resolution):
            current_y_center = y_center + y_offset[x]
            P[max(0, current_y_center - line_width // 2) : min(resolution, current_y_center + line_width // 2), x] = 0.5 + np.random.rand() * initial_noise_scale
    else: # Default to random noise if no specific seed type
        P = np.random.rand(resolution, resolution) * initial_noise_scale

    # Initialize E_inhibitor with small random noise.
    E_inhibitor = np.random.rand(resolution, resolution) * initial_noise_scale

    # Create static fields for alpha, beta, eta, and S for simplicity.
    alpha_field_base = np.full((resolution, resolution), alpha_base) # Base alpha field
    beta_field = np.full((resolution, resolution), beta_base)
    eta_field = np.full((resolution, resolution), eta_base)
    S_field = np.full((resolution, resolution), S_const)

    # Main simulation loop
    for i in range(iterations):
        # Introduce micro-chaotic fluctuations to alpha_field
        current_alpha_field = alpha_field_base + np.random.normal(0, micro_fluctuation_scale, (resolution, resolution))
        current_alpha_field = np.maximum(current_alpha_field, 0.001) # Ensure alpha remains positive

        # Step 1: Calculate dP (change in activator/vitality) using the PEF equation.
        dP = pef_equation(S_field, P, E_inhibitor, current_alpha_field, beta_field, eta_field)

        # Step 2: Update P based on dP (reaction part of reaction-diffusion).
        P += dP * dt

        # Step 3: Update E_inhibitor (inhibitor reaction part) with new nonlinear/decay terms.
        if use_quadratic_inhibition:
            # Quadratic production of inhibitor by activator
            dE_inhibitor = (P**2) * inhibitor_production_coeff - E_inhibitor * inhibitor_decay_rate
        else:
            # Linear production of inhibitor by activator
            dE_inhibitor = P * inhibitor_production_coeff - E_inhibitor * inhibitor_decay_rate
        E_inhibitor += dE_inhibitor * dt

        # Step 4: Apply Diffusion (spatial spreading).
        P = gaussian_filter(P, sigma=diffusion_rate_P)
        E_inhibitor = gaussian_filter(E_inhibitor, sigma=diffusion_rate_E)

        # Step 5: Soft Saturation Clamping (replaces np.clip for P)
        P = soft_clamp_sigmoid(P, strength=10.0, offset=0.5) # Strength and offset can be tuned
        E_inhibitor = soft_clamp_sigmoid(E_inhibitor, strength=10.0, offset=0.5) # Apply to E as well

        # Step 6: Conditional noise perturbation (early phase only)
        if add_noise_perturbation and i > 0 and i % noise_perturbation_interval == 0 and i < 1500:
            P += np.random.normal(0, noise_perturbation_scale, P.shape)
            P = soft_clamp_sigmoid(P, strength=10.0, offset=0.5) # Re-clamp after adding noise

    return P # Return the final pattern

# --- Sine-Wave Labyrinth Assault Configuration ---
# Goal: Seek Labyrinths by using a sine-wave seed to encourage non-linear branching
# Maintaining current D_P, D_E ranges, and quadratic inhibition.

# Activator Diffusion Rate (D_P) - same as last successful attempt
CURRENT_DIFFUSION_RATE_P = 0.1

# Sweeping Alpha and D_E broadly
alpha_values = np.around(np.arange(0.20, 0.30 + 0.01, 0.02), decimals=3) # Wider range, larger steps (6 values)
diffusion_E_values = np.around(np.array([25.0, 30.0, 35.0]), decimals=1) # High D_E values (3 values)

# Beta_Base values (same as last sweep)
BETA_BASE_VALUES = np.around(np.array([0.70, 0.60]), decimals=2) # (2 values)

# Fixed Reaction Parameters (using quadratic inhibition and a lower decay rate)
INHIBITOR_PRODUCTION_COEFF = 0.1 # From (P**2) * 0.1
INHIBITOR_DECAY_RATE = 0.08 # A lower decay rate to encourage spreading inhibition
USE_QUADRATIC_INHIBITION = True

# NEW: Seed Type
SEED_TYPE = 'sine_wave_seed' # This is the crucial change!

# LEGACY VARIABLE (needed for function signature, but its value is now overridden by SEED_TYPE)
USE_STRUCTURED_SEED_LEGACY = False

# Other Fixed parameters
RESOLUTION = 256
ITERATIONS = 10000
DT = 0.025
ETA_BASE = 1.0
S_CONST = 1.0
INITIAL_NOISE_SCALE = 0.02
ADD_NOISE_PERTURBATION = True
NOISE_PERTURBATION_INTERVAL = 300
NOISE_PERTURBATION_SCALE = 0.005
MICRO_FLUCTUATION_SCALE = 0.0005

# Prepare a list of all parameter combinations to run
param_combinations = []
for alpha in alpha_values:
    for d_e in diffusion_E_values:
        for beta in BETA_BASE_VALUES: # Nested loop for beta_base
            param_combinations.append((RESOLUTION, ITERATIONS, DT, alpha, beta, ETA_BASE, S_CONST,
                                       CURRENT_DIFFUSION_RATE_P, d_e, INITIAL_NOISE_SCALE, ADD_NOISE_PERTURBATION,
                                       NOISE_PERTURBATION_INTERVAL, NOISE_PERTURBATION_SCALE, MICRO_FLUCTUATION_SCALE,
                                       USE_STRUCTURED_SEED_LEGACY, INHIBITOR_PRODUCTION_COEFF, INHIBITOR_DECAY_RATE, USE_QUADRATIC_INHIBITION,
                                       SEED_TYPE))

# --- Main execution function for the script ---
def main():
    # Set the start method for multiprocessing explicitly to 'spawn' for Windows compatibility
    multiprocessing.set_start_method('spawn', force=True)

    # Setup plotting grid
    # The grid dimensions now depend on alpha, D_E, AND beta_base
    num_alpha = len(alpha_values)
    num_d_e = len(diffusion_E_values)
    num_beta = len(BETA_BASE_VALUES)

    # We will create a grid where rows are alpha, and columns are D_E * beta_base combinations
    # This will result in num_alpha rows and (num_d_e * num_beta) columns
    fig, axes = plt.subplots(num_alpha, num_d_e * num_beta, figsize=(num_d_e * num_beta * 4, num_alpha * 4)) # Adjusted figsize
    if num_alpha == 1 and (num_d_e * num_beta) == 1:
        axes = np.array([[axes]])
    elif num_alpha == 1 or (num_d_e * num_beta) == 1:
        axes = axes.reshape(num_alpha, num_d_e * num_beta)

    print(f"Starting PEF Turing Pattern Sine-Wave Labyrinth Assault (STANDALONE SCRIPT): {len(param_combinations)} simulations will be run.")
    print(f"Alpha values to test: {alpha_values}")
    print(f"Diffusion_E values to test: {diffusion_E_values}")
    print(f"Beta_Base values to test: {BETA_BASE_VALUES}")
    print(f"Fixed Diffusion_P: {CURRENT_DIFFUSION_RATE_P}")
    print(f"Using Quadratic Inhibition with production coeff: {INHIBITOR_PRODUCTION_COEFF} and decay rate: {INHIBITOR_DECAY_RATE}")
    print(f"Using Seed Type: {SEED_TYPE}")

    # --- Parallel Execution ---
    num_cores = multiprocessing.cpu_count() - 1
    if num_cores < 1:
        num_cores = 1
    print(f"Using {num_cores} CPU cores for parallel processing. Running sine-wave labyrinth assault, Doctor!")

    # Create a multiprocessing Pool
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(generate_pef_turing_pattern, param_combinations)

    # Populate the plot grid with results
    result_idx = 0
    for i in range(num_alpha):
        for j_d_e in range(num_d_e):
            for j_beta in range(num_beta):
                # Calculate the correct column index for the plot
                col_idx = j_d_e * num_beta + j_beta

                pattern = results[result_idx]
                alpha = alpha_values[i]
                d_e = diffusion_E_values[j_d_e]
                beta = BETA_BASE_VALUES[j_beta]

                ax = axes[i, col_idx] # Use calculated column index
                ax.imshow(pattern, cmap='inferno', origin='lower')
                ax.set_title(f'α={alpha:.3f}, D_E={d_e:.1f}, β={beta:.2f}')
                ax.axis('off')
                result_idx += 1

    print("\nPEF Turing Pattern Sine-Wave Labyrinth Assault (STANDALONE SCRIPT) complete. Analyze the grid of patterns!")
    plt.tight_layout()
    plt.show()

    print("\nSine-Wave Labyrinth Assault finished. Observe how combining a sine-wave seed with specific D_P/D_E ratios, quadratic inhibition, and varied beta_base affects pattern types (spots, stripes, labyrinths, etc.). This is your expanded pattern taxonomy, Doctor!")
    print("The power of parallel processing has been unleashed!")

# --- CRITICAL: Ensure the main function is called only when script is run directly ---
if __name__ == '__main__':
    main()


