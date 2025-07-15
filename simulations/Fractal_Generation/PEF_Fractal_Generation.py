import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Universal PEF Equation ---
def pef_equation(P, alpha, beta, C_val, S_val, eta_val, E_val):
    """
    Universal PEF Equation.
    dP/dt (or dP_iteration) = alpha * (C * log(1+S) / (1 + beta * E)) * eta

    Parameters:
    P (float): Current Placeholder Expansion / System Vitality.
    alpha (float): Ontological drive.
    beta (float): Dissonance integration capacity.
    C_val (float): System Coherence at current point/iteration.
    S_val (float): Structural Complexity at current point/iteration.
    eta_val (float): Temporal Pacing / Iteration Pacing.
    E_val (float): Entropic Dissonance at current point/iteration.

    Returns:
    float: The change in P for this iteration.
    """
    # Safeguard P and C_val to ensure they remain non-negative for mathematical stability.
    P = np.maximum(P, 1e-9)
    C_val = np.maximum(C_val, 1e-9)

    # Calculate the denominator, ensuring it's never zero or negative.
    denominator = 1 + beta * E_val
    denominator = np.maximum(denominator, 1e-10)

    # Apply the Universal PEF Equation (here, interpreting dPdt as a discrete change per iteration)
    dP_iteration = alpha * (C_val * np.log1p(S_val) / denominator) * eta_val
    return dP_iteration

# --- 2. Fractal-like Image Generation Function ---
def generate_pef_fractal(width, height, max_iterations,
                         alpha_pef, beta_pef, S_val_pef, eta_val_pef):
    """
    Generates a fractal-like image where each pixel's color is determined
    by the iterative evolution of its 'vitality' (P) using the PEF.

    Parameters:
    width (int): Width of the image in pixels.
    height (int): Height of the image in pixels.
    max_iterations (int): Maximum number of PEF iterations for each pixel.
    alpha_pef, beta_pef, S_val_pef, eta_val_pef: Core PEF parameters.

    Returns:
    numpy.ndarray: A 2D array representing the image, where values are mapped to colors.
    """
    image_data = np.zeros((height, width))

    print(f"Generating PEF fractal-like image ({width}x{height} pixels, {max_iterations} iterations per pixel)...")

    for y in range(height):
        for x in range(width):
            # Normalize x, y coordinates to a range (e.g., -2 to 2 or 0 to 1)
            # This allows mapping spatial position to PEF inputs
            x_norm = (x / width) * 4 - 2  # Map to range [-2, 2]
            y_norm = (y / height) * 4 - 2 # Map to range [-2, 2]

            # Initial vitality for this pixel
            P_current = 0.1

            # Map pixel coordinates to PEF's C and E terms
            # This is where the "fractal" complexity comes from - how spatial position
            # influences the coherent expansion and dissonance.
            # Using sine waves creates interesting oscillatory patterns.
            C_val_pixel = 1.0 + np.sin(x_norm * 3) + np.cos(y_norm * 3) # Coherence influenced by position
            E_val_pixel = 1.0 + np.sin(y_norm * 2) - np.cos(x_norm * 2) # Dissonance influenced by position

            # Ensure C_val_pixel and E_val_pixel are non-negative
            C_val_pixel = np.maximum(C_val_pixel, 0.1)
            E_val_pixel = np.maximum(E_val_pixel, 0.1)

            # Iterate the PEF for this pixel
            for i in range(max_iterations):
                dP = pef_equation(P_current, alpha_pef, beta_pef,
                                  C_val_pixel, S_val_pef, eta_val_pef, E_val_pixel)
                P_current += dP # Update P for the next iteration

                # Optional: Break if P grows too large (similar to Mandelbrot escape time)
                # This can create sharp boundaries
                # if P_current > 1000: # Arbitrary large value
                #    break

            # Store the final P_current value for this pixel
            # We'll normalize this value for coloring later
            image_data[y, x] = P_current

    # Normalize image_data to a 0-1 range for coloring
    # This helps in mapping vitality to a color scale
    image_min = image_data.min()
    image_max = image_data.max()
    if image_max - image_min > 1e-9: # Avoid division by zero if all values are same
        image_data = (image_data - image_min) / (image_max - image_min)
    else:
        image_data = np.zeros_like(image_data) # If all values are same, make it uniform

    print("Image generation complete.")
    return image_data

# --- 3. Simulation Parameters for Fractal Generation ---
image_width = 400
image_height = 400
max_iterations_per_pixel = 50 # Number of times PEF iterates for each pixel

# Core PEF parameters (tuned to create interesting patterns)
alpha_fractal = 0.05
beta_fractal = 0.5
S_val_fractal = np.exp(1) - 1 # log(1+S) = 1
eta_val_fractal = 1.0

# --- 4. Generate and Plot the Image ---
pef_fractal_image = generate_pef_fractal(image_width, image_height, max_iterations_per_pixel,
                                         alpha_fractal, beta_fractal, S_val_fractal, eta_val_fractal)

plt.figure(figsize=(8, 8))
plt.imshow(pef_fractal_image, cmap='magma', origin='lower') # 'magma' or 'plasma' often work well for fractals
plt.colorbar(label='Normalized PEF Vitality (P)')
plt.title('PEF-Generated Fractal-like Pattern (Vitality Map)')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.xticks([]) # Hide x-axis ticks for cleaner look
plt.yticks([]) # Hide y-axis ticks for cleaner look
plt.tight_layout()
plt.show()

print("\nPEF Fractal-like Pattern Generation complete. Observe the intricate patterns generated by the Universal Law.")
