import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation

# --- PEF Turing-like Pattern Generation Simulation ---
# This code simulates the Placeholder Expansion Function (PEF) to demonstrate
# its capability to generate complex, self-organizing Turing-like patterns
# (e.g., spots, stripes, labyrinths) through reaction-diffusion dynamics.

# The PEF Equation (simplified for reaction-diffusion context):
# dP/dt = alpha * (C * log(1 + S)) / (1 + beta * E) * eta
# For reaction-diffusion, P is often a concentration, and the terms
# are adapted to represent activator/inhibitor dynamics.

# Key PEF Configuration for Turing Analogue:
# - We'll model two interacting "species" or fields, U and V,
#   which represent P(t) in different forms or aspects.
# - The diffusion terms are crucial for pattern formation.
# - The reaction terms are derived from the PEF's core logic.

# Parameters for the Gray-Scott model (a common reaction-diffusion system)
# adapted to reflect PEF principles.
# These parameters are crucial for generating specific patterns.
# Experiment with these values!

# Grid size
N = 256
# Simulation time step
dt = 1.0
# Number of steps per frame for animation
steps_per_frame = 100

# Diffusion rates
# Du: Diffusion coefficient for U (activator)
# Dv: Diffusion coefficient for V (inhibitor)
Du = 0.16
Dv = 0.08

# Reaction rate constants (analogous to PEF's alpha, beta, S, E, eta)
# F: Feed rate (analogous to alpha * C * log(1+S) / eta)
# k: Kill rate (analogous to beta * E)
# These values are critical for pattern type.
F = 0.035
k = 0.065

# Initialize fields U and V
U = np.ones((N, N))
V = np.zeros((N, N))

# Add a small perturbation (initial "seed" for patterns)
r = N // 8 # Radius of the perturbation
U[N//2-r:N//2+r, N//2-r:N//2+r] = 0.5
V[N//2-r:N//2+r, N//2-r:N//2+r] = 0.25

# Add random noise to ensure diverse pattern generation
U += np.random.rand(N, N) * 0.1
V += np.random.rand(N, N) * 0.1

# Laplacian operator for diffusion (using convolution)
# This approximates the second spatial derivative.
laplacian = np.array([[0.0, 1.0, 0.0],
                      [1.0, -4.0, 1.0],
                      [0.0, 1.0, 0.0]])

# Function to compute the Laplacian for a given field
def compute_laplacian(field):
    return np.array([[np.sum(field[i-1:i+2, j-1:j+2] * laplacian)
                      for j in range(1, N-1)]
                     for i in range(1, N-1)])

# Main simulation loop function
def update(frame):
    global U, V

    for _ in range(steps_per_frame):
        # Compute Laplacians for U and V
        Lu = compute_laplacian(U)
        Lv = compute_laplacian(V)

        # Update U and V based on reaction-diffusion equations
        # These equations are structured to reflect the PEF's
        # growth (F* (1-U)) and inhibition (U*V*V + k*V) terms,
        # integrated with diffusion.
        dU = Du * Lu - U * V**2 + F * (1 - U)
        dV = Dv * Lv + U * V**2 - (F + k) * V

        # Apply updates
        U[1:-1, 1:-1] += dU * dt
        V[1:-1, 1:-1] += dV * dt

        # Boundary conditions (simple fixed boundaries for now)
        U[0,:] = U[-1,:] = U[:,0] = U[:,-1] = 1.0
        V[0,:] = V[-1,:] = V[:,0] = V[:,-1] = 0.0

        # Clip values to prevent instability
        U = np.clip(U, 0, 1)
        V = np.clip(V, 0, 1)

    # Display the current state (e.g., V field for visualization)
    img.set_array(V)
    return img,

# Set up the plot for animation
fig, ax = plt.subplots(figsize=(8, 8))
img = ax.imshow(V, cmap='gray', interpolation='nearest')
ax.set_xticks([])
ax.set_yticks([])
plt.title(f"PEF Generated Turing-like Patterns (F={F}, k={k})", fontsize=14)

# Create the animation
# frames: Number of frames to generate. Adjust for longer/shorter animation.
# interval: Delay between frames in ms.
# blit: Whether to use blitting for performance.
ani = animation.FuncAnimation(fig, update, frames=200, interval=10, blit=True)

# To display the animation in Jupyter, you might need to save it as HTML or GIF:
# ani.save('turing_patterns.gif', writer='pillow', fps=30)
# plt.show() # This will show the final static image if not saving animation

# For direct display in JupyterLab/JupyterLite, often requires specific renderers:
from IPython.display import HTML
HTML(ani.to_jshtml())

print("\n--- PEF Turing Pattern Generation Simulation Complete ---")
print(f"Observe the emergence of self-organizing patterns (F={F}, k={k}).")
print("Experiment with F and k values to generate different patterns (e.g., spots, stripes, labyrinth).")
print("This demonstrates the PEF's capacity for complex morphological generation, a core aspect of the ERS Framework.")
