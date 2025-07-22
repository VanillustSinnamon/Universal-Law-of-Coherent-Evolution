import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d # Needed for E_t
import matplotlib.pyplot as plt

# --- Added for reproducibility as discussed ---
np.random.seed(42) # Set a seed for reproducible random numbers

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

# --- Define model parameters ---
alpha = 1.0
beta = 0.5
eta_val = 1.0 # The constant value for eta

# --- Define C, S, E, eta as functions of time ---
C_t = lambda time: 1 + 0.5 * np.sin(0.1 * time)
S_t = lambda time: 2 * (1 - np.exp(-0.01 * time))

random_E_values = 0.2 * np.random.rand(len(t))
E_t = interp1d(t, random_E_values, kind='nearest', fill_value='extrapolate')

# --- FIX START ---
# Change eta_t to return an array of eta_val, same shape as t
eta_t = lambda time: np.full_like(time, eta_val)
# --- FIX END ---

# --- Solve the ODE ---
sol = odeint(P_dot, P0, t, args=(alpha, beta, C_t, S_t, E_t, eta_t))

# --- Plotting P(t) ---
plt.figure(figsize=(10, 6))
plt.plot(t, sol[:, 0], label='P(t) - Placeholder Expansion')
plt.xlabel('Time')
plt.ylabel('P(t)')
plt.title('Simulation of Placeholder Expansion (PEF)')
plt.grid(True)
plt.legend()
plt.show()

# --- Optional: Plot C, S, E, eta to see their influence ---
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(t, C_t(t), color='blue')
plt.ylabel('C(t)')
plt.title('Components of PEF Simulation')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, S_t(t), color='green')
plt.ylabel('S(t)')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, E_t(t), color='red')
plt.ylabel('E(t)')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, eta_t(t), color='purple') # This line now works correctly
plt.ylabel('eta(t)')
plt.xlabel('Time')
plt.grid(True)

plt.tight_layout()
plt.show()
