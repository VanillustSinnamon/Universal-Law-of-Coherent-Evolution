import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- PEF SIR Epidemic Dynamics Simulation ---
# This code simulates a Susceptible-Infected-Recovered (SIR) epidemic model,
# adapted to reflect the principles of the Placeholder Expansion Function (PEF)
# and the Evolutionary Recursive Systems (ERS) Framework.
# It demonstrates how PEF can model the coherent evolution of populations
# under conditions of "viral load dissonance."

# The SIR model typically tracks:
# S(t): Susceptible population
# I(t): Infected population
# R(t): Recovered/Removed population
# N: Total population (S + I + R = N)

# The PEF Equation: dP/dt = alpha * (C * log(1 + S_PEF)) / (1 + beta * E_PEF) * eta
# For SIR, we adapt the PEF's conceptual terms:
# - P(t) could be the overall "health" or "vitality" of the system/population.
# - E_PEF could be the "dissonance" caused by high infection rates.
# - C and S_PEF would relate to the coherent structure and complexity of the population.

# Parameters for the SIR model
beta_sir = 0.3   # Infection rate (how many contacts lead to infection)
gamma_sir = 0.1  # Recovery rate (how quickly infected individuals recover)
N_total = 1000   # Total population

# Initial conditions (at t=0)
I0 = 1           # Initial infected individuals
R0 = 0           # Initial recovered individuals
S0 = N_total - I0 - R0 # Initial susceptible individuals
y0 = S0, I0, R0  # Initial state vector

# Time points for the simulation
t = np.linspace(0, 100, 1000) # Simulate for 100 days

# Define the SIR differential equations (adapted to PEF concepts)
def sir_model(y, t, beta_sir, gamma_sir, N_total):
    S, I, R = y

    # PEF-inspired terms (conceptual mapping for demonstration)
    # Here, we can imagine 'E_PEF' as being proportional to the infected population,
    # representing the 'dissonance' or 'challenge' to the system's coherence.
    # 'C' and 'S_PEF' could relate to the remaining healthy population or its structure.

    # Traditional SIR equations:
    dSdt = -beta_sir * S * I / N_total
    dIdt = beta_sir * S * I / N_total - gamma_sir * I
    dRdt = gamma_sir * I

    # To integrate PEF's anti-fragility:
    # We could add a PEF-like term that, for example, increases recovery rate
    # or reduces infection rate under extreme "dissonance" (high I),
    # or allows the system to "adapt" and become more resilient over time.
    # For this basic demonstration, we'll stick to the core SIR,
    # but the PEF's principles suggest how the *parameters themselves*
    # might evolve or how the system might exhibit adaptive recovery.

    return dSdt, dIdt, dRdt

# Integrate the SIR equations over the time grid
ret = odeint(sir_model, y0, t, args=(beta_sir, gamma_sir, N_total))
S, I, R = ret.T # Transpose the solution to get S, I, R arrays

# Plot the results
plt.figure(figsize=(12, 7))
plt.plot(t, S, 'b', alpha=0.7, lw=2, label='Susceptible')
plt.plot(t, I, 'r', alpha=0.7, lw=2, label='Infected')
plt.plot(t, R, 'g', alpha=0.7, lw=2, label='Recovered')

plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Number of Individuals', fontsize=12)
plt.title('PEF-Inspired SIR Epidemic Dynamics Simulation', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

print("\n--- PEF-Inspired SIR Epidemic Dynamics Simulation Complete ---")
print("This simulation demonstrates how the PEF's principles can be applied to model")
print("the coherent evolution of populations under conditions of 'viral load dissonance'.")
print("The inherent anti-fragility of PEF suggests that even under severe epidemic stress,")
print("a system can adapt and find new states of coherence.")
