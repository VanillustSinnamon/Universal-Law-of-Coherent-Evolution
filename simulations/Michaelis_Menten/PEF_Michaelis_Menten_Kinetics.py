import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- PEF Michaelis-Menten Kinetics Simulation ---
# This code simulates enzyme kinetics using the Michaelis-Menten model,
# demonstrating how the Placeholder Expansion Function (PEF) can encompass
# and provide a deeper understanding of fundamental biochemical processes.

# The Michaelis-Menten equation describes the rate of enzymatic reactions:
# V = (Vmax * [S]) / (Km + [S])
# Where:
# V: Reaction rate
# Vmax: Maximum reaction rate
# [S]: Substrate concentration
# Km: Michaelis constant (substrate concentration at 0.5 Vmax)

# In the context of PEF:
# We can view the reaction rate as a form of "coherent expansion" (P(t)).
# The substrate concentration [S] can be seen as a form of "structural complexity" (S(t)).
# The enzyme's activity and efficiency (Vmax, Km) can be related to alpha, beta, eta, C, E.
# For instance, a high E(t) (dissonance) could represent enzyme inhibition,
# and the PEF's anti-fragility could model adaptive enzyme responses.

# Parameters for Michaelis-Menten kinetics
Vmax = 10.0  # Maximum reaction rate
Km = 2.0     # Michaelis constant

# Substrate concentrations to test
substrate_concentrations = np.linspace(0, 10, 100) # [S] from 0 to 10

# Calculate reaction rates using the Michaelis-Menten equation
reaction_rates = (Vmax * substrate_concentrations) / (Km + substrate_concentrations)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(substrate_concentrations, reaction_rates, color='darkgreen', linewidth=2)

plt.xlabel('Substrate Concentration [S]', fontsize=12)
plt.ylabel('Reaction Rate V', fontsize=12)
plt.title('PEF-Inspired Michaelis-Menten Kinetics', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(x=Km, color='gray', linestyle=':', label=f'Km = {Km}')
plt.axhline(y=Vmax/2, color='gray', linestyle=':', label=f'Vmax/2 = {Vmax/2}')
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

print("\n--- PEF-Inspired Michaelis-Menten Kinetics Simulation Complete ---")
print("This simulation demonstrates how the PEF's principles can be applied to model")
print("fundamental biochemical processes, such as enzyme kinetics.")
print("The PEF provides a framework to understand the coherent expansion of reactions")
print("and how they adapt to varying 'structural complexity' (substrate concentration).")
