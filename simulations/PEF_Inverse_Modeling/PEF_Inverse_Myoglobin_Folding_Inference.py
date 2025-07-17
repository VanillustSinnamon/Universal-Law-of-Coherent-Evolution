"""
PEF_Benchmark_Myoglobin_Folding.py
-----------------------------------
Loads the Myoglobin protein structure from the Protein Data Bank (PDB) and estimates
a folding vitality curve using hydrophobic residue burial as a proxy.

Then reverse-fits the curve using the Universal Law of Coherent Evolution (PEF).
Requires Biopython to load the PDB and extract residue information.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Bio.PDB import PDBParser

# 🧬 STEP 1: Load Myoglobin PDB Structure
pdb_id = "1MBN"
pdb_file = f"{pdb_id}.pdb"

parser = PDBParser(QUIET=True)
structure = parser.get_structure(pdb_id, pdb_file)

# 🧪 STEP 2: Identify Hydrophobic Residues (simplified list)
hydrophobic = ["ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"]

burial_depths = []
residue_indices = []

# STEP 3: Estimate Burial Depth = inverse of average atom distance from origin
for model in structure:
    for chain in model:
        for i, residue in enumerate(chain):
            if residue.get_resname() in hydrophobic:
                coords = [atom.get_coord() for atom in residue if atom.element != 'H']
                if coords:
                    avg_dist = np.mean([np.linalg.norm(coord) for coord in coords])
                    burial_score = 1 / avg_dist  # smaller distance → deeper burial
                    burial_depths.append(burial_score)
                    residue_indices.append(i)

# 🧠 Normalize burial score to form vitality curve (P)
P_data = np.array(burial_depths)
P_data = (P_data - P_data.min()) / (P_data.max() - P_data.min())  # Normalize 0–1
t_data = np.linspace(0, 100, len(P_data))  # Synthetic time axis

dP_obs = np.gradient(P_data, t_data)

# 🧬 STEP 4: Reverse PEF Fit
def reconstructed_dP_dt(P, t, alpha, beta, a, b, c, d, omega):
    C = a * P
    S = b * np.sqrt(np.maximum(P, 0))
    E = c * P**2
    eta = d * np.sin(omega * t) + 1
    numerator = C * np.log(1 + S)
    denominator = 1 + beta * E
    return alpha * (numerator / denominator) * eta

def fit_func(t, alpha, beta, a, b, c, d, omega):
    return reconstructed_dP_dt(P_data, t, alpha, beta, a, b, c, d, omega)

initial_guesses = [1.0, 0.1, 1.0, 1.0, 0.01, 0.5, 0.1]
bounds = ([0]*7, [10]*7)
popt, _ = curve_fit(fit_func, t_data, dP_obs, p0=initial_guesses, bounds=bounds, maxfev=10000)
dP_pred = fit_func(t_data, *popt)

# 🧪 STEP 5: Plot Real Folding Curve
plt.figure(figsize=(10, 5))
plt.plot(t_data, P_data, label='Estimated Vitality Curve (Hydrophobic Burial)', color='purple')
plt.xlabel('Time')
plt.ylabel('Folding Vitality (P)')
plt.title('Real Protein Folding — Myoglobin (PDB: 1MBN)')
plt.grid(True)
plt.legend()
plt.show()

# Plot dP/dt Fit
plt.figure(figsize=(10, 5))
plt.plot(t_data, dP_obs, label='Observed dP/dt', color='blue')
plt.plot(t_data, dP_pred, label='Predicted dP/dt (Reverse PEF)', linestyle='--', color='orange')
plt.xlabel('Time')
plt.ylabel('Rate of Change (dP/dt)')
plt.title('Reverse PEF Fit — Myoglobin Folding')
plt.grid(True)
plt.legend()
plt.show()

# 🧬 Print Fitted Parameters
print("🔬 Fitted PEF Parameters — Myoglobin Folding")
param_names = ['alpha', 'beta', 'a', 'b', 'c', 'd', 'omega']
for name, value in zip(param_names, popt):
    print(f"{name}: {value:.6f}")
