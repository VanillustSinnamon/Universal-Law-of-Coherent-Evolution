import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.stats import bootstrap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- 1. Define the Universal PEF Equation for ODE Integration ---
# P_max is now a fitted parameter, allowing the model to learn the true saturation point
def pef_ode(P, t, alpha, beta, a, c, S_const, eta_const, P_max):
    # Ensure P_val is never zero or negative to avoid log(0) or division by zero
    P_val = np.maximum(P, 1e-9) 
    
    # Define the functional forms of C, S, E, eta based on previous successful fits for AI learning
    C = a * P_val                          
    S = S_const                            
    E = c * P_val 
    eta = eta_const 

    # Calculate the numerator and denominator of the PEF equation
    numerator = C * np.log1p(S) 
    denominator = 1 + beta * E
    denominator = np.maximum(denominator, 1e-10) 

    dPdt = alpha * (numerator / denominator) * eta
    
    # Saturation Cap: Forces dPdt to zero as P_val approaches P_max
    dPdt *= (P_max - P_val) 
    
    return dPdt

# 🧠 STEP 2: Define Wrapper for Curve Fitting (Integrate PEF)
def integrated_pef_model(t, alpha, beta, a, c, S_const, eta_const, P_max, P_data_local):
    P0 = P_data_local[0] 
    P_integrated = odeint(pef_ode, P0, t, args=(alpha, beta, a, c, S_const, eta_const, P_max))
    return P_integrated.T[0]

# --- GAN Definition and Training ---
print("🧠 STEP 1: Setting up and training a simple GAN...")

# Device configuration (CPU only for simplicity)
device = torch.device('cpu')

# Hyperparameters
latent_dim = 10 # Dimension of the noise vector
data_dim = 2 # We will generate simple 2D points (e.g., from a Gaussian distribution)
num_epochs = 100 # Number of training epochs for the GAN
batch_size = 64
lr = 0.0002 # Learning rate

# --- Simple Data Generation (for the "real" data) ---
# We'll generate a simple 2D Gaussian distribution
def generate_real_data(num_samples):
    mean = torch.tensor([0.0, 0.0])
    cov = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    data = torch.distributions.MultivariateNormal(mean, cov).sample((num_samples,))
    return data.to(device)

# --- Generator Network ---
class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, data_dim),
        )
    def forward(self, input):
        return self.main(input)

# --- Discriminator Network ---
class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(), # Output a probability between 0 and 1
        )
    def forward(self, input):
        return self.main(input)

# Initialize Generator and Discriminator
netG = Generator(latent_dim, data_dim).to(device)
netD = Discriminator(data_dim).to(device)

# Loss function and optimizers
criterion = nn.BCELoss() # Binary Cross-Entropy Loss
optimizerD = optim.Adam(netD.parameters(), lr=lr)
optimizerG = optim.Adam(netG.parameters(), lr=lr)

# Lists to store learning metrics for PEF
discriminator_accuracy_history = [] # This will be our P_data (vitality)

# Training Loop
for epoch in range(num_epochs):
    # --- Train Discriminator ---
    netD.zero_grad()

    # Train with real data
    real_data = generate_real_data(batch_size)
    label_real = torch.full((batch_size, 1), 1.0, dtype=torch.float, device=device) # Label real data as 1
    output_real = netD(real_data)
    errD_real = criterion(output_real, label_real)
    errD_real.backward()

    # Train with fake data
    noise = torch.randn(batch_size, latent_dim, device=device)
    fake_data = netG(noise).detach() # Detach to prevent gradients from flowing to Generator
    label_fake = torch.full((batch_size, 1), 0.0, dtype=torch.float, device=device) # Label fake data as 0
    output_fake = netD(fake_data)
    errD_fake = criterion(output_fake, label_fake)
    errD_fake.backward()

    errD = errD_real + errD_fake
    optimizerD.step()

    # --- Train Generator ---
    netG.zero_grad()
    label_real_for_gen = torch.full((batch_size, 1), 1.0, dtype=torch.float, device=device) # Generator wants Discriminator to think fakes are real
    output_gen = netD(netG(torch.randn(batch_size, latent_dim, device=device)))
    errG = criterion(output_gen, label_real_for_gen)
    errG.backward()
    optimizerG.step()

    # --- Collect Vitality (Discriminator Accuracy) for PEF ---
    # We'll use the discriminator's accuracy on both real and fake data as a proxy for GAN "vitality"
    # A higher accuracy means the GAN is learning to distinguish better, or the generator is getting better
    # and challenging the discriminator more.
    with torch.no_grad():
        # Test discriminator on a fresh batch of real and fake data
        test_real_data = generate_real_data(batch_size)
        test_noise = torch.randn(batch_size, latent_dim, device=device)
        test_fake_data = netG(test_noise)

        output_real_test = netD(test_real_data)
        output_fake_test = netD(test_fake_data)

        # Calculate accuracy for real and fake predictions
        acc_real = ((output_real_test > 0.5) == 1.0).float().mean().item()
        acc_fake = ((output_fake_test <= 0.5) == 1.0).float().mean().item() # Discriminator wants fake to be < 0.5

        # Average accuracy of discriminator on real and fake data
        # This metric represents how well the GAN is learning to discriminate/generate
        # A value closer to 0.5 means the GAN is in equilibrium (discriminator can't tell real/fake apart)
        # A value closer to 1.0 means discriminator is winning easily (generator is bad)
        # We want to model the overall *learning process*, so we can use (acc_real + acc_fake) / 2
        # Or, more simply, how well the discriminator is doing overall.
        # Let's use 1 - abs(output_fake_test.mean() - 0.5) * 2 to represent how close the fake data is to real
        # Or simply the discriminator's accuracy on real data (as a proxy for how well it's learning to identify real)
        
        # For PEF, we need a metric that increases as "coherence" or "vitality" increases.
        # Let's use the average of discriminator's accuracy on real and fake data
        # A higher value means the discriminator is performing better overall.
        discriminator_total_acc = (acc_real + acc_fake) / 2.0
        discriminator_accuracy_history.append(discriminator_total_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], D_Loss: {errD.item():.4f}, G_Loss: {errG.item():.4f}, D_Acc: {discriminator_total_acc:.4f}")

# Convert history to numpy arrays for PEF
P_data_gan = np.array(discriminator_accuracy_history)
t_data_gan = np.arange(1, len(P_data_gan) + 1) # Epochs as time

# Ensure P_data is within (0, 1) range
P_data = np.clip(P_data_gan, 1e-9, 1.0 - 1e-9) 
t_data = t_data_gan

print(f"GAN training complete. Collected {len(P_data)} data points for PEF analysis.")

# 🧠 STEP 3: Fit Parameters (Initial Fit)
print("🧠 STEP 3: Fitting PEF parameters to the GAN learning curve...")

# Initial guesses and bounds for PEF parameters.
# P_max for GAN discriminator accuracy might be around 0.9-0.95
initial_guesses = [0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9] 
lower_bounds = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.01] 
upper_bounds = [100.0, 1000.0, 100.0, 100.0, 100.0, 100.0, 1.0] 

popt, pcov = curve_fit(
    lambda t, alpha, beta, a, c, S_const, eta_const, P_max_fit: integrated_pef_model(t, alpha, beta, a, c, S_const, eta_const, P_max_fit, P_data),
    t_data,
    P_data, 
    p0=initial_guesses,
    bounds=(lower_bounds, upper_bounds),
    maxfev=100000 
)

# 🧠 STEP 4: Predict P and dP/dt using fitted PEF parameters
print("🧠 STEP 4: Predicting P and dP/dt using fitted PEF parameters...")
P_pred = odeint(pef_ode, P_data[0], t_data, args=tuple(popt)).T[0]
dP_pred = np.gradient(P_pred, t_data)

# --- Calculate Confidence Intervals using Bootstrapping ---
rng = np.random.default_rng(seed=42) 
bootstrap_samples = []
n_resamples = 50 
print(f"🧠 Calculating {n_resamples} bootstrap resamples for confidence intervals... This may take a moment.")

for _ in range(n_resamples):
    resample_indices = rng.integers(0, len(t_data), size=len(t_data))
    t_resampled = t_data[resample_indices]
    P_resampled = P_data[resample_indices]
    
    sort_indices = np.argsort(t_resampled)
    t_resampled = t_resampled[sort_indices]
    P_resampled = P_resampled[sort_indices]

    try:
        popt_resample, _ = curve_fit(
            lambda t, alpha, beta, a, c, S_const, eta_const, P_max_fit: integrated_pef_model(t, alpha, beta, a, c, S_const, eta_const, P_max_fit, P_resampled),
            t_resampled,
            P_resampled,
            p0=popt, 
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000 
        )
        bootstrap_samples.append(odeint(pef_ode, P_data[0], t_data, args=tuple(popt_resample)).T[0])
    except RuntimeError as e:
        print(f"Warning: Bootstrap fit failed for a resample: {e}")
        continue

if len(bootstrap_samples) > 0: 
    bootstrap_samples = np.array(bootstrap_samples)
    ci_lower = np.percentile(bootstrap_samples, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_samples, 97.5, axis=0)
else:
    print("Warning: No successful bootstrap samples. Confidence interval will not be plotted.")
    ci_lower = P_pred 
    ci_upper = P_pred


# 🧠 STEP 5: Plot Learning Curve (Observed vs Predicted P with CI)
print("🧠 STEP 5: Generating plots...")
plt.figure(figsize=(10, 5))
plt.plot(t_data, P_data * 100, label='GAN Discriminator Accuracy (Observed)', color='purple')
plt.plot(t_data, P_pred * 100, label='Predicted PEF Fit', linestyle='--', color='orange') 
if len(bootstrap_samples) > 0: 
    plt.fill_between(t_data, ci_lower * 100, ci_upper * 100, color='orange', alpha=0.2, label='95% CI')
plt.xlabel('Epochs') 
plt.ylabel('Discriminator Accuracy (%)') 
plt.title('GAN Learning: Vitality Trajectory (PEF Fit with CI)')
plt.grid(True)
plt.legend()
plt.show()

# 🧠 STEP 6: Plot Observed vs Predicted dP/dt
dP_obs_from_data = np.gradient(P_data, t_data) 
plt.figure(figsize=(10, 5))
plt.plot(t_data, dP_obs_from_data, label='Observed dP/dt (from data)', color='blue')
plt.plot(t_data, dP_pred, label='Predicted dP/dt (from PEF Fit)', linestyle='--', color='orange') 
plt.xlabel('Epochs') 
plt.ylabel('Rate of Change (dP/dt)')
plt.title('Reverse PEF Fit — GAN Learning Dynamics')
plt.grid(True)
plt.legend()
plt.show()

# 🧠 STEP 7: Display Fitted Parameters
print("🔬 Fitted PEF Parameters — GAN Learning:")
param_names = ['alpha', 'beta', 'a', 'c', 'S_const', 'eta_const', 'P_max'] 
for name, value in zip(param_names, popt):
    print(f"{name}: {value:.6f}")
