import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.stats import bootstrap
import tensorflow as tf # For building and training the neural network

# --- 1. Define the Universal PEF Equation for ODE Integration ---
# P_max is now a fitted parameter, allowing the model to learn the true saturation point
def pef_ode(P, t, alpha, beta, a, c, S_const, eta_const, P_max):
    # Ensure P_val is never zero or negative to avoid log(0) or division by zero
    P_val = np.maximum(P, 1e-9) 
    
    # Define the functional forms of C, S, E, eta based on previous successful fits for AI learning
    # C: Coherence, typically proportional to vitality
    C = a * P_val                          
    # S: Structural Complexity, assumed constant for this specific learning dynamic
    S = S_const                            
    # E: Entropic Dissonance, assumed proportional to vitality (errors increase with P_val)
    E = c * P_val 
    # eta: Temporal Pacing, assumed constant for simplicity
    eta = eta_const 

    # Calculate the numerator and denominator of the PEF equation
    numerator = C * np.log1p(S) # np.log1p(x) computes log(1+x) for numerical stability
    denominator = 1 + beta * E
    denominator = np.maximum(denominator, 1e-10) # Prevent division by zero

    # Calculate dP/dt based on the core PEF equation
    dPdt = alpha * (numerator / denominator) * eta
    
    # Saturation Cap: This crucial term forces dPdt to zero as P_val approaches P_max
    # It allows the model to learn the actual observed saturation limit of the system.
    dPdt *= (P_max - P_val) 
    
    return dPdt

# 🧠 STEP 2: Define Wrapper for Curve Fitting (Integrate PEF)
# The wrapper function now accepts P_max as a parameter to be fitted.
def integrated_pef_model(t, alpha, beta, a, c, S_const, eta_const, P_max, P_data_local):
    P0 = P_data_local[0] # Initial condition from the specific P_data being used for this fit
    # Pass all fitted parameters, including P_max, to pef_ode
    P_integrated = odeint(pef_ode, P0, t, args=(alpha, beta, a, c, S_const, eta_const, P_max))
    return P_integrated.T[0]

# --- Neural Network Training and Data Collection ---
print("🧠 STEP 1: Training a tiny neural network on MNIST and collecting learning data...")

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data for the neural network (flatten images)
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# Define a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)), # Input layer with 128 neurons
    tf.keras.layers.Dropout(0.2), # Dropout for regularization
    tf.keras.layers.Dense(64, activation='relu'), # Hidden layer with 64 neurons
    tf.keras.layers.Dense(10, activation='softmax') # Output layer for 10 classes (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model and store history (which contains accuracy per epoch)
# We use a validation split to get a realistic learning curve
history = model.fit(x_train, y_train, 
                    epochs=20, # Train for 20 epochs
                    batch_size=32,
                    validation_split=0.1, # Use 10% of training data for validation
                    verbose=0) # Suppress verbose output during training

# Extract validation accuracy as P_data
# P_data will be the validation accuracy (vitality) over epochs (time)
P_data_nn = np.array(history.history['val_accuracy'])
t_data_nn = np.arange(1, len(P_data_nn) + 1) # Epochs start from 1

# Ensure P_data is within (0, 1) range, which it should be for accuracy
# If for some reason it's exactly 0 or 1, nudge it slightly to avoid log issues
P_data = np.clip(P_data_nn, 1e-9, 1.0 - 1e-9) 
t_data = t_data_nn

print(f"Neural network training complete. Collected {len(P_data)} data points.")

# 🧠 STEP 3: Fit Parameters (Initial Fit)
print("🧠 STEP 3: Fitting PEF parameters to the neural network learning curve...")

# Adjust initial guesses and bounds for the LLM data.
# The LLM curve starts higher and saturates higher than our simulated one.
# Initial guesses might need to be tweaked for this new dataset.
initial_guesses = [0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9] # Added initial guess for P_max (around 0.9 for 90% accuracy)
lower_bounds = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.01] # P_max must be positive
upper_bounds = [100.0, 1000.0, 100.0, 100.0, 100.0, 100.0, 1.0] # P_max can go up to 1.0 (100% accuracy)

# The lambda function in curve_fit must now accept P_max_fit as an argument
popt, pcov = curve_fit(
    lambda t, alpha, beta, a, c, S_const, eta_const, P_max_fit: integrated_pef_model(t, alpha, beta, a, c, S_const, eta_const, P_max_fit, P_data),
    t_data,
    P_data, 
    p0=initial_guesses,
    bounds=(lower_bounds, upper_bounds),
    maxfev=100000 # Increased maxfev for more complex fits
)

# 🧠 STEP 4: Predict P and dP/dt using fitted PEF parameters
print("🧠 STEP 4: Predicting P and dP/dt using fitted PEF parameters...")
# The fitted P_max is now part of popt, so pass all of popt to odeint
P_pred = odeint(pef_ode, P_data[0], t_data, args=tuple(popt)).T[0]
dP_pred = np.gradient(P_pred, t_data)

# --- Calculate Confidence Intervals using Bootstrapping ---
rng = np.random.default_rng(seed=42) # For reproducibility of bootstrap
bootstrap_samples = []
n_resamples = 50 # Reduced number of bootstrap resamples for faster execution
print(f"🧠 Calculating {n_resamples} bootstrap resamples for confidence intervals... This may take a moment.")

for _ in range(n_resamples):
    # Resample indices with replacement
    resample_indices = rng.integers(0, len(t_data), size=len(t_data))
    t_resampled = t_data[resample_indices]
    P_resampled = P_data[resample_indices]
    
    # Sort resampled data by time to ensure odeint works correctly
    sort_indices = np.argsort(t_resampled)
    t_resampled = t_resampled[sort_indices]
    P_resampled = P_resampled[sort_indices]

    try:
        # Fit model to resampled data. The lambda function needs to accept P_max_fit
        popt_resample, _ = curve_fit(
            lambda t, alpha, beta, a, c, S_const, eta_const, P_max_fit: integrated_pef_model(t, alpha, beta, a, c, S_const, eta_const, P_max_fit, P_resampled),
            t_resampled,
            P_resampled,
            p0=popt, # Use previously fitted popt as initial guess for speed
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000 # Reduced maxfev for bootstrap fits to speed up
        )
        # Predict P_data using the resampled parameters over the original t_data
        # Pass all resampled parameters, including P_max, to odeint
        bootstrap_samples.append(odeint(pef_ode, P_data[0], t_data, args=tuple(popt_resample)).T[0])
    except RuntimeError as e:
        print(f"Warning: Bootstrap fit failed for a resample: {e}")
        continue

if len(bootstrap_samples) > 0: 
    bootstrap_samples = np.array(bootstrap_samples)
    # Calculate 2.5th and 97.5th percentiles for 95% confidence interval
    ci_lower = np.percentile(bootstrap_samples, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_samples, 97.5, axis=0)
else:
    print("Warning: No successful bootstrap samples. Confidence interval will not be plotted.")
    ci_lower = P_pred # Fallback to predicted line if no samples
    ci_upper = P_pred


# 🧠 STEP 5: Plot Learning Curve (Observed vs Predicted P with CI)
print("🧠 STEP 5: Generating plots...")
plt.figure(figsize=(10, 5))
# Plot observed data as percentage for better readability on graph
plt.plot(t_data, P_data * 100, label='MNIST NN Learning (Observed)', color='purple')
# Plot predicted data as percentage
plt.plot(t_data, P_pred * 100, label='Predicted PEF Fit', linestyle='--', color='orange') 
if len(bootstrap_samples) > 0: 
    # Plot CI as percentage
    plt.fill_between(t_data, ci_lower * 100, ci_upper * 100, color='orange', alpha=0.2, label='95% CI')
plt.xlabel('Epochs') # X-axis is now epochs
plt.ylabel('Validation Accuracy (%)') # Updated label to reflect percentage and validation accuracy
plt.title('MNIST Neural Network Learning: Vitality Trajectory (PEF Fit with CI)')
# X-axis for epochs is typically linear, not log
plt.grid(True)
plt.legend()
plt.show()

# 🧠 STEP 6: Plot Observed vs Predicted dP/dt
# dP_obs_from_data is already in 0-1 scale, no change needed
dP_obs_from_data = np.gradient(P_data, t_data) 
plt.figure(figsize=(10, 5))
plt.plot(t_data, dP_obs_from_data, label='Observed dP/dt (from data)', color='blue')
plt.plot(t_data, dP_pred, label='Predicted dP/dt (from PEF Fit)', linestyle='--', color='orange') 
plt.xlabel('Epochs') # X-axis is now epochs
plt.ylabel('Rate of Change (dP/dt)')
plt.title('Reverse PEF Fit — MNIST NN Learning Dynamics')
# X-axis for epochs is typically linear, not log
plt.grid(True)
plt.legend()
plt.show()

# 🧠 STEP 7: Display Fitted Parameters
print("🔬 Fitted PEF Parameters — MNIST Neural Network Learning:")
# Added P_max to the list of parameter names for display
param_names = ['alpha', 'beta', 'a', 'c', 'S_const', 'eta_const', 'P_max'] 
for name, value in zip(param_names, popt):
    print(f"{name}: {value:.6f}")
