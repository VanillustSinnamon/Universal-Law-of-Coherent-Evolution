import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.stats import bootstrap

# --- 1. Define the Universal PEF Equation for ODE Integration ---
# ***CRITICAL CHANGE: Added P_max as a parameter***
def pef_ode(P, t, alpha, beta, a, c, S_const, eta_const, P_max):
    P_val = np.maximum(P, 1e-9) 
    
    C = a * P_val                          
    S = S_const                            
    E = c * P_val 
    eta = eta_const 

    numerator = C * np.log1p(S) 
    denominator = 1 + beta * E
    denominator = np.maximum(denominator, 1e-10) 

    dPdt = alpha * (numerator / denominator) * eta
    dPdt *= (P_max - P_val) # MODIFIED Saturation Cap: Forces dPdt to zero as P_val approaches P_max
    
    return dPdt

# 🧠 STEP 1: Load Real-World LLM Learning Vitality Curve
raw_data_string = """1.023666835409042, 8.22147651006712
1.0903185129615582, 9.161073825503351
1.1613099287637665, 10.100671140939596
1.2369236462674393, 11.040268456375841
1.2901693959919087, 11.979865771812072
1.3741732452867004, 12.919463087248317
1.463646645105839, 13.859060402684563
1.5589457217838825, 14.798657718120808
1.6604497892949635, 15.738255033557053
1.731927033810127, 16.677852348993284
1.8446940378870695, 17.61744966442953
1.9648033820050377, 18.557046979865774
2.092733131159377, 19.49664429530202
2.1828188408780553, 20.436241610738264
2.3247804253361855, 21.140939597315437
2.476322476141744, 22.31543624161074
2.6375575982407184, 23.25503355704697
2.809290854104074, 24.194630872483216
2.9922057847066217, 25.13422818791946
3.121011019223216, 26.073825503355707
3.32422227204064, 27.013422818791952
3.540664754423506, 27.953020134228183
3.771199961163011, 28.892617449664428
4.01674547959046, 29.832214765100666
4.189653989471853, 30.77181208053691
4.462445348050264, 31.711409395973156
4.7529983464925, 32.651006711409394
5.0628250763862415, 33.8255033557047
5.506924101455843, 35
5.865895883422949, 36.174496644295296
6.380438377718912, 37.3489932885906
6.795873269897965, 38.28859060402685
7.238865890238035, 39.46308724832214
7.7101936098366375, 40.402684563758385
8.212786688739378, 41.577181208053695
8.74752708595732, 42.516778523489926
9.317084811720855, 43.45637583892617
9.923726847117189, 44.39597315436242
10.569867777988526, 45.335570469798654
11.024868000537277, 46.2751677852349
11.987732264009617, 46.2751677852349
13.311349509424128, 46.51006711409396
14.781113046211729, 46.74496644295302
16.413159517012986, 46.97986577181208
18.2266875439303, 47.4496644295302
20.667298381044123, 47.68456375838926
23.43471469193815, 47.91946308724832
26.57456338563535, 48.38926174496644
30.13298107586173, 48.6241610738255
33.460093537949604, 48.85906040268456
37.15456684321175, 49.09395973154362
41.25696288747673, 49.328859060402685
45.81232218051332, 49.56375838926174
50.870658349120006, 49.79865771812081
56.48750724043507, 50.033557046979865
62.72453665411243, 50.26845637583892
69.65022339765741, 50.50335570469799
77.34060509836418, 50.738255033557046
85.8801150260534, 50.9731543624161
95.36250909218873, 51.20805369127517
105.89189520296932, 51.442953020134226
117.58387626773462, 51.677852348993284
130.5851636663541, 52.38255033557047
145.0138209862955, 52.852348993288594
164.44317877354845, 53.32214765100671
182.61288663705116, 53.79194630872483
1988.5462323690488, 58.7248322147651
2162.8248683306215, 59.66442953020134
2352.377498156773, 60.60402684563758
2558.722481000381, 61.77852348993288
2782.972064205545, 62.718120805369125
3089.600875488325, 62.248322147651
3289.1493020358366, 61.54362416107382
3651.8054009715615, 61.308724832214764
4055.586710525908, 62.013422818791945
4410.713479368823, 62.718120805369125
5001.321898545705, 62.95302013422818
5552.3692512367425, 62.48322147651007
5913.472573986669, 63.18791946308724
6430.833065572376, 63.65771812080537
7292.454086598407, 64.12751677852349
8097.076620920228, 64.12751677852349
9376.173290489261, 64.59731543624162
10855.803885527475, 64.59731543624162
12568.931305970302, 64.59731543624162
13664.727522877538, 64.12751677852349
14857.102040683501, 63.89261744966443
16494.062611159887, 63.42281879194631
18315.243234673093, 63.65771812080537
19916.216261122325, 63.89261744966443
22115.251635458306, 64.12751677852349
23551.882467862506, 64.59731543624162
26156.021763678677, 65.30201342281879
27855.145424495993, 65.7718120805369
30930.752223630563, 66.00671140939596
34341.125213274536, 65.7718120805369
38124.84210274518, 65.30201342281879
43220.769565180715, 64.83221476510067
49008.166662702584, 65.06711409395973
56746.00949877807, 65.30201342281879
63011.581262216365, 65.53691275167785
71444.01168399921, 65.53691275167785
81004.89946860094, 65.53691275167785
89961.63751905909, 66.24161073825503
99894.69009510407, 66.47651006711409"""

# Parse the raw data string into numpy arrays
data_lines = raw_data_string.strip().split('\n')
t_data_llm = []
P_data_llm = []
for line in data_lines:
    x_str, y_str = line.split(',')
    t_data_llm.append(float(x_str.strip()))
    P_data_llm.append(float(y_str.strip()) / 100.0) # Convert percentage to 0-1 scale

t_data = np.array(t_data_llm)
P_data = np.array(P_data_llm)

# 🧠 STEP 2: Define Wrapper for Curve Fitting (Integrate PEF)
# ***CRITICAL CHANGE: integrated_pef_model now accepts P_data_local as an argument***
def integrated_pef_model(t, alpha, beta, a, c, S_const, eta_const, P_max, P_data_local): # P_max added here
    P0 = P_data_local[0] # Initial condition from the specific P_data being used for this fit
    # Pass P_max to pef_ode
    P_integrated = odeint(pef_ode, P0, t, args=(alpha, beta, a, c, S_const, eta_const, P_max))
    return P_integrated.T[0]

# 🧠 STEP 3: Fit Parameters (Initial Fit)
# Adjust initial guesses and bounds for the LLM data.
# The LLM curve starts higher and saturates higher than our simulated one.
# Initial guesses might need to be tweaked for this new dataset.
# ***CRITICAL CHANGE: Added P_max to initial_guesses and bounds***
initial_guesses = [0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7] # Added initial guess for P_max (around 0.7 for 70%)
lower_bounds = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.01] # P_max must be positive
upper_bounds = [100.0, 1000.0, 100.0, 100.0, 100.0, 100.0, 1.0] # P_max can go up to 1.0 (100%)

# ***CRITICAL CHANGE: Pass P_data explicitly as an argument to curve_fit's args***
# Also, the lambda function now needs to accept P_max
popt, pcov = curve_fit(
    lambda t, alpha, beta, a, c, S_const, eta_const, P_max_fit: integrated_pef_model(t, alpha, beta, a, c, S_const, eta_const, P_max_fit, P_data),
    t_data,
    P_data, 
    p0=initial_guesses,
    bounds=(lower_bounds, upper_bounds),
    maxfev=100000 
)

# 🧠 STEP 4: Predict P and dP/dt using fitted PEF parameters
# Pass P_max from popt to odeint
P_pred = odeint(pef_ode, P_data[0], t_data, args=tuple(popt)).T[0]
dP_pred = np.gradient(P_pred, t_data)

# --- Calculate Confidence Intervals using Bootstrapping ---
rng = np.random.default_rng(seed=42) # For reproducibility of bootstrap
bootstrap_samples = []
n_resamples = 100 # Number of bootstrap resamples (can increase for more precision, but takes longer)

print(f"Calculating {n_resamples} bootstrap resamples for confidence intervals... This may take a moment.")
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
        # Fit model to resampled data
        # Lambda function needs to accept P_max_fit
        popt_resample, _ = curve_fit(
            lambda t, alpha, beta, a, c, S_const, eta_const, P_max_fit: integrated_pef_model(t, alpha, beta, a, c, S_const, eta_const, P_max_fit, P_resampled),
            t_resampled,
            P_resampled,
            p0=popt, # Use previously fitted popt as initial guess for speed
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000 # Reduced maxfev for bootstrap fits to speed up
        )
        # Predict P_data using the resampled parameters over the original t_data
        # Pass P_max from popt_resample to odeint
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
plt.figure(figsize=(10, 5))
# Plot observed data as percentage for better readability on graph
plt.plot(t_data, P_data * 100, label='GPT-3 In-Context Learning (Observed)', color='purple')
# Plot predicted data as percentage
plt.plot(t_data, P_pred * 100, label='Predicted PEF Fit', linestyle='--', color='orange') 
if len(bootstrap_samples) > 0: 
    # Plot CI as percentage
    plt.fill_between(t_data, ci_lower * 100, ci_upper * 100, color='orange', alpha=0.2, label='95% CI')
plt.xlabel('Number of Examples in Context')
plt.ylabel('Accuracy (%)') # Updated label to reflect percentage
plt.title('GPT-3 In-Context Learning: Vitality Trajectory (PEF Fit with CI)')
plt.xscale('log') 
plt.grid(True)
plt.legend()
plt.show()

# 🧠 STEP 6: Plot Observed vs Predicted dP/dt
# dP_obs_from_data is already in 0-1 scale, no change needed
dP_obs_from_data = np.gradient(P_data, t_data) 
plt.figure(figsize=(10, 5))
plt.plot(t_data, dP_obs_from_data, label='Observed dP/dt (from data)', color='blue')
plt.plot(t_data, dP_pred, label='Predicted dP/dt (from PEF Fit)', linestyle='--', color='orange') 
plt.xlabel('Number of Examples in Context')
plt.ylabel('Rate of Change (dP/dt)')
plt.title('Reverse PEF Fit — GPT-3 In-Context Learning Dynamics')
plt.xscale('log') 
plt.grid(True)
plt.legend()
plt.show()

# 🧠 STEP 7: Display Fitted Parameters
print("🔬 Fitted PEF Parameters — GPT-3 In-Context Learning:")
param_names = ['alpha', 'beta', 'a', 'c', 'S_const', 'eta_const', 'P_max'] # Added P_max to names
for name, value in zip(param_names, popt):
    print(f"{name}: {value:.6f}")
